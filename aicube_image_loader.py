"""
AICube Image Loader - Satellite Image Discovery Library

Provides a simple interface for discovering and loading satellite images
delivered to AICube containers. Handles config parsing, image discovery,
metadata loading, wait-for-arrival patterns, GeoTIFF reading, projection
conversion, per-band slicing, and output-directory management.

Image naming convention:
    {outputPrefix}_{index}.{ext}            e.g. img_0.png, img_1.png
    {outputPrefix}_{index}_metadata.json    e.g. img_0_metadata.json

Required dependencies: Python stdlib only (os, glob, json, re, time,
datetime, logging, dataclasses).

Optional dependencies (only needed for specific helpers):
    rasterio + pyproj    for read_image() and pixel_to_latlon()
    numpy                for band() / bands() (any caller passing arrays)

Usage:
    from aicube_image_loader import AICubeImageLoader

    loader = AICubeImageLoader()
    log = AICubeImageLoader.get_logger("my_app")
    loader.require_bands(["blue", "green", "red", "nir-2"])

    # Idiomatic loop — handles wait, sidecar check, GeoTIFF read, errors
    for record in loader.iter_images(validate_bands=["blue", "green", "red", "nir-2"]):
        log.info(f"[{record.index}/{record.total}] {record.name}")
        green = loader.band(record.data, "green")
        nir   = loader.band(record.data, "nir-2")
        # ... compute, then write outputs
        loader.write_json("results.json", {...})
"""

import datetime
import glob
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Image file extensions we look for in the data directory.
_IMAGE_EXTENSIONS = ('png', 'jpg', 'jpeg', 'tiff', 'tif', 'raw', 'bin')

# Match a filename of the form "{prefix}_{index}.{ext}" and extract
# (prefix, index). Example: "img_3.png" -> ("img", 3).
_FILENAME_RE = re.compile(r'^(.+)_(\d+)$')

# Dashboard band id → Sentinel-2 band name. Mirrors the table in
# tools/sentinel_hub_integration/sentinel_imagery.py and the dashboard's
# BAND_CATALOG. Used by selected_bands_s2() so apps that reason in S2 names
# (B02, B03, ...) can keep their existing index-by-name lookups working.
_DASHBOARD_TO_S2_BAND = {
    'coastal-blue': 'B01',
    'blue':         'B02',
    'green':        'B03',
    'red':          'B04',
    'red-edge':     'B05',
    'nir-1':        'B06',
    'nir-2':        'B08',
    'water-vapor':  'B09',
}

# Reverse for transparent S2-name → dashboard-id lookup in band()/bands().
_S2_TO_DASHBOARD_BAND = {v: k for k, v in _DASHBOARD_TO_S2_BAND.items()}

# Default fallback when player.config has no `bands` field at all (matches
# the dashboard's RGB default).
_DEFAULT_BANDS = ['red', 'green', 'blue']

_DEFAULT_INPUT_DIR = '/opt/ilc_player/data'
_DEFAULT_OUTPUT_DIR = '/opt/ilc_player/results'


@dataclass
class ImageRecord:
    """One image yielded by AICubeImageLoader.iter_images().

    Bundles everything the per-image processing step usually needs: the path,
    derived names, sidecar metadata, and (when ``load=True``) the decoded
    raster + geo dict. ``data`` and ``geo`` are ``None`` when ``load=False``
    or when rasterio is unavailable.
    """
    path: str
    name: str
    aoi_name: str
    index: int                                 # 1-based, matches processed_count
    total: int                                 # total expected guidance targets
    metadata: Optional[dict] = None            # sidecar JSON, may be None
    data: Optional[Any] = None                 # numpy ndarray (bands, H, W) or None
    geo: Optional[dict] = None                 # {transform, crs, bounds, ...} or None


class AICubeImageLoader:
    """Discovers and tracks satellite images delivered to AICube containers.

    Parses the task configuration to determine expected image count,
    finds images in AOI subdirectories, tracks which images have
    been returned, and provides access to companion metadata JSON files.

    Args:
        data_dir: Directory containing AOI subdirectories with images.
            Defaults to ILC_INPUT_DIR env var or /opt/ilc_player/data.
        output_dir: Directory results are written to.
            Defaults to ILC_OUTPUT_DIR env var or /opt/ilc_player/results.
        config_path: Path to player.config JSON file.
            Defaults to {data_dir}/player.config.
        max_wait: Default maximum wait time in seconds for blocking calls.
        poll_interval: Default polling interval in seconds for blocking calls.
    """

    def __init__(self, data_dir=None, output_dir=None, config_path=None,
                 max_wait=300, poll_interval=5):
        self.data_dir = data_dir or os.environ.get('ILC_INPUT_DIR', _DEFAULT_INPUT_DIR)
        self._output_dir = output_dir or os.environ.get('ILC_OUTPUT_DIR', _DEFAULT_OUTPUT_DIR)
        self.max_wait = max_wait
        self.poll_interval = poll_interval

        # Parse task config
        config_path = config_path or os.path.join(self.data_dir, 'player.config')
        try:
            with open(config_path) as f:
                self.config = json.load(f)
            logger.info(f"Loaded task config from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using empty config")
            self.config = {}

        # Sum guidance targets across all AOIs
        self.total_guidance_targets = 0
        for aoi in self.config.get('aoi_exec', []):
            self.total_guidance_targets += aoi.get('guidanceTargets', 0)
        logger.info(f"Expecting {self.total_guidance_targets} total images")

        # Internal tracking
        self._returned_images = set()   # image_path
        self._metadata_cache = {}       # image_path -> metadata dict

    # ------------------------------------------------------------------
    # Band introspection
    # ------------------------------------------------------------------

    @property
    def selected_bands(self):
        """Dashboard band ids selected for this task, in stored order.

        Falls back to ['red', 'green', 'blue'] when player.config has no
        `bands` field — matches the dashboard's RGB default.

        Returns:
            list[str]: Dashboard band ids (e.g. ['blue', 'green', 'red', 'nir-2']).
        """
        bands = self.config.get('bands') or _DEFAULT_BANDS
        return list(bands)

    @property
    def selected_bands_s2(self):
        """Selected bands translated to Sentinel-2 names (e.g. B02, B03, B04).

        Useful for apps that work in Sentinel-2 names internally — keeps
        existing `available_bands.index('B04')` style lookups working when
        you pass this list straight in.

        Returns:
            list[str]: Sentinel-2 band names in the same order as
                `selected_bands`. Unknown ids pass through unchanged so the
                caller can detect bad input rather than have it silently mapped.
        """
        return [_DASHBOARD_TO_S2_BAND.get(b, b) for b in self.selected_bands]

    def require_bands(self, required):
        """Verify the listed dashboard band ids are all present in player.config.

        Order doesn't matter — only presence. Returns a `{band_id: index}` dict
        the caller can use to slice into the captured TIFF, since the data
        array's channel order matches `selected_bands` exactly.

        Args:
            required: Iterable of dashboard band ids the app needs (e.g.
                ['blue', 'green', 'red', 'nir-2']).

        Returns:
            dict[str, int]: Mapping from each required band id to its index
                in the captured image's band axis.

        Raises:
            ValueError: If any required band is missing. The message names
                every missing band so it's easy to spot in console.txt.
        """
        bands = self.selected_bands
        missing = [b for b in required if b not in bands]
        if missing:
            raise ValueError(
                f"Required bands {missing} are missing from player.config. "
                f"Selected: {bands}. Re-create the task selecting the "
                f"required bands."
            )
        return {b: bands.index(b) for b in required}

    def _band_index(self, band_id):
        """Resolve a dashboard id OR Sentinel-2 name to its array-axis index.

        Centralises the dual-naming so band()/bands() accept either scheme.

        Raises:
            ValueError: when band_id is neither in selected_bands nor maps
                back to one via the S2 → dashboard-id table.
        """
        bands = self.selected_bands
        if band_id in bands:
            return bands.index(band_id)
        dashboard_id = _S2_TO_DASHBOARD_BAND.get(band_id)
        if dashboard_id and dashboard_id in bands:
            return bands.index(dashboard_id)
        raise ValueError(
            f"Band '{band_id}' is not in selected bands {bands}. "
            f"Did you forget to add it in the dashboard?"
        )

    # ------------------------------------------------------------------
    # Discovery / iteration
    # ------------------------------------------------------------------

    @property
    def processed_count(self):
        """Number of images returned so far."""
        return len(self._returned_images)

    @property
    def all_images_received(self):
        """True if all expected images have been returned."""
        return self.total_guidance_targets > 0 and self.processed_count >= self.total_guidance_targets

    def get_next_image(self):
        """Return the next unprocessed image path, or None if no new images.

        Non-blocking. Scans the data directory for image files that haven't
        been returned yet. Images are returned in index order.

        Returns:
            str: Path to the next image, or None if no new images available.
        """
        for image_path in self._discover_images():
            if image_path not in self._returned_images:
                self._mark_returned(image_path)
                return image_path
        return None

    def wait_for_next_image(self, timeout=None, poll_interval=None):
        """Block until the next image arrives or timeout is reached.

        Args:
            timeout: Max seconds to wait. Defaults to instance max_wait.
            poll_interval: Seconds between checks. Defaults to instance poll_interval.

        Returns:
            str: Path to the next image, or None if timeout reached.
        """
        timeout = timeout if timeout is not None else self.max_wait
        poll_interval = poll_interval if poll_interval is not None else self.poll_interval
        waited = 0

        while waited < timeout:
            logger.info(f"Waiting for next image... ({waited}/{timeout}s)")
            image_path = self.get_next_image()
            if image_path is not None:
                logger.info(f"Found new image: {os.path.basename(image_path)}")
                return image_path
            time.sleep(poll_interval)
            waited += poll_interval

        logger.info(f"Timed out waiting for image after {timeout}s")
        return None

    def reset(self):
        """Reset all tracking state so images can be reprocessed.

        Clears the returned images set and metadata cache, allowing
        get_next_image() and related methods to return all images again.
        """
        self._returned_images.clear()
        self._metadata_cache.clear()
        logger.info("Image loader reset - all images available for reprocessing")

    def get_all_new_images(self):
        """Return all currently available unprocessed images.

        Non-blocking. Returns a list of all image paths that haven't
        been returned yet.

        Returns:
            list[str]: List of image paths (may be empty).
        """
        new_images = []
        for image_path in self._discover_images():
            if image_path not in self._returned_images:
                self._mark_returned(image_path)
                new_images.append(image_path)
        return new_images

    def iter_images(self, load=True, validate_bands=None, dtype="float32"):
        """Yield ImageRecord for each image as it arrives.

        Replaces the wait-loop boilerplate. Stops when all expected images
        are received OR a wait_for_next_image() call times out. Images that
        fail the optional sidecar band check or fail to decode are skipped
        (logged at WARNING / ERROR) so the iterator delivers only usable records.

        Args:
            load: When True (default), each record's `data` and `geo` are
                populated by reading the GeoTIFF via rasterio. Set False for
                metadata-only iteration.
            validate_bands: Optional iterable of dashboard band ids. For each
                image whose sidecar exposes a `bands_saved` field, verify the
                listed bands are all present; skip if any are missing.
            dtype: numpy dtype to coerce the loaded data to. Default float32.

        Yields:
            ImageRecord: one per usable image.
        """
        while not self.all_images_received:
            path = self.wait_for_next_image()
            if path is None:
                logger.info(
                    f"Timed out waiting for image after {self.max_wait}s. "
                    f"Processed {self.processed_count}/{self.total_guidance_targets}."
                )
                return

            if validate_bands is not None:
                ok, missing = self.verify_sidecar_bands(path, validate_bands)
                if not ok:
                    logger.warning(
                        f"Skipping {self.image_name(path)}: sidecar reports "
                        f"missing bands {missing}. Re-run task with image_format=tiff."
                    )
                    continue

            record = ImageRecord(
                path=path,
                name=self.image_name(path),
                aoi_name=self.get_aoi_name(path),
                index=self.processed_count,
                total=self.total_guidance_targets,
                metadata=self.get_metadata(path),
            )

            if load:
                try:
                    record.data, record.geo = self.read_image(path, dtype=dtype)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
                    continue

            yield record

    # ------------------------------------------------------------------
    # Metadata sidecar
    # ------------------------------------------------------------------

    def get_metadata(self, image_path):
        """Get the metadata JSON for an image.

        Looks for a companion metadata file with the same base name
        and `_metadata.json` suffix. For example, `img_0.png` pairs
        with `img_0_metadata.json` in the same directory.

        Args:
            image_path: Path to the image file.

        Returns:
            dict: Parsed metadata JSON, or None if metadata file not found.
        """
        if image_path in self._metadata_cache:
            return self._metadata_cache[image_path]

        metadata_path = self._find_metadata_path(image_path)
        if metadata_path is None:
            return None

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            self._metadata_cache[image_path] = metadata
            return metadata
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read metadata {metadata_path}: {e}")
            return None

    def get_capture_time(self, image_path):
        """Parsed UTC capture time, or None if absent / unparseable.

        Reads the `capture_date` sidecar field (ISO 8601 with 'Z').
        """
        meta = self.get_metadata(image_path) or {}
        raw = meta.get("capture_date")
        if not raw:
            return None
        try:
            # Python <3.11 doesn't accept the trailing 'Z' in fromisoformat.
            return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            logger.warning(f"Unparseable capture_date '{raw}' in {image_path}")
            return None

    def get_bbox(self, image_path):
        """Bounding box as ``(min_lng, min_lat, max_lng, max_lat)`` or None.

        Reads the sidecar `bbox` object.
        """
        meta = self.get_metadata(image_path) or {}
        bbox = meta.get("bbox")
        if not bbox:
            return None
        try:
            return (bbox["min_lng"], bbox["min_lat"], bbox["max_lng"], bbox["max_lat"])
        except KeyError:
            return None

    def get_polygon(self, image_path):
        """Capture polygon as a list of ``(lon, lat)`` tuples or None.

        Reads the sidecar `polygon` field, which is stored as a list of
        ``[lon, lat]`` pairs.
        """
        meta = self.get_metadata(image_path) or {}
        poly = meta.get("polygon")
        if not poly:
            return None
        return [(p[0], p[1]) for p in poly]

    def get_provider(self, image_path):
        """Sidecar `provider` (e.g. 'orbitlab') or None."""
        meta = self.get_metadata(image_path) or {}
        return meta.get("provider")

    def get_satellite_id(self, image_path):
        """Sidecar `satellite_id` (e.g. 'sentinel2', 'moi1') or None."""
        meta = self.get_metadata(image_path) or {}
        return meta.get("satellite_id")

    def get_cloud_coverage(self, image_path):
        """Sidecar `max_cloud_coverage` (int %) or None.

        Sentinel-2 sandbox captures populate this from the user's request
        threshold; flight imagery may set a measured value once available.
        """
        meta = self.get_metadata(image_path) or {}
        val = meta.get("max_cloud_coverage")
        return int(val) if val is not None else None

    def verify_sidecar_bands(self, image_path, required):
        """Return ``(ok, missing)`` for the sidecar's `bands_saved` field.

        ``ok`` is True when the sidecar is absent (we can't check) or when
        every required dashboard band id appears in `bands_saved`. ``missing``
        is the list of bands the sidecar reports as not saved (empty when ok).
        """
        meta = self.get_metadata(image_path)
        if not meta:
            return True, []
        saved = meta.get("bands_saved")
        if saved is None:
            return True, []
        missing = [b for b in required if b not in saved]
        return (len(missing) == 0), missing

    def require_metadata_keys(self, image_path, keys):
        """Assert the sidecar contains all listed top-level keys; return it.

        Raises ValueError listing the missing keys when any are absent.
        """
        meta = self.get_metadata(image_path) or {}
        missing = [k for k in keys if k not in meta]
        if missing:
            raise ValueError(
                f"Sidecar for {os.path.basename(image_path)} is missing keys "
                f"{missing}. Present keys: {sorted(meta.keys())}."
            )
        return meta

    # ------------------------------------------------------------------
    # Path / naming helpers
    # ------------------------------------------------------------------

    def image_name(self, image_path, with_aoi=False):
        """Return the image's base filename without extension.

        With ``with_aoi=True`` prefixes the AOI directory name (e.g.
        ``aoi1/img_3``), useful when the same prefix appears in multiple AOIs.
        """
        name = self._extract_prefix(image_path)
        if with_aoi:
            return f"{self.get_aoi_name(image_path)}/{name}"
        return name

    def get_aoi_name(self, image_path):
        """Return the immediate parent directory name (the AOI subdir)."""
        return os.path.basename(os.path.dirname(image_path))

    def images_by_aoi(self):
        """Group all currently-discovered images by AOI directory name.

        Returns:
            dict[str, list[str]]: AOI-name → sorted list of image paths.
        """
        groups = {}
        for path in self._discover_images():
            aoi = self.get_aoi_name(path)
            groups.setdefault(aoi, []).append(path)
        return groups

    # ------------------------------------------------------------------
    # Output directory + writers
    # ------------------------------------------------------------------

    @property
    def output_dir(self):
        """Result directory path (no side effects).

        Defaults to ILC_OUTPUT_DIR env var or /opt/ilc_player/results.
        Use ensure_output_dir() to create it before writing.
        """
        return self._output_dir

    def ensure_output_dir(self):
        """Create output_dir if it doesn't exist; return its path."""
        os.makedirs(self._output_dir, exist_ok=True)
        return self._output_dir

    def output_path_for(self, image_path, suffix=".png", subdir=None):
        """Build an output path mirroring an input image's stem.

        ``output_path_for("…/aoi1/img_3.tif", "_annotated.png")``
        → ``{output_dir}/img_3_annotated.png``.

        With ``subdir=...`` the file lands in ``{output_dir}/{subdir}/...``;
        the subdirectory is created on demand.
        """
        stem = self.image_name(image_path)
        directory = self._output_dir
        if subdir:
            directory = os.path.join(directory, subdir)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, stem + suffix)

    def write_json(self, filename, data, indent=2):
        """Write ``data`` as JSON to ``{output_dir}/{filename}``; return path."""
        self.ensure_output_dir()
        path = os.path.join(self._output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=_json_default)
        return path

    def write_text(self, filename, text):
        """Write ``text`` verbatim to ``{output_dir}/{filename}``; return path."""
        self.ensure_output_dir()
        path = os.path.join(self._output_dir, filename)
        with open(path, "w") as f:
            f.write(text)
        return path

    # ------------------------------------------------------------------
    # GeoTIFF reading + projection helpers (require rasterio)
    # ------------------------------------------------------------------

    def read_image(self, image_path, dtype="float32", validate_bands=True):
        """Open a GeoTIFF and return ``(data, geo)``.

        ``data`` is shape ``(bands, H, W)`` with the requested dtype.
        ``geo`` is ``{transform, crs, bounds, width, height}``.

        With ``validate_bands=True`` (default) raises ValueError when the
        file's band count doesn't match ``len(selected_bands)`` — this catches
        captures truncated to JPEG/PNG that silently drop bands beyond 3.

        Raises ImportError if rasterio isn't installed.
        """
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "read_image() requires the 'rasterio' package. "
                "Install it (pip install rasterio) or call get_next_image() "
                "directly and decode the file yourself."
            ) from e

        with rasterio.open(image_path) as src:
            data = src.read()
            if dtype:
                data = data.astype(dtype)
            geo = {
                "transform": src.transform,
                "crs": src.crs,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
            }

        if validate_bands:
            expected = len(self.selected_bands)
            actual = data.shape[0]
            if expected and actual != expected:
                raise ValueError(
                    f"{os.path.basename(image_path)} has {actual} bands but "
                    f"player.config declares {expected} ({self.selected_bands}). "
                    f"The capture format may have truncated bands — re-run "
                    f"with image_format=tiff."
                )

        return data, geo

    @staticmethod
    def pixel_to_latlon(row, col, geo):
        """Convert pixel ``(row, col)`` to ``(lat, lon)`` rounded to 6 dp.

        Returns ``(None, None)`` when ``geo`` is missing. Reprojects to
        WGS84 (EPSG:4326) when the source CRS is something else. Requires
        rasterio + pyproj.
        """
        if geo is None:
            return None, None
        try:
            from rasterio.transform import xy
            import pyproj
        except ImportError as e:
            raise ImportError(
                "pixel_to_latlon() requires 'rasterio' and 'pyproj'. "
                "Install them (pip install rasterio pyproj)."
            ) from e

        crs_x, crs_y = xy(geo["transform"], row, col)
        crs = geo.get("crs")

        if crs and str(crs) != "EPSG:4326":
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(crs_x, crs_y)
        else:
            lon, lat = crs_x, crs_y

        return round(lat, 6), round(lon, 6)

    @staticmethod
    def estimate_gsd_meters(geo):
        """Return ground-sample distance in metres, or None if not derivable.

        Reads the affine transform's pixel-width and only returns a value when
        the CRS is projected (geographic CRSes have degree units, not metres,
        and we don't approximate latitude-dependent scaling here).
        """
        if not geo or not geo.get("transform") or not geo.get("crs"):
            return None
        if not geo["crs"].is_projected:
            return None
        return abs(geo["transform"][0])

    def require_min_gsd(self, geo, meters):
        """Assert the image's GSD is at least as fine as ``meters``; return it.

        Raises ValueError when the actual GSD is coarser. Returns None
        (without raising) when GSD can't be derived — caller decides.
        """
        gsd = self.estimate_gsd_meters(geo)
        if gsd is None:
            return None
        if gsd > meters:
            raise ValueError(
                f"Image GSD ({gsd:.2f} m) is coarser than required ({meters} m). "
                f"Use a higher-resolution capture or relax the requirement."
            )
        return gsd

    # ------------------------------------------------------------------
    # Per-band slicing (numpy required by caller)
    # ------------------------------------------------------------------

    def band(self, data, band_id, axis=0):
        """Return the slice of ``data`` for one band.

        Accepts both dashboard ids (``'blue'``) and Sentinel-2 names
        (``'B02'``). ``axis`` is the band axis in ``data`` (default 0,
        matching ``read_image()``'s ``(bands, H, W)`` layout).
        """
        idx = self._band_index(band_id)
        # Slice manually rather than np.take to avoid requiring numpy as a
        # hard dependency just to index — works for any sequence-like.
        slicer = [slice(None)] * data.ndim
        slicer[axis] = idx
        return data[tuple(slicer)]

    def bands(self, data, band_ids, axis=0, stack_axis=-1):
        """Stack multiple bands into one array.

        Accepts dashboard ids or S2 names mixed in any order. ``axis`` is the
        band axis in ``data`` (default 0). ``stack_axis`` is the axis the
        result is stacked along — default ``-1`` gives ``(H, W, N)`` which is
        what RGB display code wants. Use ``stack_axis=0`` for ``(N, H, W)``.

        Requires numpy (caller will already need it for arithmetic).
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("bands() requires numpy.") from e
        slices = [self.band(data, b, axis=axis) for b in band_ids]
        return np.stack(slices, axis=stack_axis)

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    @staticmethod
    def get_logger(name="aicube", log_file=None, output_dir=None,
                   level=logging.INFO):
        """Return a stdlib logger writing to stdout + a file.

        Defaults the file to ``{output_dir or ILC_OUTPUT_DIR}/cust_output.txt``
        — matching the convention every sample app already uses, and what the
        dashboard's task-detail view parses. Idempotent: repeated calls with
        the same name return the same configured logger without piling up
        handlers.
        """
        if log_file is None:
            output_dir = (output_dir or
                          os.environ.get("ILC_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR))
            log_file = os.path.join(output_dir, "cust_output.txt")

        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

        log = logging.getLogger(name)
        log.setLevel(level)
        log.propagate = False  # don't double-print via the root logger

        if log.handlers:
            return log

        fmt = logging.Formatter("[%(asctime)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)

        try:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError as e:
            # Filesystem may be read-only in some sandboxes; keep stdout going.
            log.warning(f"Could not open log file {log_file}: {e}")

        return log

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover_images(self):
        """Find all image files in AOI subdirectories.

        Returns files sorted by (subdirectory, index) so img_0 comes before
        img_10. Metadata JSON files are excluded.

        Returns:
            list[str]: Sorted list of image file paths.
        """
        found = []
        for ext in _IMAGE_EXTENSIONS:
            for path in glob.glob(os.path.join(self.data_dir, '*', f'*.{ext}')):
                base = os.path.splitext(os.path.basename(path))[0]
                if base.endswith('_metadata'):
                    continue
                match = _FILENAME_RE.match(base)
                # Key: (directory, numeric index) — falls back to base name for
                # files that don't match the expected {prefix}_{index} pattern.
                if match:
                    sort_key = (os.path.dirname(path), int(match.group(2)))
                else:
                    sort_key = (os.path.dirname(path), -1, base)
                found.append((sort_key, path))

        found.sort(key=lambda x: x[0])
        return [p for _, p in found]

    def _mark_returned(self, image_path):
        """Record an image as returned."""
        self._returned_images.add(image_path)
        logger.info(f"Image {self.processed_count}/{self.total_guidance_targets}: {os.path.basename(image_path)}")

    @staticmethod
    def _extract_prefix(filepath):
        """Return the image base name without extension (e.g. 'img_0').

        Kept for backward compatibility — prefer the public ``image_name()``
        method on a loader instance.
        """
        return os.path.splitext(os.path.basename(filepath))[0]

    @staticmethod
    def _find_metadata_path(image_path):
        """Derive the metadata JSON path for an image file.

        For an image at `dir/img_0.png`, returns `dir/img_0_metadata.json`
        if the file exists.

        Args:
            image_path: Path to the image file.

        Returns:
            str: Path to metadata file if it exists, None otherwise.
        """
        directory = os.path.dirname(image_path)
        base = os.path.splitext(os.path.basename(image_path))[0]
        metadata_path = os.path.join(directory, base + '_metadata.json')
        if os.path.isfile(metadata_path):
            return metadata_path

        logger.debug(f"No metadata file found at {metadata_path}")
        return None


def _json_default(obj):
    """JSON serializer for objects write_json() may encounter that aren't
    natively encodable (datetimes, numpy scalars).
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    # Numpy scalars and arrays: lazy import to avoid a hard numpy dep.
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
