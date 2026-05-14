"""
Crop health classifier — sample AICube app.

What this app demonstrates:
    1. Receiving multi-band satellite imagery from the AICube task pipeline
       via `AICubeImageLoader.iter_images()`.
    2. Running ONNX inference on the AICube's GPU (CUDAExecutionProvider
       when available, CPUExecutionProvider as fallback) — see
       load_onnx_model().
    3. Slicing named spectral bands out of a multi-band GeoTIFF using
       `loader.band()` even when the dashboard saved bands in a different
       order than the model expects.
    4. A complete remote-sensing analysis pipeline: tile → classify →
       compute vegetation indices (NDVI / NDRE / SAVI / NDWI) → score
       per-pixel health → bucket into stress zones → emit advisories.
    5. Writing structured outputs (JSON + side-by-side PNG + text report)
       through the loader's write_json() / write_text() helpers so the
       dashboard can parse them.

Required task configuration (set via the OrbitLab dashboard before
uploading the package):
    image_format: tiff
    bands:        ["blue", "green", "red", "nir-2"]   (any order)
    optional:     add "red-edge" for the full 5-channel model path —
                  without it, prepare_model_input() leaves the red-edge
                  model channel at zero and compute_vegetation_indices()
                  skips NDRE entirely.

Why TIFF (not JPEG / PNG / raw):
    NDVI / NDRE / NDWI / SAVI are all band-arithmetic on raw reflectance
    values. JPEG/PNG gamma-compress and silently truncate beyond 3 bands;
    raw writes a header-less binary blob rasterio can't open. TIFF
    preserves both the bit depth and the per-band layout the loader and
    model expect.
"""

import os
import time
import datetime
import numpy as np
from aicube_image_loader import AICubeImageLoader

# Path to the EuroSAT ResNet18 ONNX file. Override via env var for local
# testing; the default matches where the AICube uplink package lands the
# model inside the running container.
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/eurosat_resnet18.onnx")
# MODEL_PATH = os.environ.get("MODEL_PATH", "./eurosat_resnet18.onnx")

# Bands the script absolutely needs. The model itself was trained on 5
# channels (B02/B03/B04/B05/B08); red-edge (B05) is omitted from the
# REQUIRED list so users can run with a 4-band capture and still get
# classification + most indices. When red-edge IS captured it lights up
# the 5th model channel (see prepare_model_input) and adds NDRE.
REQUIRED_BANDS = ['blue', 'green', 'red', 'nir-2']

# Module-level logger so helpers above main() can call log.info() without
# threading the instance through every signature. get_logger() writes to
# stdout AND to {ILC_OUTPUT_DIR}/cust_output.txt — the dashboard's task-
# detail view tails that file.
log = AICubeImageLoader.get_logger("crop_health")


# EuroSAT class names in alphabetical order (matches model output indices)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Classes that count as crops for health analysis
CROP_CLASSES = {"AnnualCrop", "PermanentCrop", "Pasture"}

# Mapping from our bands to the 5-channel model input
# Model was trained on 5 bands: [B02, B03, B04, B05, B08] → channels 0-4
BAND_TO_MODEL_CHANNEL = {
    "B02": 0,   # Blue  (490nm)
    "B03": 1,   # Green (560nm)
    "B04": 2,   # Red   (665nm)
    "B05": 3,   # Red Edge (705nm)
    "B08": 4,   # NIR   (842nm)
}
MODEL_INPUT_CHANNELS = 5

# Classification map colors (BGR for OpenCV)
CLASS_COLORS_BGR = {
    "AnnualCrop":           (0, 215, 255),    # yellow
    "Forest":               (34, 139, 34),     # dark green
    "HerbaceousVegetation": (50, 205, 50),     # medium green
    "Highway":              (128, 128, 128),   # gray
    "Industrial":           (19, 69, 139),     # brown
    "Pasture":              (144, 238, 144),   # light green
    "PermanentCrop":        (32, 165, 218),    # dark yellow
    "Residential":          (60, 20, 220),     # crimson
    "River":                (225, 105, 65),    # royal blue
    "SeaLake":              (128, 0, 0),       # navy
}

# Health zone colors (BGR for OpenCV)
HEALTH_COLORS_BGR = {
    "Healthy":         (0, 170, 0),      # green
    "Moderate Stress": (0, 255, 255),    # yellow
    "Stressed":        (0, 140, 255),    # orange
    "Severe Stress":   (0, 0, 255),      # red
}

CONFIG = {
    # Tiling
    "tile_size": 64,
    "onnx_batch_size": 16,
    "classification_confidence_threshold": 0.4,

    # NDVI thresholds
    "ndvi_healthy_min": 0.45,
    "ndvi_moderate_min": 0.25,
    "ndvi_stressed_min": 0.15,

    # NDRE thresholds
    "ndre_healthy_min": 0.30,
    "ndre_nitrogen_stress": 0.20,

    # SAVI
    "savi_L": 0.5,
    "savi_healthy_min": 0.40,

    # NDWI
    "ndwi_water_stress": -0.20,

    # Health score weights (sum to 1.0)
    "weight_ndvi": 0.35,
    "weight_ndre": 0.30,
    "weight_savi": 0.20,
    "weight_ndwi": 0.15,

    # Resolution and timing
    "resolution_m": 10.0,
    "new_image_timeout_s": 300,
}


# =============================================================================
# Tiling
# =============================================================================

def tile_image(data, tile_size=64):
    """Tile a (C, H, W) image into non-overlapping (tile_size x tile_size) patches.

    The EuroSAT model takes fixed 64x64 inputs, but our captures are
    arbitrary larger sizes. We chop the image into a grid of 64x64 tiles
    so each one can be classified independently, then reassemble the per-
    tile predictions into a full-image classification map (see
    assemble_classification_map). Right/bottom edges are zero-padded so
    the dimensions divide evenly; the padding gets cropped back off after
    reassembly.

    Returns:
        tiles: array of shape (N, C, tile_size, tile_size) where N is
            (n_rows * n_cols).
        tile_positions: parallel list of (row_start, col_start) for each
            tile — used by assemble_classification_map() to put predictions
            back in the right place.
        original_shape: (H, W) of the input before padding.
    """
    c, h, w = data.shape
    original_shape = (h, w)

    # Pad to multiples of tile_size on the right/bottom. Zero is a safe
    # fill since EuroSAT was trained on land scenes — pure-zero patches
    # at the edge will just classify as low-confidence non-crop.
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    _, h_pad, w_pad = data.shape
    n_rows = h_pad // tile_size
    n_cols = w_pad // tile_size

    tiles = []
    tile_positions = []
    for r in range(n_rows):
        for c_idx in range(n_cols):
            r0 = r * tile_size
            c0 = c_idx * tile_size
            tile = data[:, r0:r0 + tile_size, c0:c0 + tile_size]
            tiles.append(tile)
            tile_positions.append((r0, c0))

    tiles = np.stack(tiles, axis=0)
    log.info(f"Tiled image into {len(tiles)} patches ({n_rows}x{n_cols} grid)")
    return tiles, tile_positions, original_shape


# =============================================================================
# ONNX Inference
# =============================================================================

def load_onnx_model(model_path):
    """Load the ONNX model, preferring CUDA if available.

    The AICube has an NVIDIA Orin NX, so CUDAExecutionProvider should be
    available in the production runtime — typically delivers ~10x speedup
    over CPU for this model. We list CPUExecutionProvider as the fallback
    so the script still runs in a local dev container without CUDA.
    """
    import onnxruntime as ort

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, providers=providers)
    active = session.get_providers()
    log.info(f"ONNX model loaded, providers: {active}")
    return session


def prepare_model_input(loader, tiles):
    """Map captured bands into the 5-channel layout the model expects.

    The model was trained on Sentinel-2 channels [B02, B03, B04, B05, B08]
    (blue, green, red, red-edge, NIR) in that fixed channel order. The
    captured tiles, however, are in whatever order the dashboard saved
    bands in player.config — and may not include all 5 bands at all (red-
    edge is optional).

    We therefore pre-allocate a zero array of the model's expected shape
    and only fill in the channels for bands that ARE present. Missing
    bands stay at zero, which is the closest the model can get to "no
    information" given its fixed input layout — accuracy degrades a bit
    but inference still produces sensible outputs.

    Args:
        loader: AICubeImageLoader — used to slice the captured tiles by
            band name without the caller knowing the axis index.
        tiles: (N, C, 64, 64) where C = len(loader.selected_bands).

    Returns:
        float32 array of shape (N, 5, 64, 64), normalized to [0, 1].
    """
    n = tiles.shape[0]
    ts = tiles.shape[2]
    model_input = np.zeros((n, MODEL_INPUT_CHANNELS, ts, ts), dtype=np.float32)

    # Walk our captured bands; if one of them is in the model's input
    # vocabulary, copy it into the matching model channel. Bands the model
    # doesn't know about are silently skipped.
    for band_name in loader.selected_bands_s2:
        if band_name in BAND_TO_MODEL_CHANNEL:
            ch = BAND_TO_MODEL_CHANNEL[band_name]
            # axis=1 because tiles is (N, C, H, W) — C is axis 1.
            band_data = loader.band(tiles, band_name, axis=1).astype(np.float32)
            # Sentinel-2 L2A surface reflectance is encoded as scaled
            # integers (0-10000 = 0-1.0 reflectance). The model wants [0, 1]
            # floats. Heuristic: if values exceed 1.0, assume the encoded
            # form and rescale; otherwise the data is already in [0, 1].
            if band_data.max() > 1.0:
                band_data = band_data / 10000.0
            model_input[:, ch, :, :] = band_data

    return model_input


def classify_tiles(session, model_input, batch_size=16):
    """Run ONNX inference on tiles in batches.

    Returns:
        class_indices: (N,) predicted class index per tile
        confidences: (N,) max softmax probability per tile
        class_probs: (N, 10) full probability distribution per tile
    """
    n = model_input.shape[0]
    input_name = session.get_inputs()[0].name

    all_logits = []
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch = model_input[start:start + batch_size]
        output = session.run(None, {input_name: batch})
        all_logits.append(output[0])

    logits = np.concatenate(all_logits, axis=0)  # (N, 10)

    # Softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    class_probs = exp / exp.sum(axis=1, keepdims=True)

    class_indices = class_probs.argmax(axis=1)
    confidences = class_probs.max(axis=1)

    elapsed = time.time() - t0
    log.info(f"ONNX inference: {n} patches in {elapsed:.2f}s ({n / elapsed:.0f} patches/s)")
    return class_indices, confidences, class_probs


# =============================================================================
# Classification Map
# =============================================================================

def assemble_classification_map(class_indices, confidences, tile_positions,
                                original_shape, tile_size=64):
    """Build full-image classification and confidence maps from tile results."""
    h, w = original_shape
    # Allocate padded size
    max_r = max(p[0] for p in tile_positions) + tile_size
    max_c = max(p[1] for p in tile_positions) + tile_size
    class_map = np.zeros((max_r, max_c), dtype=np.uint8)
    conf_map = np.zeros((max_r, max_c), dtype=np.float32)

    for idx, (r0, c0) in enumerate(tile_positions):
        class_map[r0:r0 + tile_size, c0:c0 + tile_size] = class_indices[idx]
        conf_map[r0:r0 + tile_size, c0:c0 + tile_size] = confidences[idx]

    # Crop to original size
    return class_map[:h, :w], conf_map[:h, :w]


def compute_crop_statistics(classification_map, confidence_map):
    """Compute per-class statistics and a boolean crop mask."""
    res = CONFIG["resolution_m"]
    pixel_area_ha = (res * res) / 10000.0
    total_pixels = classification_map.size

    stats = {}
    for i, cls_name in enumerate(EUROSAT_CLASSES):
        mask = classification_map == i
        count = int(mask.sum())
        stats[cls_name] = {
            "pixels": count,
            "area_hectares": round(count * pixel_area_ha, 2),
            "percentage": round(100.0 * count / total_pixels, 1),
        }

    crop_mask = np.zeros_like(classification_map, dtype=bool)
    for cls_name in CROP_CLASSES:
        cls_idx = EUROSAT_CLASSES.index(cls_name)
        crop_mask |= (classification_map == cls_idx)

    crop_pixels = int(crop_mask.sum())
    return {
        "per_class": stats,
        "crop_mask": crop_mask,
        "total_crop_pixels": crop_pixels,
        "total_crop_area_hectares": round(crop_pixels * pixel_area_ha, 2),
        "total_crop_percentage": round(100.0 * crop_pixels / total_pixels, 1),
        "total_area_hectares": round(total_pixels * pixel_area_ha, 2),
    }


# =============================================================================
# Vegetation Indices
# =============================================================================

def compute_vegetation_indices(loader, data, crop_mask):
    """Compute four vegetation indices on the pixels classified as crops.

    All four are normalised band ratios — they share the form
    (band_a − band_b) / (band_a + band_b), which sits in [-1, +1] and
    cancels out absolute illumination differences (so two captures of the
    same field at different sun angles compare apples-to-apples). Each
    captures a different aspect of vegetation:

        NDVI — chlorophyll content / overall greenness. Healthy leaves
               reflect NIR strongly and absorb Red; positive NDVI ≈
               vigorous canopy.
        NDRE — (red-edge variant) more sensitive to subtle chlorophyll
               drops and thus to early nitrogen stress, before NDVI
               notices. Only computed when red-edge was captured.
        SAVI — soil-adjusted vegetation index. A modified NDVI with an L
               correction term that suppresses bare-soil contamination in
               sparse-canopy scenes (early growth, recently planted fields).
        NDWI — McFeeters' water index here, doubling as a leaf-water /
               crop-moisture proxy on land. Less negative = wetter.

    We restrict the math to the crop mask so we don't waste cycles on
    pixels that classified as Forest / Highway / Water / etc. Non-crop
    pixels stay NaN throughout, which makes np.nanmean() friendly later.
    """
    eps = 1e-10  # guards the divisions against 0/0 over no-data pixels

    def _get_band(name):
        return loader.band(data, name).astype(np.float32)

    green = _get_band("B03")
    red = _get_band("B04")
    nir = _get_band("B08")
    has_re = "B05" in loader.selected_bands_s2
    red_edge = _get_band("B05") if has_re else None

    h, w = red.shape
    ndvi = np.full((h, w), np.nan, dtype=np.float32)
    ndre = np.full((h, w), np.nan, dtype=np.float32) if has_re else None
    savi = np.full((h, w), np.nan, dtype=np.float32)
    ndwi = np.full((h, w), np.nan, dtype=np.float32)

    m = crop_mask  # alias for readability below

    # NDVI: most common vegetation index, defined for over 50 years.
    ndvi[m] = (nir[m] - red[m]) / (nir[m] + red[m] + eps)

    # NDRE: red-edge swap for subtler stress detection.
    if has_re:
        ndre[m] = (nir[m] - red_edge[m]) / (nir[m] + red_edge[m] + eps)

    # SAVI: L=0.5 is the canonical mid-density choice (use L=1 for very
    # sparse canopies; L=0 collapses back to NDVI).
    L = CONFIG["savi_L"]
    savi[m] = ((nir[m] - red[m]) / (nir[m] + red[m] + L + eps)) * (1 + L)

    # NDWI: same family, swapping Red for Green and inverting.
    ndwi[m] = (green[m] - nir[m]) / (green[m] + nir[m] + eps)

    result = {"ndvi": ndvi, "savi": savi, "ndwi": ndwi}
    if has_re:
        result["ndre"] = ndre
    return result


# =============================================================================
# Health Scoring
# =============================================================================

def compute_health_scores(indices, crop_mask):
    """Compute per-pixel health scores (0-100) using piecewise linear mapping.

    Each vegetation index gets mapped to a 0-100 sub-score via np.interp
    against an empirically-tuned breakpoint table — values below the first
    breakpoint score 0, values above the last score 100, and intermediate
    values are linearly interpolated. The breakpoints are *not* universal:
    they were tuned for typical mid-latitude row crops on Sentinel-2 L2A
    surface reflectance, and would need adjustment for arid regions, rice
    paddies, etc.

    The four sub-scores are then combined into one health score with the
    weights from CONFIG (defaults: NDVI 35% / NDRE 30% / SAVI 20% / NDWI
    15%). When red-edge wasn't captured the NDRE weight is redistributed
    proportionally to the other three so the total still sums to 1.0.

    Returns:
        health_scores: (H, W) float array, NaN for non-crop
        index_scores: dict of individual score arrays for diagnostics
    """
    h, w = crop_mask.shape
    m = crop_mask

    # NDVI breakpoints: 0.0 = bare/sick, 0.15 = sparse, 0.25 = struggling,
    # 0.45 = healthy, 0.85 = lush canopy.
    ndvi_score = np.full((h, w), np.nan, dtype=np.float32)
    ndvi_score[m] = np.interp(
        indices["ndvi"][m],
        [0.0, 0.15, 0.25, 0.45, 0.85],
        [0.0, 15.0, 40.0, 70.0, 100.0],
    )

    # SAVI: similar shape to NDVI but shifted slightly because of the
    # soil-correction L term.
    savi_score = np.full((h, w), np.nan, dtype=np.float32)
    savi_score[m] = np.interp(
        indices["savi"][m],
        [0.0, 0.20, 0.40, 0.80],
        [0.0, 30.0, 70.0, 100.0],
    )

    # NDWI for crops sits in roughly [-0.6, +0.1]: very negative = dry
    # leaves / drought, less negative = well-watered, near zero = saturated.
    ndwi_score = np.full((h, w), np.nan, dtype=np.float32)
    ndwi_score[m] = np.interp(
        indices["ndwi"][m],
        [-0.60, -0.40, -0.20, -0.10, 0.10],
        [0.0, 20.0, 50.0, 80.0, 100.0],
    )

    index_scores = {"ndvi": ndvi_score, "savi": savi_score, "ndwi": ndwi_score}

    has_ndre = "ndre" in indices
    if has_ndre:
        # NDRE specifically tracks chlorophyll/nitrogen status; lower
        # cutoffs than NDVI because red-edge values run smaller.
        ndre_score = np.full((h, w), np.nan, dtype=np.float32)
        ndre_score[m] = np.interp(
            indices["ndre"][m],
            [0.0, 0.20, 0.30, 0.60],
            [0.0, 30.0, 70.0, 100.0],
        )
        index_scores["ndre"] = ndre_score

    # Pick the four sub-score weights. With NDRE present we use the
    # configured weights as-is; without it we redistribute the NDRE share
    # across the others so the weighted sum still tops out at 100.
    if has_ndre:
        w_ndvi = CONFIG["weight_ndvi"]
        w_ndre = CONFIG["weight_ndre"]
        w_savi = CONFIG["weight_savi"]
        w_ndwi = CONFIG["weight_ndwi"]
    else:
        remaining = CONFIG["weight_ndvi"] + CONFIG["weight_savi"] + CONFIG["weight_ndwi"]
        w_ndvi = CONFIG["weight_ndvi"] / remaining
        w_ndre = 0.0
        w_savi = CONFIG["weight_savi"] / remaining
        w_ndwi = CONFIG["weight_ndwi"] / remaining

    health = np.full((h, w), np.nan, dtype=np.float32)
    health[m] = w_ndvi * ndvi_score[m] + w_savi * savi_score[m] + w_ndwi * ndwi_score[m]
    if has_ndre:
        health[m] += w_ndre * ndre_score[m]

    health[m] = np.clip(health[m], 0.0, 100.0)
    return health, index_scores


def classify_health_zones(health_scores, crop_mask):
    """Bucket per-pixel health scores into four discrete stress zones.

    Cutoffs (uniform 20-point bands above 30 with a wider Healthy bucket):
        score ≥ 70  → Healthy           (zone value 1, green)
        50 ≤ s < 70 → Moderate Stress   (zone value 2, yellow)
        30 ≤ s < 50 → Stressed          (zone value 3, orange)
        score < 30  → Severe Stress     (zone value 4, red)
        non-crop pixels stay at zone value 0 throughout.

    These thresholds map directly to colour swatches in the annotated PNG
    and to the per-zone area stats in the JSON output. Adjust them if your
    customers want a different stress sensitivity.
    """
    h, w = crop_mask.shape
    zone_map = np.zeros((h, w), dtype=np.uint8)
    m = crop_mask
    scores = health_scores[m]

    zone_values = np.zeros(scores.shape, dtype=np.uint8)
    zone_values[scores >= 70] = 1
    zone_values[(scores >= 50) & (scores < 70)] = 2
    zone_values[(scores >= 30) & (scores < 50)] = 3
    zone_values[scores < 30] = 4

    zone_map[m] = zone_values

    res = CONFIG["resolution_m"]
    pixel_area_ha = (res * res) / 10000.0
    zone_names = {1: "Healthy", 2: "Moderate Stress", 3: "Stressed", 4: "Severe Stress"}
    crop_total = int(crop_mask.sum())

    zone_stats = {}
    for val, name in zone_names.items():
        count = int((zone_map == val).sum())
        zone_stats[name] = {
            "pixels": count,
            "area_hectares": round(count * pixel_area_ha, 2),
            "percentage": round(100.0 * count / max(crop_total, 1), 1),
        }

    return zone_map, zone_stats


# =============================================================================
# Advisory Generation
# =============================================================================

def generate_advisories(indices, index_scores, zone_stats, crop_stats):
    """Generate rule-based advisory messages sorted by priority.

    A simple expert-system pattern: each `if` block fires when one
    threshold is crossed and contributes one advisory dict (priority,
    category, human-readable message, affected area). The list is then
    sorted HIGH → MEDIUM → LOW so the most important messages appear
    first in the dashboard. To add a new advisory, copy any of the blocks
    below and adjust the trigger condition.
    """
    advisories = []
    crop_mask = crop_stats["crop_mask"]
    crop_total = crop_stats["total_crop_pixels"]

    if crop_total == 0:
        return advisories

    # Severe vegetation stress
    severe_pct = zone_stats["Severe Stress"]["percentage"]
    if severe_pct > 10:
        advisories.append({
            "priority": "HIGH",
            "category": "vegetation_stress",
            "message": (
                f"Severe vegetation stress detected in {severe_pct:.1f}% of crop area. "
                "Possible causes: drought, pest damage, or nutrient deficiency. "
                "Immediate field inspection recommended."
            ),
            "affected_area_hectares": zone_stats["Severe Stress"]["area_hectares"],
        })

    # Nitrogen deficiency (NDRE)
    if "ndre" in indices and indices["ndre"] is not None:
        ndre_vals = indices["ndre"][crop_mask]
        mean_ndre = float(np.nanmean(ndre_vals))
        if mean_ndre < CONFIG["ndre_nitrogen_stress"]:
            advisories.append({
                "priority": "HIGH",
                "category": "nutrient_deficiency",
                "message": (
                    f"Low NDRE values (mean={mean_ndre:.3f}) indicate possible nitrogen "
                    "deficiency. Consider soil testing and nitrogen supplementation."
                ),
                "affected_area_hectares": crop_stats["total_crop_area_hectares"],
            })

    # Water stress (NDWI)
    ndwi_vals = indices["ndwi"][crop_mask]
    water_stressed = float(np.nanmean(ndwi_vals < CONFIG["ndwi_water_stress"])) * 100
    if water_stressed > 15:
        priority = "HIGH" if water_stressed > 30 else "MEDIUM"
        advisories.append({
            "priority": priority,
            "category": "water_stress",
            "message": (
                f"Water stress detected in {water_stressed:.0f}% of crop area "
                f"(NDWI < {CONFIG['ndwi_water_stress']}). "
                "Irrigation recommended within 48-72 hours."
            ),
            "affected_area_hectares": round(
                water_stressed / 100 * crop_stats["total_crop_area_hectares"], 2
            ),
        })

    # Moderate stress monitoring
    mod_pct = zone_stats["Moderate Stress"]["percentage"]
    if mod_pct > 20:
        advisories.append({
            "priority": "MEDIUM",
            "category": "monitoring",
            "message": (
                f"Moderate stress in {mod_pct:.1f}% of crop area. "
                "Monitor closely over the next 1-2 weeks for progression."
            ),
            "affected_area_hectares": zone_stats["Moderate Stress"]["area_hectares"],
        })

    # Sparse canopy (low SAVI)
    savi_vals = indices["savi"][crop_mask]
    mean_savi = float(np.nanmean(savi_vals))
    if mean_savi < 0.25:
        advisories.append({
            "priority": "MEDIUM",
            "category": "canopy_coverage",
            "message": (
                f"Low canopy coverage detected (mean SAVI={mean_savi:.3f}). "
                "Crops may be in early growth stage or experiencing stunted growth."
            ),
            "affected_area_hectares": crop_stats["total_crop_area_hectares"],
        })

    # Positive feedback
    healthy_pct = zone_stats["Healthy"]["percentage"]
    if healthy_pct > 60:
        advisories.append({
            "priority": "LOW",
            "category": "positive",
            "message": (
                f"Majority of crop area ({healthy_pct:.1f}%) shows healthy vegetation. "
                "Current management practices appear effective."
            ),
            "affected_area_hectares": zone_stats["Healthy"]["area_hectares"],
        })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    advisories.sort(key=lambda a: priority_order.get(a["priority"], 9))
    return advisories


# =============================================================================
# Per-Image Processing
# =============================================================================

def process_image(loader, record, onnx_session):
    """Run the full crop-health pipeline on one captured image.

    `record` comes from `loader.iter_images()`, which already opened the
    GeoTIFF, validated band counts against player.config, and packaged the
    decoded raster + geo metadata into a single ImageRecord. So
    `record.data` is the (bands, H, W) float32 array and `record.geo` is
    the {transform, crs, bounds, ...} dict — same shape rasterio would
    give us if we'd opened the file ourselves.

    The pipeline is:
        tile → ONNX classify → reassemble class map → derive crop mask →
        compute vegetation indices → score health → bucket into zones →
        emit advisories.

    If almost no crops are detected we short-circuit after classification
    and return early — the health analysis would just be noise on a
    handful of pixels.
    """
    data = record.data
    geo_metadata = record.geo

    log.info(f"Loaded image: shape={data.shape}, bands={loader.selected_bands_s2}")

    # Use the actual ground-sample distance from the GeoTIFF's CRS where
    # available; per-class area calculations downstream depend on this.
    gsd = loader.estimate_gsd_meters(geo_metadata)
    if gsd is not None:
        CONFIG["resolution_m"] = gsd

    original_shape = data.shape[1:]  # (H, W)

    # Step 1 — chop the image into 64x64 patches the EuroSAT model accepts.
    tiles, tile_positions, orig_shape = tile_image(data, CONFIG["tile_size"])

    # Step 2 — re-arrange captured bands into the 5-channel layout the
    # model expects (missing bands → zeros).
    model_input = prepare_model_input(loader, tiles)

    # Step 3 — run the model. Returns a class index + softmax confidence
    # per tile.
    class_indices, confidences, class_probs = classify_tiles(
        onnx_session, model_input, CONFIG["onnx_batch_size"]
    )

    # Step 4 — paste per-tile predictions back onto the original-sized canvas.
    classification_map, confidence_map = assemble_classification_map(
        class_indices, confidences, tile_positions, original_shape, CONFIG["tile_size"]
    )

    # Step 5 — derive a boolean crop_mask + per-class hectare stats by
    # marking AnnualCrop / PermanentCrop / Pasture pixels as "crop".
    crop_stats = compute_crop_statistics(classification_map, confidence_map)
    log.info(
        f"Classification: {crop_stats['total_crop_percentage']:.1f}% crop "
        f"({crop_stats['total_crop_area_hectares']:.1f} ha)"
    )

    # Bail out if there's barely any crop area — the rest of the pipeline
    # would be a noisy waste of compute on a handful of pixels.
    if crop_stats["total_crop_percentage"] < 1.0:
        log.info("Less than 1% crop area detected, skipping health analysis")
        return {
            "data": data,
            "geo_metadata": geo_metadata,
            "classification_map": classification_map,
            "confidence_map": confidence_map,
            "crop_stats": crop_stats,
            "indices": None,
            "health_scores": None,
            "health_zones": None,
            "zone_stats": None,
            "advisories": [],
        }

    # Step 6 — compute NDVI / NDRE / SAVI / NDWI on crop pixels only.
    crop_mask = crop_stats["crop_mask"]
    indices = compute_vegetation_indices(loader, data, crop_mask)

    ndvi_mean = float(np.nanmean(indices["ndvi"][crop_mask]))
    log.info(f"Vegetation indices computed (mean NDVI={ndvi_mean:.3f})")

    # Step 7 — fold the four indices into one 0-100 health score per pixel.
    health_scores, index_scores = compute_health_scores(indices, crop_mask)
    overall = float(np.nanmean(health_scores[crop_mask]))
    log.info(f"Overall health score: {overall:.1f}/100")

    # Step 8 — bucket scores into Healthy / Moderate / Stressed / Severe zones.
    zone_map, zone_stats = classify_health_zones(health_scores, crop_mask)
    for name, stats in zone_stats.items():
        log.info(f"  {name}: {stats['percentage']:.1f}% ({stats['area_hectares']:.1f} ha)")

    # Step 9 — fire rule-based advisory messages from the zones + index means.
    advisories = generate_advisories(indices, index_scores, zone_stats, crop_stats)
    log.info(f"Generated {len(advisories)} advisory message(s)")

    return {
        "data": data,
        "geo_metadata": geo_metadata,
        "classification_map": classification_map,
        "confidence_map": confidence_map,
        "crop_stats": crop_stats,
        "indices": indices,
        "health_scores": health_scores,
        "health_zones": zone_map,
        "zone_stats": zone_stats,
        "advisories": advisories,
    }


# =============================================================================
# Output Generation
# =============================================================================

def _make_rgb(loader, data):
    """Create a uint8 RGB image from band data (R=B04, G=B03, B=B02).

    Sentinel-2 surface reflectance over land sits in roughly [0, 0.3] —
    if you cast it straight to uint8 it looks black. The gain (3.5) and
    gamma (1.1) values multiply and tonemap the values into the [0, 255]
    range that produces a natural-looking true-colour preview, matching
    the TRUE_COLOR evalscript used elsewhere in the SDK. Adjust per scene
    if your imagery is brighter (snow) or darker (water) than typical land.
    """
    gain = 3.5
    gamma = 1.1

    rgb = loader.bands(data, ["B04", "B03", "B02"]).astype(np.float32)
    rgb = np.power(np.clip(gain * rgb, 0, None), 1.0 / gamma)
    rgb = np.clip(rgb, 0, 1) * 255
    return rgb.astype(np.uint8)


def _draw_legend(canvas, items, start_x, start_y, box_size=48, spacing=64):
    """Draw a color legend on an OpenCV image."""
    import cv2

    y = start_y
    for color_bgr, label in items:
        cv2.rectangle(canvas, (start_x, y), (start_x + box_size, y + box_size), color_bgr, -1)
        cv2.rectangle(canvas, (start_x, y), (start_x + box_size, y + box_size), (0, 0, 0), 3)
        cv2.putText(
            canvas, label, (start_x + box_size + 14, y + box_size - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3,
        )
        y += spacing


def generate_outputs(loader, all_results, image_names):
    """Write JSON results, annotated PNG, and text report."""
    import cv2

    # Use the result with the most crop area for annotation
    best = None
    for r in all_results:
        if best is None or r["crop_stats"]["total_crop_percentage"] > \
                best["crop_stats"]["total_crop_percentage"]:
            best = r

    # Aggregate stats from the best (or first) result for the report
    result = best if best else all_results[0]
    crop_stats = result["crop_stats"]
    zone_stats = result.get("zone_stats") or {}
    advisories = result.get("advisories") or []
    indices = result.get("indices")

    # --- Compute summary index stats ---
    index_summary = {}
    if indices:
        crop_mask = crop_stats["crop_mask"]
        for name, arr in indices.items():
            vals = arr[crop_mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                index_summary[name] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }

    # Compute overall health score
    overall_health = None
    if result["health_scores"] is not None:
        h_vals = result["health_scores"][crop_stats["crop_mask"]]
        h_vals = h_vals[np.isfinite(h_vals)]
        if len(h_vals) > 0:
            overall_health = round(float(np.mean(h_vals)), 1)

    # --- Compute bounds ---
    bounds_dict = None
    geo = result.get("geo_metadata")
    if geo and geo.get("bounds"):
        b = geo["bounds"]
        bounds_dict = {
            "north": round(b.top, 6),
            "south": round(b.bottom, 6),
            "east": round(b.right, 6),
            "west": round(b.left, 6),
        }

    # ==================== JSON ====================
    json_data = {
        "task": "crop_health_analysis",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "images_processed": len(image_names),
        "image_names": image_names,
        "model": "eurosat_resnet18",
        "resolution_m": CONFIG["resolution_m"],
        "analysis_region": {
            "bounds": bounds_dict,
            "total_area_hectares": crop_stats["total_area_hectares"],
        },
        "land_classification": {
            "total_patches": sum(
                1 for s in crop_stats["per_class"].values() if s["pixels"] > 0
            ),
            "classes": crop_stats["per_class"],
        },
        "crop_health": {
            "total_crop_area_hectares": crop_stats["total_crop_area_hectares"],
            "crop_percentage_of_image": crop_stats["total_crop_percentage"],
            "vegetation_indices": index_summary,
            "health_zones": zone_stats,
            "overall_health_score": overall_health,
        },
        "advisories": advisories,
    }

    json_path = loader.write_json("crop_health_results.json", json_data)
    log.info(f"Wrote {json_path}")

    # ==================== Annotated PNG ====================
    png_path = None
    if best and best["data"] is not None:
        data = best["data"]
        cls_map = best["classification_map"]
        zone_map = best.get("health_zones")

        rgb = _make_rgb(loader, data)
        h, w = rgb.shape[:2]

        # Scale factor: target at least 500px height
        scale = max(3, int(np.ceil(500 / h)))

        # Panel 1: Classification map
        cls_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for i, cls_name in enumerate(EUROSAT_CLASSES):
            mask = cls_map == i
            if mask.any():
                cls_overlay[mask] = CLASS_COLORS_BGR[cls_name]

        cls_panel = cv2.addWeighted(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.4,
            cls_overlay, 0.6, 0,
        )
        cls_panel = cv2.resize(cls_panel, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        # Panel 2: Health map (crop areas colored, non-crop as RGB)
        health_panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        if zone_map is not None:
            zone_names = {1: "Healthy", 2: "Moderate Stress", 3: "Stressed", 4: "Severe Stress"}
            health_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            for val, name in zone_names.items():
                mask = zone_map == val
                if mask.any():
                    health_overlay[mask] = HEALTH_COLORS_BGR[name]
            crop_mask_3ch = np.stack([crop_stats["crop_mask"]] * 3, axis=-1)
            blended = cv2.addWeighted(health_panel, 0.4, health_overlay, 0.6, 0)
            health_panel = np.where(crop_mask_3ch, blended, health_panel)
        health_panel = cv2.resize(health_panel, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        sh, sw = h * scale, w * scale

        # Add titles
        cv2.putText(
            cls_panel, "Land Classification", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
        )
        cv2.putText(
            health_panel, "Crop Health", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
        )

        # Add legends
        cls_legend = [(CLASS_COLORS_BGR[c], c) for c in EUROSAT_CLASSES]
        _draw_legend(cls_panel, cls_legend, 10, 80)

        if zone_map is not None:
            health_legend = [(HEALTH_COLORS_BGR[n], n) for n in
                             ["Healthy", "Moderate Stress", "Stressed", "Severe Stress"]]
            _draw_legend(health_panel, health_legend, 10, 80)

        # Separator line
        sep = np.full((sh, 4, 3), 200, dtype=np.uint8)

        # Concatenate side by side
        composite = np.concatenate([cls_panel, sep, health_panel], axis=1)

        png_path = os.path.join(loader.output_dir, "crop_health_annotated.png")
        cv2.imwrite(png_path, composite)
        log.info(f"Wrote {png_path}")

    # ==================== Text Report ====================
    lines = []
    lines.append("=" * 60)
    lines.append("  CROP HEALTH MONITORING REPORT")
    lines.append("=" * 60)
    lines.append(f"Date: {json_data['timestamp']}")
    lines.append(f"Images processed: {len(image_names)}")
    for name in image_names:
        lines.append(f"  - {name}")
    lines.append("Model: EuroSAT ResNet18 (ONNX)")
    lines.append(f"Resolution: {CONFIG['resolution_m']}m/pixel")

    lines.append("-" * 60)
    lines.append("LAND CLASSIFICATION")
    for cls_name in EUROSAT_CLASSES:
        s = crop_stats["per_class"][cls_name]
        if s["pixels"] > 0:
            lines.append(
                f"  {cls_name:25s}  {s['area_hectares']:8.1f} ha  "
                f"({s['percentage']:5.1f}%)"
            )

    lines.append("-" * 60)
    lines.append("CROP HEALTH ANALYSIS")
    lines.append(
        f"  Total crop area: {crop_stats['total_crop_area_hectares']:.1f} hectares "
        f"({crop_stats['total_crop_percentage']:.1f}% of image)"
    )

    if index_summary:
        lines.append("")
        lines.append("  Vegetation Indices (crop areas only):")
        for idx_name, stats in index_summary.items():
            lines.append(
                f"    {idx_name.upper():6s}  mean={stats['mean']:.4f}  "
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
            )

    if zone_stats:
        lines.append("")
        lines.append("  Health Zones:")
        for zone_name in ["Healthy", "Moderate Stress", "Stressed", "Severe Stress"]:
            if zone_name in zone_stats:
                zs = zone_stats[zone_name]
                lines.append(
                    f"    {zone_name:18s}  {zs['area_hectares']:8.1f} ha  "
                    f"({zs['percentage']:5.1f}%)"
                )

    if overall_health is not None:
        lines.append("")
        lines.append(f"  Overall Health Score: {overall_health:.1f} / 100")

    if advisories:
        lines.append("-" * 60)
        lines.append("ADVISORIES")
        for adv in advisories:
            lines.append("")
            lines.append(
                f"  [{adv['priority']}] {adv['category'].replace('_', ' ').title()}:"
            )
            # Wrap message text
            msg = adv["message"]
            while msg:
                lines.append(f"    {msg[:56]}")
                msg = msg[56:]
            if "affected_area_hectares" in adv:
                lines.append(f"    Affected area: {adv['affected_area_hectares']} hectares")

    lines.append("")
    lines.append("=" * 60)

    report_path = loader.write_text("crop_health_report.txt", "\n".join(lines) + "\n")
    log.info(f"Wrote {report_path}")
    return json_path, png_path, report_path


# =============================================================================
# Main
# =============================================================================

# =============================================================================
# Main — the canonical AICube task pattern
# =============================================================================
#
# Every AICube task script follows roughly the same skeleton:
#
#     loader = AICubeImageLoader(...)        # read player.config, find data dir
#     loader.require_bands(MY_NEEDED_BANDS)  # fail fast if bands are missing
#     for record in loader.iter_images(...): # block-wait per image
#         my_processing(record.data, record.geo, ...)
#     loader.write_json(...) / write_text(...)  # results land in output dir
#
# The loader handles all the satellite-side plumbing (waiting for new
# captures to arrive, validating sidecar metadata, decoding GeoTIFFs,
# tracking how many of the expected images we've processed). Your code
# focuses on what to do with each image — here, run the model and derive
# crop-health products.

def main():
    loader = AICubeImageLoader(
        max_wait=CONFIG["new_image_timeout_s"],
        poll_interval=1,
    )
    loader.ensure_output_dir()

    log.info("Starting crop health monitoring app...")
    log.info(f"Watching for images in {loader.data_dir}/*/*")

    # Bail early if the model file isn't where we expect — saves the user
    # from waiting through a full image-arrival cycle just to discover the
    # ONNX file is missing. Test locally with MODEL_PATH=./eurosat_resnet18.onnx.
    if not os.path.exists(MODEL_PATH):
        log.info(f"ERROR: ONNX model not found at {MODEL_PATH}")
        log.info(
            "Run export_eurosat_onnx.py first, then set MODEL_PATH env var "
            "or place eurosat_resnet18.onnx in /workspace/"
        )
        return

    onnx_session = load_onnx_model(MODEL_PATH)

    # Fail fast if the dashboard task wasn't configured with the bands we
    # need. Without this guard we'd crash later inside compute_vegetation_indices()
    # with a less helpful error.
    try:
        loader.require_bands(REQUIRED_BANDS)
    except ValueError as e:
        log.info(f"FATAL: {e}")
        return
    log.info(
        f"Bands ok. Selected: {loader.selected_bands} → "
        f"S2 names: {loader.selected_bands_s2}"
    )

    # total_guidance_targets is summed across every aoi_exec entry in
    # player.config — it's how many images this task is expected to deliver.
    guidance_targets = loader.total_guidance_targets
    log.info(f"Expecting {guidance_targets} guidance target(s) from player.config")

    if guidance_targets == 0:
        log.info("No guidance targets configured, exiting")
        return

    all_results = []
    image_names = []

    # iter_images() yields one ImageRecord per arriving capture, blocking
    # on each wait_for_next_image() under the hood. validate_bands tells
    # it to skip images whose sidecar reports bands_saved missing any of
    # our required bands (catches captures truncated to JPEG/PNG).
    for record in loader.iter_images(validate_bands=REQUIRED_BANDS):
        log.info(f"Processing [{record.index}/{record.total}]: {record.name}")

        result = process_image(loader, record, onnx_session)
        if result is None:
            log.info(f"Skipping {record.name} (failed to load)")
            continue

        # Tag the result so the combined report can attribute findings to
        # the source image they came from.
        result["source_image"] = record.name
        image_names.append(record.name)
        all_results.append(result)

    log.info(f"All images processed: {len(image_names)} image(s)")

    if all_results:
        generate_outputs(loader, all_results, image_names)

    log.info(f"Results written to {loader.output_dir}")
    log.info("Task completed successfully!")


if __name__ == "__main__":
    main()
