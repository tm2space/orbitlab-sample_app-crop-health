# Crop Health Monitoring — AICube Docker Sample

A complete in-orbit crop-health analysis pipeline: classify land cover with an
ONNX-optimised EuroSAT ResNet18 model, compute four vegetation indices (NDVI /
NDRE / SAVI / NDWI) on the crop pixels, score each pixel for stress, bucket
into health zones, and emit rule-based agricultural advisories. Demonstrates
GPU ONNX inference on the AICube and the canonical
[`AICubeImageLoader`](aicube_image_loader.py) usage pattern.

---

## Overview

This sample is a Docker-packaged application built for the AICube platform
(NVIDIA Orin NX, ARM64). It receives multispectral GeoTIFF captures from the
AICube task pipeline, runs them through a tiled CNN classifier, and produces
crop-health products that downlink as a small JSON + annotated PNG instead of
raw imagery.

It is intended both as a **runnable example** for agricultural-monitoring
customers and as a **reference implementation** demonstrating:

- The `AICubeImageLoader.iter_images()` task pattern
- Per-band slicing of multispectral data by name (`loader.band(...)`)
- ONNX Runtime with the CUDA execution provider on Orin NX
- Tile-based inference reassembled into a full-image classification map
- Vegetation-index arithmetic on raw reflectance values
- Writing structured outputs the OrbitLab dashboard can render

---

## Application Details

| Field | Value |
|---|---|
| **Type** | Docker container running Python + ONNX Runtime |
| **Container image** | `tm2space/aicube-base:latest` (extended via `Dockerfile`) |
| **Entry point** | `crop_health_app.py` |
| **Model** | EuroSAT ResNet18 (`eurosat_resnet18.onnx`, ~43 MB) |
| **AI framework** | ONNX Runtime with CUDAExecutionProvider (CPU fallback) |
| **Required bands** | `blue`, `green`, `red`, `nir-2` (red-edge optional for full 5-channel model path) |
| **Required image format** | `tiff` |
| **Max execution** | 20 minutes (per `player.config`) |
| **Hardware** | NVIDIA Orin NX (CUDA strongly recommended; falls back to CPU) |
| **Output** | JSON results, side-by-side annotated PNG, text report |

---

## Pipeline

```
ImageRecord (from loader.iter_images())
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  PER-IMAGE PIPELINE                                          │
│                                                              │
│  1. Tile         — chop (bands, H, W) into 64×64 patches     │
│  2. Prepare      — re-arrange captured bands into 5-channel  │
│                    model input (missing bands → zeros)       │
│  3. Classify     — ONNX inference per batch (GPU when avail.)│
│  4. Reassemble   — paste per-tile predictions back as a map  │
│  5. Crop stats   — boolean crop_mask + per-class hectares    │
│  6. Indices      — NDVI / NDRE / SAVI / NDWI on crop pixels  │
│  7. Health score — piecewise-linear interp + weighted sum    │
│  8. Zone classify— bucket into Healthy / Moderate / Stressed │
│                    / Severe                                  │
│  9. Advisories   — rule-based messages from zone+index stats │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
JSON + annotated PNG + text report → loader.write_json/write_text
```

EuroSAT class outputs that count as "crop" for downstream analysis:
`AnnualCrop`, `PermanentCrop`, `Pasture`. Other classes
(`Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Residential`,
`River`, `SeaLake`) appear in the classification map and the per-class
hectare table but are excluded from the vegetation-index work.

---

## Files in this Package

| File | Purpose |
|---|---|
| `player.config` | Task config — bands, image_format, AOI guidance |
| `crop_health_app.py` | Main script (entry point) |
| `aicube_image_loader.py` | Single-file helper library — also available as a standalone download from the AICube Libraries tab |
| `eurosat_resnet18.onnx` | Pretrained classifier (10-class EuroSAT) |
| `Dockerfile` | Builds the runtime image on top of `tm2space/aicube-base:latest` |
| `requirements.txt` | Python deps installed on top of the base image |
| `README.md` | This file |

---

## Required Task Configuration

The bundled `player.config` already sets these correctly. If you create a new
task in the OrbitLab dashboard, mirror these settings:

```json
{
  "container": "docker",
  "type": "app",
  "entry": "crop_health_app.py",
  "container_image": "tm2space/aicube-base:latest",
  "bands": ["blue", "green", "red", "nir-2"],
  "aoi_exec": [
    {
      "image_format": "tiff",
      "guidance": [...],
      ...
    }
  ]
}
```

**Why TIFF:** the script does NDVI / NDRE / NDWI / SAVI on raw reflectance
values. JPEG/PNG would gamma-compress and silently truncate beyond 3 channels;
raw is a header-less binary blob rasterio can't open. TIFF preserves both the
bit depth and the per-band layout the loader and model expect.

**Optional band:** add `"red-edge"` to the `bands` list to enable the full
5-channel model path and the NDRE index. Without it the model's red-edge
channel stays at zero and `compute_vegetation_indices()` skips NDRE.

---

## Quick Start

1. **Download** `crop_health_app_task.bz2` from the dashboard's
   *Developer Examples & Templates → Docker* tab.
2. **Unpack:**
   ```bash
   tar -xjf crop_health_app_task.bz2
   cd crop_health_app
   ```
3. **Upload** the unpacked folder via the OrbitLab dashboard's task creation
   flow. The dashboard packages it into a satellite uplink package and queues
   the task.
4. **Wait** for the task to run on the AICube; results appear in the
   dashboard's task-detail view.

---

## Building the Docker Image Locally

To test before submitting to OrbitLab, or to extend the app:

```bash
# 1. Build the image
docker build -t my-crop-health-app .

# 2. Run it with mock input/output dirs
docker run --rm \
    -v $(pwd)/data:/opt/ilc_player/data \
    -v $(pwd)/results:/opt/ilc_player/results \
    -e ILC_INPUT_DIR=/opt/ilc_player/data \
    -e ILC_OUTPUT_DIR=/opt/ilc_player/results \
    my-crop-health-app

# 3. To create an uplink package for OrbitLab submission, use the helper
#    script from the sample_apps repo:
./create-uplink-package.sh my-crop-health-app crop_health_uplink.tar.gz
```

The Dockerfile inherits from `tm2space/aicube-base:latest`, which already
includes Python 3, PyTorch, TensorFlow, ONNX Runtime, GDAL, Rasterio, PyProj,
NumPy, OpenCV, scikit-image, and SciPy. `requirements.txt` re-pins what this
app uses for transparency.

---

## Output Format

All outputs land in `{ILC_OUTPUT_DIR}` (defaults to `/opt/ilc_player/results`).

### `crop_health_results.json`

```json
{
  "task": "crop_health_analysis",
  "timestamp": "2026-05-08T12:34:56+00:00",
  "images_processed": 4,
  "image_names": ["img_0", "img_1", "img_2", "img_3"],
  "model": "eurosat_resnet18",
  "resolution_m": 10.0,
  "analysis_region": {
    "bounds": {"north": 22.61, "south": 22.47, "east": 87.93, "west": 87.78},
    "total_area_hectares": 32400.0
  },
  "land_classification": {
    "total_patches": 7,
    "classes": {
      "AnnualCrop":           {"pixels": 1340000, "area_hectares":  4020.0, "percentage": 12.4},
      "Forest":               {"pixels":  890000, "area_hectares":  2670.0, "percentage":  8.2},
      "HerbaceousVegetation": {"pixels": 2110000, "area_hectares":  6330.0, "percentage": 19.5},
      ...
    }
  },
  "crop_health": {
    "total_crop_area_hectares": 7230.0,
    "crop_percentage_of_image": 22.3,
    "vegetation_indices": {
      "ndvi": {"mean": 0.4521, "min": -0.0123, "max": 0.8901, "std": 0.1233},
      "savi": {...}, "ndwi": {...}, "ndre": {...}
    },
    "health_zones": {
      "Healthy":         {"pixels": 1820000, "area_hectares": 5460.0, "percentage": 75.5},
      "Moderate Stress": {"pixels":  340000, "area_hectares": 1020.0, "percentage": 14.1},
      "Stressed":        {"pixels":  180000, "area_hectares":  540.0, "percentage":  7.5},
      "Severe Stress":   {"pixels":   70000, "area_hectares":  210.0, "percentage":  2.9}
    },
    "overall_health_score": 78.4
  },
  "advisories": [
    {
      "priority": "MEDIUM",
      "category": "monitoring",
      "message": "Moderate stress in 14.1% of crop area. Monitor closely over the next 1-2 weeks for progression.",
      "affected_area_hectares": 1020.0
    },
    ...
  ]
}
```

### `crop_health_annotated.png`

Side-by-side composite: **Land Classification** (left panel) and **Crop
Health** (right panel), each with a colour-coded legend. Built from the image
with the largest crop coverage so the preview is informative.

### `crop_health_report.txt`

Human-readable text report mirroring the JSON, suitable for tailing in the
dashboard's task-detail view or pasting into operational reports.

### `cust_output.txt`

Timestamped log file (mirrors what's printed to stdout). Configured by
`AICubeImageLoader.get_logger("crop_health")`.

---

## AICubeImageLoader API Reference (methods used by this app)

The loader is a single-file helper that handles every interaction with the
AICube task pipeline. This app uses the following surface — see the loader's
own docstring for the full API.

### Construction & lifecycle

#### `AICubeImageLoader(data_dir=None, output_dir=None, config_path=None, max_wait=300, poll_interval=5)`
Reads `player.config`, scans the data directory, sums `guidanceTargets`
across AOIs to know how many images to expect.

- `data_dir` — defaults to `ILC_INPUT_DIR` env var, then `/opt/ilc_player/data`
- `output_dir` — defaults to `ILC_OUTPUT_DIR` env var, then `/opt/ilc_player/results`
- `max_wait` — total seconds `iter_images()` will wait between arrivals before giving up
- `poll_interval` — seconds between filesystem polls inside `wait_for_next_image()`

#### `loader.ensure_output_dir() → str`
Creates `output_dir` if it doesn't exist. Returns the path. Idempotent.

### Band introspection

#### `loader.selected_bands → list[str]`
Dashboard band ids selected for this task, in the order saved by the dashboard
(e.g. `['blue', 'green', 'red', 'nir-2']`). Falls back to RGB if `player.config`
has no `bands` field.

#### `loader.selected_bands_s2 → list[str]`
Same list, translated to Sentinel-2 names (`['B02', 'B03', 'B04', 'B08']`).
Used by this app to drive `BAND_TO_MODEL_CHANNEL` lookups since the EuroSAT
model speaks Sentinel-2 names internally.

#### `loader.require_bands(required) → dict[str, int]`
Verifies every band id in `required` is present. Returns a `{band_id: axis_index}`
map. Raises `ValueError` listing the missing bands if any are absent.

```python
loader.require_bands(['blue', 'green', 'red', 'nir-2'])
# → {'blue': 0, 'green': 1, 'red': 2, 'nir-2': 3}
```

This app calls it inside `main()` to fail fast before model loading kicks in.

### Iteration

#### `loader.iter_images(load=True, validate_bands=None, dtype="float32") → Iterator[ImageRecord]`
Generator that yields one `ImageRecord` per arriving capture. Replaces the
manual `while not loader.all_images_received: ...` loop. Internally:

- Calls `wait_for_next_image()` (blocks up to `max_wait`) to get the next path
- If `validate_bands` is set, checks the sidecar's `bands_saved` field and
  skips images that don't include every required band
- If `load=True`, opens the GeoTIFF via `read_image()` and decodes it
- Logs and skips images that fail to decode

```python
for record in loader.iter_images(validate_bands=['blue','green','red','nir-2']):
    log.info(f"[{record.index}/{record.total}] {record.name}")
    process(record.data, record.geo, record.metadata)
```

### `ImageRecord` dataclass

| Field | Type | Description |
|---|---|---|
| `path` | str | Absolute path to the image file |
| `name` | str | Base filename without extension (e.g. `img_3`) |
| `aoi_name` | str | Immediate parent directory (the AOI subfolder) |
| `index` | int | 1-based, matches `loader.processed_count` |
| `total` | int | Total expected guidance targets (from player.config) |
| `metadata` | dict \| None | Parsed sidecar JSON (`bbox`, `polygon`, `capture_date`, …) |
| `data` | np.ndarray \| None | `(bands, H, W)` float32 raster (None if `load=False`) |
| `geo` | dict \| None | `{transform, crs, bounds, width, height}` (None if `load=False`) |

### Image / band helpers

#### `loader.band(data, band_id, axis=0) → np.ndarray`
Slice one band out of a multi-band array. Accepts both dashboard ids
(`'blue'`) and Sentinel-2 names (`'B02'`). For tile arrays whose band axis
is 1 (shape `(N, C, H, W)`), pass `axis=1`.

```python
green = loader.band(record.data, 'green')         # (H, W)
green = loader.band(record.data, 'B03')           # same thing
band  = loader.band(tiles, 'B02', axis=1)         # (N, H, W) for tile batches
```

#### `loader.bands(data, band_ids, axis=0, stack_axis=-1) → np.ndarray`
Stack multiple bands. Default `stack_axis=-1` returns `(H, W, N)` — what
OpenCV/PIL want for an RGB display image. Use `stack_axis=0` to get
`(N, H, W)` if you're going to do channel-wise math.

```python
rgb = loader.bands(record.data, ['B04', 'B03', 'B02'])   # (H, W, 3) for cv2
```

#### `loader.estimate_gsd_meters(geo) → float | None`
Reads the affine transform's pixel-width. Returns metres-per-pixel when the
CRS is projected; `None` otherwise (e.g. geographic EPSG:4326 — degrees,
not metres). This app uses it to set `CONFIG["resolution_m"]` for accurate
per-class hectare math.

### Output writers

#### `loader.write_json(filename, data, indent=2) → str`
JSON-dump `data` to `{output_dir}/{filename}`. Creates `output_dir` if
needed. Built-in serialiser handles `datetime` and NumPy scalars/arrays.
Returns the full path written.

#### `loader.write_text(filename, text) → str`
Plain-text write to `{output_dir}/{filename}`. Returns the full path.

#### `loader.output_dir → str`
The result directory path (no side effects). Used in this app as
`os.path.join(loader.output_dir, "crop_health_annotated.png")` for the
OpenCV `imwrite` since cv2 can't go through `write_text`.

### Logger

#### `AICubeImageLoader.get_logger(name="aicube", log_file=None, output_dir=None) → logging.Logger`
Returns a stdlib logger writing to **stdout AND** `{output_dir}/cust_output.txt`.
Idempotent — repeated calls with the same name return the configured logger
without duplicate handlers. Format matches what the dashboard's task-detail
view parses.

```python
log = AICubeImageLoader.get_logger("crop_health")
log.info("Starting...")
```

---

## EuroSAT Model Details

| Property | Value |
|---|---|
| Architecture | ResNet18 |
| Classes | 10 (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`) |
| Input shape | `(N, 5, 64, 64)` — channels are `[B02, B03, B04, B05, B08]` in that order |
| Input range | float32 in `[0, 1]` (Sentinel-2 L2A reflectance / 10000) |
| Output | logits `(N, 10)` — softmax for class probabilities |
| File size | ~43 MB |
| Training data | EuroSAT (27,000 Sentinel-2 patches) |

The model expects the 5-channel layout even when the user has only captured
4 bands. `prepare_model_input()` zero-fills missing channels rather than
erroring — accuracy degrades a bit but inference still produces sensible
outputs.

---

## Customising / Extending

### Use a different model
Replace `eurosat_resnet18.onnx` and update three things:
1. `EUROSAT_CLASSES` — class names in the order the model emits them
2. `CROP_CLASSES` — which of those count as "crop" for downstream analysis
3. `BAND_TO_MODEL_CHANNEL` + `MODEL_INPUT_CHANNELS` — input layout

### Adjust health thresholds
Edit the `np.interp` breakpoints in `compute_health_scores()`. The defaults
were tuned for typical mid-latitude row crops on Sentinel-2 L2A surface
reflectance — arid regions, rice paddies, etc. may want different cutoffs.

### Add a new vegetation index
Add the band-arithmetic to `compute_vegetation_indices()` (it returns a
dict, just add a key) and an interp table to `compute_health_scores()`. If
you want it in the weighted health score, also add a weight in `CONFIG`.

### Add a new advisory rule
Copy any of the `if ... advisories.append(...)` blocks in
`generate_advisories()` and adjust the trigger condition / message /
priority. The list is sorted HIGH → MEDIUM → LOW automatically.

---

## Troubleshooting

**`FATAL: Required bands [...] are missing from player.config`**
The dashboard task wasn't configured with the bands this app needs. Re-create
the task with `bands: ["blue", "green", "red", "nir-2"]` (and optionally
`"red-edge"`).

**`ERROR: ONNX model not found at /workspace/eurosat_resnet18.onnx`**
The model wasn't included in the docker image. The bundled `Dockerfile`
COPYs it from the build context — make sure `eurosat_resnet18.onnx` sits next
to the Dockerfile when you build.

**`ONNX model loaded, providers: ['CPUExecutionProvider']`**
CUDA isn't available in the runtime. Inference will be ~10x slower but still
correct. If you're testing locally and want CUDA, install
`onnxruntime-gpu` instead of `onnxruntime`.

**Image has wrong band count / `band 'X' is not in selected bands`**
The capture probably wasn't TIFF — JPEG/PNG truncate beyond 3 channels.
Confirm `image_format: "tiff"` in `player.config.aoi_exec[*]`.

**`Less than 1% crop area detected`**
Either the scene genuinely has no crops, or the EuroSAT model is misclassifying
your specific land cover. Check `crop_health_annotated.png` to see how the
model labelled the scene; you may need to fine-tune for your region.

---

## Repackaging After Modifications

When you change anything in this directory, refresh the dashboard download:

```bash
# from the parent of this directory:
tar -cjf crop_health_app_task.bz2 crop_health_app/
```

Drop the resulting bz2 into `orbitlab-dashboard/public/sample-aic-apps/`,
replacing the old file. The `last_updated` field in `ExamplesDialog.tsx`
should be bumped to today's date.
