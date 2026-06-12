# AeroSeg — Aerial Land Cover Segmentation Pipeline

> **Build Spec for AI Coding Tool** — Build every file described here from scratch, exactly as specified.

---

## Project Summary

A deep learning pipeline that performs **semantic segmentation on satellite/aerial imagery** to classify land surface types (vegetation, impervious/pavement, buildings, water, agriculture, barren ground). Segmented regions are exported as **GeoJSON polygons** with per-class area measurements — ready for any GIS tool. A Streamlit web app provides end-to-end demo capability.

**This is a general-purpose remote sensing tool. It is NOT specific to any one company or use case.**

---

## Tech Stack

| Layer | Tool / Version |
|---|---|
| Deep Learning | PyTorch ≥ 2.0 |
| Segmentation Model | segmentation-models-pytorch ≥ 0.3.3 |
| Augmentation | Albumentations ≥ 1.3 |
| Image Processing | OpenCV-headless ≥ 4.7 |
| GIS / Geospatial | Rasterio ≥ 1.3, GeoPandas ≥ 0.13, Shapely ≥ 2.0 |
| Metrics | torchmetrics ≥ 1.0 |
| Visualization | Folium ≥ 0.14, Matplotlib ≥ 3.7 |
| Frontend | Streamlit ≥ 1.25 |
| Utilities | tqdm, PyYAML, pandas, numpy, pathlib |

---

## Dataset

**DeepGlobe Land Cover Classification Challenge**
- **Source**: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
- **Free**: Yes (Kaggle account required)
- **Size**: 803 RGB satellite images at 2448×2448px with pixel-wise RGB label masks
- **Split**: Use 80% train / 10% val / 10% test (random seed 42)

### Class Definitions (RGB mask → class index)

| Class | RGB in Mask | Index |
|---|---|---|
| Urban / Impervious | (0, 255, 255) | 0 |
| Agriculture | (255, 255, 0) | 1 |
| Rangeland | (255, 0, 255) | 2 |
| Forest / Vegetation | (0, 255, 0) | 3 |
| Water | (0, 0, 255) | 4 |
| Barren Land | (255, 255, 255) | 5 |
| Unknown | (0, 0, 0) | — ignored (mask value = 255, excluded from loss) |

---

## Architecture Overview

```
[Input: RGB Satellite Image (any size)]
            ↓
[Preprocessing: Resize → 512×512, Normalize with encoder stats]
            ↓
[U-Net (ResNet34 encoder, pretrained ImageNet)]
            ↓
[Output: Segmentation mask (512×512, 6 classes)]
            ↓
[Post-processing: Morphological closing, remove small blobs < 50px]
            ↓
[Polygonization: rasterio.features.shapes() per class]
            ↓
[GeoDataFrame: class_id, class_name, area_m2, geometry (Polygon/MultiPolygon)]
            ↓
[Export: GeoJSON FeatureCollection  |  Shapefile]
            ↓
[Streamlit UI: Upload → Infer → Overlay → Download GeoJSON]
```

---

## Project File Structure

```
aeroseg/
├── data/
│   ├── raw/                        # Unzipped DeepGlobe dataset goes here
│   │   ├── train/                  # Contains *_sat.jpg and *_mask.png pairs
│   │   └── valid/
│   ├── processed/                  # Output of data_prep.py (resized tiles)
│   └── samples/                    # 5 demo images committed to repo (pre-inferred)
├── checkpoints/                    # Saved model weights (.pth files)
├── logs/
│   └── metrics.csv                 # Training metrics (epoch, train_loss, val_loss, val_miou)
├── outputs/                        # Inference outputs (masks, overlays, GeoJSON)
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gis_export.py
├── app.py                          # Streamlit demo
├── data_prep.py                    # Dataset preparation script
├── config.yaml
├── requirements.txt
└── README.md
```

---

## `config.yaml` — Full Configuration

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  samples_dir: "data/samples"
  img_size: 512
  num_classes: 6
  ignore_index: 255
  random_seed: 42
  train_split: 0.8
  val_split: 0.1

model:
  architecture: "unet"
  encoder: "resnet34"
  encoder_weights: "imagenet"
  in_channels: 3

training:
  epochs: 30
  batch_size: 8
  learning_rate: 0.0001
  weight_decay: 0.0001
  grad_clip: 1.0
  mixed_precision: true
  checkpoint_dir: "checkpoints"
  log_file: "logs/metrics.csv"

inference:
  default_checkpoint: "checkpoints/best.pth"
  output_dir: "outputs"
  overlay_alpha: 0.45
  min_polygon_area_px: 50

gis:
  default_resolution_m: 0.5      # metres per pixel (assumed when no georef)
  simplify_tolerance: 0.5
  export_formats: ["geojson", "shapefile"]

class_names:
  - "Urban/Impervious"
  - "Agriculture"
  - "Rangeland"
  - "Forest/Vegetation"
  - "Water"
  - "Barren Land"

class_colors_rgb:               # For overlay visualization
  - [0, 255, 255]
  - [255, 255, 0]
  - [255, 0, 255]
  - [0, 255, 0]
  - [0, 0, 255]
  - [255, 255, 255]
```

---

## `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0
opencv-python-headless>=4.7.0
rasterio>=1.3.6
geopandas>=0.13.0
shapely>=2.0.0
folium>=0.14.0
streamlit>=1.25.0
torchmetrics>=1.0.0
matplotlib>=3.7.0
pandas>=2.0.0
pyyaml>=6.0
numpy>=1.24.0
tqdm>=4.65.0
Pillow>=9.5.0
watchdog>=3.0.0
```

---

## File-by-File Implementation Spec

### `data_prep.py`

**Purpose**: One-time preparation of the raw DeepGlobe dataset.

**Logic**:
1. Walk `data/raw/train/` and `data/raw/valid/`; collect all `*_sat.jpg` + matching `*_mask.png` pairs.
2. Resize each pair to 512×512 using `cv2.INTER_AREA` for images, `cv2.INTER_NEAREST` for masks.
3. Save to `data/processed/{split}/images/` and `data/processed/{split}/masks/` as PNG.
4. Implement train/val/test split using `random.seed(42)` before shuffling.
5. Print summary: number of images per split.

**CLI**: `python data_prep.py --config config.yaml`

---

### `src/dataset.py`

**Class**: `DeepGlobeDataset(torch.utils.data.Dataset)`

**Constructor args**: `split` (str: train/val/test), `config` (dict), `augment` (bool)

**`__getitem__`**:
- Load image (RGB, float32, divide by 255)
- Load mask PNG → convert to class index using exact RGB lookup table from config
- Unknown pixels (0,0,0) → set to `ignore_index` (255)
- Apply Albumentations pipeline if `augment=True`
- Apply encoder preprocessing normalization (mean/std from ImageNet)
- Return: `{"image": tensor(3,H,W), "mask": tensor(H,W, dtype=torch.long)}`

**Augmentation pipeline (train only)**:
```
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5)
RandomBrightnessContrast(p=0.3)
```

**Helper function**: `get_dataloaders(config) -> (train_loader, val_loader, test_loader)`
- `num_workers=4`, `pin_memory=True`, `shuffle=True` for train only

---

### `src/model.py`

**Function**: `get_model(config) -> nn.Module`
- Returns: `smp.Unet(encoder_name=config["encoder"], encoder_weights=config["encoder_weights"], in_channels=3, classes=config["num_classes"], activation=None)`

**Function**: `get_preprocessing_fn(config) -> callable`
- Returns: `smp.encoders.get_preprocessing_fn(config["encoder"], config["encoder_weights"])`

**Function**: `load_checkpoint(model, checkpoint_path, device) -> model`
- Loads state dict, maps to device, sets model to eval mode

---

### `src/train.py`

**Entry point**: `python src/train.py --config config.yaml`

**Training loop**:
- Device: auto-detect CUDA → MPS → CPU
- Loss: `combined_loss = 0.5 * ce_loss + 0.5 * dice_loss`
  - `ce_loss = nn.CrossEntropyLoss(ignore_index=255)`
  - `dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=255)`
- Optimizer: `AdamW(lr, weight_decay)`
- Scheduler: `CosineAnnealingLR(T_max=epochs)`
- Mixed precision: `torch.cuda.amp.GradScaler` + `autocast` (skip on CPU/MPS)
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

**Per epoch**:
- Train: forward → loss → backward → step → scheduler step
- Val: `torch.no_grad()` → compute val loss + val mIoU
- Save checkpoint if val mIoU improves (save as `checkpoints/best.pth`)
- Always save latest as `checkpoints/last.pth`
- Append row to `logs/metrics.csv`: `epoch, train_loss, val_loss, val_miou`
- Print: `Epoch {e}/{total} | Train Loss: {x:.4f} | Val Loss: {y:.4f} | Val mIoU: {z:.4f}`

---

### `src/evaluate.py`

**Entry point**: `python src/evaluate.py --config config.yaml --split test --checkpoint checkpoints/best.pth`

**Function**: `compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> dict`

Returns:
```python
{
  "per_class_iou": [float] * num_classes,    # torchmetrics.JaccardIndex per class
  "miou": float,
  "per_class_dice": [float] * num_classes,   # torchmetrics.Dice per class
  "mean_dice": float,
  "per_class_f1": [float] * num_classes,
  "macro_f1": float,
  "accuracy": float                          # torchmetrics.Accuracy
}
```

**Function**: `evaluate_dataset(model, dataloader, config, device) -> dict`
- Accumulates predictions over full split
- Prints formatted table:
  ```
  Class               IoU     Dice    F1
  Urban/Impervious   0.723   0.839   0.839
  Agriculture        0.811   0.896   0.896
  ...
  ─────────────────────────────────────
  Mean (mIoU)        0.748   0.851   0.851
  Accuracy:          0.923
  ```

---

### `src/predict.py`

**Function**: `predict_image(image_input, model, config, device) -> dict`
- `image_input`: file path (str/Path) or numpy array (H×W×3)
- Preprocess: resize to 512×512, normalize
- Run inference
- Resize mask back to original image dimensions using `cv2.INTER_NEAREST`

Returns:
```python
{
  "original_image": np.ndarray,    # H×W×3 uint8
  "predicted_mask": np.ndarray,    # H×W int (class indices)
  "overlay": np.ndarray,           # H×W×3 uint8 (colored overlay blended)
  "class_pixel_counts": dict,      # {class_name: pixel_count}
  "class_area_pct": dict           # {class_name: percentage of image}
}
```

**Overlay generation**: For each class, create binary mask → colorize using config RGB → blend with original image at `alpha` from config.

**CLI**: `python src/predict.py --image path/to/image.jpg --checkpoint checkpoints/best.pth --config config.yaml --output outputs/`
- Saves: `{stem}_mask.png`, `{stem}_overlay.png`, `{stem}_areas.json`

---

### `src/gis_export.py`

**Function**: `mask_to_geojson(mask: np.ndarray, config: dict, transform=None, crs: str = "EPSG:4326") -> str`

**Logic**:
1. If `transform` is None, create dummy affine: `rasterio.transform.from_bounds(0, 0, mask.shape[1]*res, mask.shape[0]*res, mask.shape[1], mask.shape[0])` where `res = config["gis"]["default_resolution_m"]`
2. For each class index (0 to num_classes-1):
   a. Create binary mask: `(mask == class_idx).astype(np.uint8)`
   b. Run `list(rasterio.features.shapes(binary_mask, transform=transform))` to get polygon-value pairs
   c. Keep only polygons with value == 1
   d. Convert each to Shapely geometry via `shapely.geometry.shape()`
   e. Filter: drop polygons with area < `config["inference"]["min_polygon_area_px"]` × resolution²
   f. Simplify: `geom.simplify(config["gis"]["simplify_tolerance"])`
3. Build GeoDataFrame with columns: `class_id` (int), `class_name` (str), `area_m2` (float), `geometry`
4. Set CRS: `gdf.set_crs(crs)`
5. Return `gdf.to_json()` (GeoJSON FeatureCollection string)

**Function**: `export_geojson(gdf: gpd.GeoDataFrame, output_path: Path) -> None`

**Function**: `export_shapefile(gdf: gpd.GeoDataFrame, output_path: Path) -> None`
- Truncate column names to ≤10 chars for Shapefile compatibility

**Function**: `compute_area_stats(gdf: gpd.GeoDataFrame) -> pd.DataFrame`
- Returns DataFrame: class_name, total_area_m2, polygon_count, pct_of_total

---

## `app.py` — Streamlit Application

**Page config**: `st.set_page_config(title="AeroSeg", layout="wide", page_icon="🛰️")`

**Sidebar**:
- Model checkpoint selector: dropdown from `checkpoints/*.pth` files
- Overlay opacity slider: 0.1 → 0.9 (default 0.45)
- Class visibility: multi-select checkboxes for the 6 classes
- "About" section with brief project description

**Main area — 3 tabs**:

#### Tab 1: "Inference"
1. `st.file_uploader("Upload satellite/aerial image", type=["jpg", "jpeg", "png", "tif"])`
2. On upload: display original image in col1
3. "Run Segmentation" button → call `predict_image()` → call `mask_to_geojson()`
4. col2: display overlay image
5. Below: two columns:
   - col1: Matplotlib horizontal bar chart of class area percentages (use class colors from config)
   - col2: DataFrame table with class_name, area_m2, pct columns
6. Download buttons:
   - `st.download_button("Download Overlay (PNG)", ...)`
   - `st.download_button("Download Mask (PNG)", ...)`
   - `st.download_button("Download GeoJSON", ...)`
7. Expandable section: raw GeoJSON preview (`st.code(geojson_str[:2000], language="json")`)

#### Tab 2: "Training Metrics"
1. Check if `logs/metrics.csv` exists
2. If yes: load it and plot with Matplotlib:
   - Dual-axis line chart: Loss (train + val) on left y-axis, mIoU on right y-axis, x-axis = epoch
3. If no: `st.info("No training log found. Run training first.")`
4. Show sample images from `data/samples/` with pre-computed overlays if available

#### Tab 3: "How It Works"
- Static text explaining the pipeline (3–4 paragraphs)
- ASCII architecture diagram in a code block
- Links: dataset source, model paper reference (U-Net: Ronneberger et al. 2015)
- Table of evaluation metrics with brief description of each

**State management**: Use `st.session_state` to cache model between runs (load once, reuse).

---

## `README.md` Content Requirements

Include all of the following sections:

1. **Header**: Project name, one-line description, badges (Python version, PyTorch, License: MIT)
2. **Overview**: What the project does and why it matters (3–4 sentences)
3. **Architecture**: ASCII diagram (same as in app.py Tab 3)
4. **Quickstart**:
   ```bash
   git clone <repo>
   cd aeroseg
   pip install -r requirements.txt
   # Download dataset from Kaggle, unzip to data/raw/
   python data_prep.py --config config.yaml
   python src/train.py --config config.yaml
   streamlit run app.py
   ```
5. **Dataset Setup**: Step-by-step Kaggle download instructions
6. **Training**: Command + expected output snippet
7. **Evaluation**: Command + sample results table (use placeholder values)
8. **Inference + GeoJSON Export**:
   ```bash
   python src/predict.py --image my_image.jpg --checkpoint checkpoints/best.pth --config config.yaml
   ```
9. **GIS Output**: Example GeoJSON snippet showing polygon structure with class_name and area_m2
10. **Results**: Placeholder table (fill after training)
11. **Project Structure**: Tree diagram
12. **License**: MIT

---

## Implementation Rules for the AI Coding Tool

1. **All paths via `pathlib.Path`** — no `os.path` anywhere
2. **Type hints** on every function signature
3. **Docstrings** on every function (one-line summary + Args/Returns)
4. **Device detection**:
   ```python
   device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
   ```
5. **Config loading**: Every script accepts `--config config.yaml` and parses with `yaml.safe_load()`
6. **RGB-to-class lookup**: Use exact integer tuple matching, not approximate/nearest-color
7. **GeoJSON export** must work with no real georeferencing — dummy transform fallback is mandatory
8. **Training** must run (slowly) on CPU — no CUDA-only ops
9. **Streamlit app** must function without a trained checkpoint: show a warning and use `data/samples/` demo images
10. **No hardcoded paths** — everything from config or CLI args
11. **Error handling**: Wrap file I/O and model inference in try/except with clear error messages
12. **tqdm** progress bars in all loops (training, evaluation, data preparation)
13. `data/samples/` must contain 5 placeholder sample images downloadable from public domain (use OpenAerialMap or similar links in README, or add a `download_samples.py` helper script)

---

## Form-Filling Guide (Internal Reference — Not Part of Codebase)

> Use this to fill in the recruitment form after building the project.

| Form Field | Answer |
|---|---|
| CV Frameworks used | **PyTorch**, **OpenCV** |
| CV tasks worked on | **Semantic Segmentation** |
| Models trained/fine-tuned | **U-Net** |
| Describe a CV project | Describe AeroSeg: satellite image segmentation → GeoJSON export |
| Technically challenging problem | Polygonizing segmentation masks into GIS-compatible GeoJSON with correct topology, simplification, and area measurements using Rasterio + GeoPandas, without real georeferencing data |
| GIS-compatible format conversion | **Yes** — GeoJSON, Shapefile |
| Geospatial technologies | **GeoJSON**, **GeoPandas**, **Rasterio** |
| Imagery types worked with | **Satellite Imagery**, **Aerial Photography** |
| Evaluation metrics | **IoU**, **Dice Coefficient**, **F1 Score**, **Accuracy** |

---

*End of spec. Build all files from scratch as described above.*
