# Demura Analysis Platform

**Display Panel ROI Processing Pipeline**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)

A Python pipeline for processing display panel images captured by mono cameras. It extracts the Region of Interest (ROI), corrects perspective distortion, converts camera-resolution data to display-pixel resolution via area-sum averaging, and outputs 16-bit normalized images.

---

## Table of Contents

- [Features](#features)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Pipeline Details](#pipeline-details)
- [Output Files](#output-files)
- [Documentation Generator](#documentation-generator)
- [Author](#author)
- [License](#license)

---

## Features

- **Image Cropping** — Remove unnecessary border regions to speed up processing
- **ROI Detection** — Automatic display panel boundary detection via adaptive thresholding
- **Perspective Correction** — Tilt correction using `INTER_NEAREST` to preserve original pixel values
- **Area-Sum Resizing** — Fractional-pixel-weighted summation for accurate camera-to-display mapping
- **16-bit Normalization** — Full dynamic range output for downstream analysis
- **Documentation Generator** — Auto-generate PPT with intermediate visualizations

---

## Pipeline Overview

```
 ┌─────────────────────┐
 │   Original Image    │  14144 x 10640 px, 16-bit
 │   (G32_cal.tif)     │
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │   Step 1: Crop      │  Region: (1700,300) → (11700,9900)
 │ (3_image_cropper)   │  → 10000 x 9600 px
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │  Step 2: ROI Detect │  Threshold → Morphology → Contour → Corners
 │ (2_roi_detector)    │  → 9374 x 8894 px, tilt = -0.03°
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │  Step 3: Warp       │  4-corner perspective transform
 │ (4_perspective_     │  INTER_NEAREST (preserves pixel values)
 │      warper)        │  → 9374 x 8894 px (straightened)
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │  Step 4: Area-Sum   │  Camera pixels → Display pixels
 │      Resize         │  Scale: ~3.89x in both axes
 │ (5_area_sum_        │  → 2412 x 2288 px
 │      resizer)       │
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │  Step 5: Normalize  │  Map to [0, 65535] range
 │ (6_image_           │  → 2412 x 2288 px, uint16
 │      normalizer)    │
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │    Final Output     │  06_final_normalized.tif
 │  2412 x 2288 x 16b │
 └─────────────────────┘
```

---

## Project Structure

```
Demura_Analysis_Platform/
├── src/                                 # Core pipeline modules
│   ├── __init__.py                      # Package init & class re-exports
│   ├── 1_utils.py                       # Utility functions (load/save/draw)
│   ├── 2_roi_detector.py                # ROI detection (ROIDetector, AdaptiveROIDetector)
│   ├── 3_image_cropper.py               # Image cropping (ImageCropper, CropRegion)
│   ├── 4_perspective_warper.py          # Perspective warp (PerspectiveWarper)
│   ├── 5_area_sum_resizer.py            # Area-sum resize (AreaSumResizer)
│   ├── 6_image_normalizer.py            # Bit-depth normalization (ImageNormalizer)
│   ├── 7_display_panel_processor.py     # Pipeline orchestrator (DisplayPanelProcessor)
│   └── main.py                          # CLI entry point
│
├── data/
│   └── G32_cal.tif                      # Sample input (16-bit mono camera capture, not tracked)
│
├── output/                              # Generated output files (not tracked)
│
├── docs/                                # Auto-generated documentation
│   ├── run_doc_generator.py             # One-click doc generation
│   ├── generate_images.py               # Pipeline visualization images
│   ├── create_ppt.py                    # PPT builder
│   ├── images/                          # Intermediate step images (14 PNGs)
│   └── ROI_Pipeline_Documentation.pptx  # Generated presentation
│
├── tests/                               # Test scripts
├── notebooks/                           # Jupyter notebooks
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy>=1.21.0 opencv-python>=4.5.0 matplotlib>=3.4.0 scipy>=1.7.0 pillow>=8.0.0
```

Optional (for accelerated area-sum resize):

```bash
pip install numba
```

Optional (for PPT generation):

```bash
pip install python-pptx
```

---

## Quick Start

### Run the Full Pipeline

```bash
# Default settings — processes data/G32_cal.tif → output/
python src/main.py
```

### Custom Input/Output

```bash
python src/main.py --input data/G32_cal.tif --output results/
```

### Custom Display Resolution

```bash
python src/main.py --width 1920 --height 1080
```

### Adjust ROI Detection Threshold

```bash
python src/main.py --threshold 100
```

### Skip Cropping

```bash
python src/main.py --no-crop
```

### Full Example with All Options

```bash
python src/main.py \
    --input data/G32_cal.tif \
    --output output/ \
    --width 2412 \
    --height 2288 \
    --threshold 50 \
    --crop-region 1700,300,11700,9900
```

---

## CLI Reference

| Option                | Short | Default                  | Description                          |
|-----------------------|-------|--------------------------|--------------------------------------|
| `--input`             | `-i`  | `data/G32_cal.tif`       | Input image path                     |
| `--output`            | `-o`  | `output`                 | Output directory                     |
| `--width`             | `-W`  | `2412`                   | Target display width (pixels)        |
| `--height`            | `-H`  | `2288`                   | Target display height (pixels)       |
| `--threshold`         | `-t`  | `50`                     | ROI detection threshold              |
| `--crop-region`       |       | `1700,300,11700,9900`    | Crop region as `x1,y1,x2,y2`        |
| `--no-crop`           |       | *(off)*                  | Disable image cropping               |
| `--no-intermediates`  |       | *(off)*                  | Don't save intermediate files        |

---

## Python API

### Full Pipeline

```python
from src import DisplayPanelProcessor, ProcessingConfig

# Configure the pipeline
config = ProcessingConfig(
    crop_x_start=1700,
    crop_y_start=300,
    crop_x_end=11700,
    crop_y_end=9900,
    use_crop=True,
    roi_threshold=50,
    morph_kernel_size=51,
    display_width=2412,
    display_height=2288,
    output_bit_depth=16,
    save_intermediates=True
)

# Create processor and run
processor = DisplayPanelProcessor(config)
result = processor.process("data/G32_cal.tif")

# Save all outputs
processor.save_results(result, "output/")
processor.save_visualization(result, "output/")

# Access intermediate results
print(f"Original shape : {result.original_image.shape}")
print(f"Cropped shape  : {result.cropped_image.shape}")
print(f"Warped shape   : {result.warped_image.shape}")
print(f"Resized shape  : {result.resized_image.shape}")
print(f"Final shape    : {result.normalized_image.shape}")
print(f"Tilt angle     : {result.stats['tilt_angle']:.4f}°")
print(f"Scale factors  : {result.stats['scale_factors']}")
```

### Individual Modules

Each pipeline step can be used independently:

#### 1. Image Cropping (`3_image_cropper.py`)

```python
from src import ImageCropper, CropRegion
import cv2

image = cv2.imread("data/G32_cal.tif", cv2.IMREAD_UNCHANGED)

cropper = ImageCropper(CropRegion(1700, 300, 11700, 9900))
cropped = cropper.crop(image)

print(f"Original: {image.shape[1]}x{image.shape[0]}")
print(f"Cropped:  {cropped.shape[1]}x{cropped.shape[0]}")
```

#### 2. ROI Detection (`2_roi_detector.py`)

```python
from src import ROIDetector, AdaptiveROIDetector

# Basic detector
detector = ROIDetector(threshold=50, morph_kernel_size=51)
roi = detector.detect(cropped)

if roi is not None:
    print(f"Corners (TL, TR, BR, BL):\n{roi.corners}")
    print(f"Tilt angle : {roi.angle:.4f}°")
    print(f"Dimensions : {roi.width:.0f} x {roi.height:.0f} px")

# Adaptive detector (per-corner threshold optimization)
adaptive = AdaptiveROIDetector(
    initial_threshold=200,
    corner_region_size=60,
    threshold_range=(50, 600),
    display_pixel_pitch=(3.5, 3.5)
)
roi_adaptive = adaptive.detect(cropped)
```

#### 3. Perspective Warp (`4_perspective_warper.py`)

```python
from src import PerspectiveWarper
import cv2

warper = PerspectiveWarper(interpolation=cv2.INTER_NEAREST)
warp_result = warper.warp(cropped, roi.corners)

warped = warp_result.image
print(f"Warped size: {warp_result.width} x {warp_result.height}")
```

#### 4. Area-Sum Resize (`5_area_sum_resizer.py`)

```python
from src import AreaSumResizer

resizer = AreaSumResizer(show_progress=True)
result = resizer.resize(warped, (2412, 2288))

resized = result.image
print(f"Scale: {result.scale_x:.4f} x {result.scale_y:.4f}")
```

#### 5. Normalization (`6_image_normalizer.py`)

```python
from src import ImageNormalizer

normalizer = ImageNormalizer(bit_depth=16)
norm_result = normalizer.normalize(resized)

print(f"Input range  : [{norm_result.original_min:.2f}, {norm_result.original_max:.2f}]")
print(f"Output range : [{norm_result.normalized_min}, {norm_result.normalized_max}]")
```

---

## Pipeline Details

### Key Design Principles

| Principle | Implementation | Why |
|-----------|---------------|-----|
| **Pixel value preservation** | `INTER_NEAREST` interpolation | Measurement accuracy — no blending of values during warp |
| **Area-sum resize** | Weighted pixel summation | Every camera pixel contributes; no data is discarded |
| **Modular architecture** | Independent classes per step | Each module is testable and reusable in isolation |
| **16-bit output** | Full dynamic range normalization | Preserves fine luminance gradients for display analysis |

### Area-Sum Resize Concept

Unlike bilinear/bicubic interpolation, Area-Sum resize accounts for **every camera pixel** within each display pixel's footprint:

```
Camera pixel grid (one display pixel covers ~3.89 x 3.89 camera pixels):

  ┌───┬───┬───┬───┐
  │ A │ B │ C │ D │   Display Pixel Value = mean(A..P)
  ├───┼───┼───┼───┤   weighted by fractional overlap
  │ E │ F │ G │ H │
  ├───┼───┼───┼───┤   → Preserves total light energy
  │ I │ J │ K │ L │   → No aliasing artifacts
  ├───┼───┼───┼───┤
  │ M │ N │ O │ P │
  └───┴───┴───┴───┘
```

---

## Output Files

| File | Description |
|------|-------------|
| `00_cropped_image.tif` | Cropped region from original |
| `04_warped_image.tif` | Perspective-corrected image |
| `05_resized_image.tif` | Area-sum resized to display resolution |
| `06_final_normalized.tif` | **Main output** — 16-bit normalized |
| `06_final_normalized_preview.png` | 8-bit preview for quick viewing |
| `06_final_comparison.png` | Side-by-side comparison of pipeline stages |

---

## Documentation Generator

Auto-generate a 16-slide PPT with intermediate visualizations:

```bash
# Generate images + PPT in one step
python docs/run_doc_generator.py

# Or generate separately
python docs/generate_images.py    # Create docs/images/*.png
python docs/create_ppt.py         # Create docs/ROI_Pipeline_Documentation.pptx
```

---

## Author

**Byung Geun (BG) Jun**
Display Algorithm Architect
MR Display Hardware Team

## License

Internal use only — Meta Platforms, Inc.
