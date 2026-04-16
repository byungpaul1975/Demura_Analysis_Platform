# -*- coding: utf-8 -*-
"""
Run Sequential Tracking Algorithm - No bright pixel skipping
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper, AreaSumResizer
import cv2

# Settings
TARGET_SIZE = (2412, 2288)
PITCH_RANGE = (3.7, 4.0)
OUTPUT_DIR = "output"

print("=" * 60)
print("Sequential Tracking Algorithm (No Skipping)")
print("=" * 60)

# 1. Load image
print("\n[Step 1] Loading image...")
image_path = "data/G32_cal.tif"
image = tifffile.imread(image_path)
print(f"  Image size: {image.shape[1]} x {image.shape[0]}")

# 2. ROI Detection
print("\n[Step 2] ROI Detection...")
detector = AdaptiveROIDetector()
roi = detector.detect_by_crossing(image)
print(f"  ROI corners detected")

# 3. Perspective Warp
print("\n[Step 3] Perspective Warp...")
warper = PerspectiveWarper()

src_corners = np.array([
    detector._cross_points['TL'],
    detector._cross_points['TR'],
    detector._cross_points['BR'],
    detector._cross_points['BL']
], dtype=np.float32)

warp_result = warper.warp(image, src_corners)
warped = warp_result.image
print(f"  Warped size: {warped.shape[1]} x {warped.shape[0]}")

# 4. Sequential Tracking Resize
print("\n[Step 4] Sequential Tracking Resize (No Skipping)...")
resizer = AreaSumResizer(show_progress=True)
result, centers_x, centers_y = resizer.resize_sequential_tracking(
    warped, TARGET_SIZE, pitch_range=PITCH_RANGE
)

print(f"  Result size: {result.shape[1]} x {result.shape[0]}")

# 5. Save result
print("\n[Step 5] Saving results...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "28_sequential_tracking_3x3_mean.tif"),
                 result_16bit, compression='lzw')
print(f"  Saved: 28_sequential_tracking_3x3_mean.tif")

# 6. Create visualization - Input vs Output with pixel values and 3x3 boxes
print("\n[Step 6] Creating visualization...")

# Get center region coordinates for 100x100 output pixels
out_h, out_w = result.shape
center_out_y = out_h // 2
center_out_x = out_w // 2
view_size = 50  # Show 50x50 output pixels for clearer visualization

# Output region
out_y_start = center_out_y - view_size // 2
out_y_end = out_y_start + view_size
out_x_start = center_out_x - view_size // 2
out_x_end = out_x_start + view_size

# Get corresponding input region from centers
input_centers_x = centers_x[out_y_start:out_y_end, out_x_start:out_x_end]
input_centers_y = centers_y[out_y_start:out_y_end, out_x_start:out_x_end]

# Calculate input region bounds
in_x_min = max(0, input_centers_x.min() - 5)
in_x_max = min(warped.shape[1], input_centers_x.max() + 5)
in_y_min = max(0, input_centers_y.min() - 5)
in_y_max = min(warped.shape[0], input_centers_y.max() + 5)

# Extract regions
input_region = warped[in_y_min:in_y_max, in_x_min:in_x_max].astype(np.float32)
output_region = result[out_y_start:out_y_end, out_x_start:out_x_end]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(24, 12))

# Left: Input image with 3x3 boxes
ax1 = axes[0]
ax1.imshow(input_region, cmap='gray', vmin=0, vmax=input_region.max())
ax1.set_title(f'Input Image (Warped)\nRegion: [{in_y_min}:{in_y_max}, {in_x_min}:{in_x_max}]', fontsize=12)

# Draw 3x3 boxes for each output pixel
colors = plt.cm.rainbow(np.linspace(0, 1, min(view_size * view_size, 500)))
box_count = 0
for oj in range(min(view_size, 10)):  # Show fewer boxes for clarity
    for oi in range(min(view_size, 10)):
        out_j = out_y_start + oj * 5  # Every 5th row
        out_i = out_x_start + oi * 5  # Every 5th column

        if out_j >= out_y_end or out_i >= out_x_end:
            continue

        cx = centers_x[out_j, out_i]
        cy = centers_y[out_j, out_i]

        # Adjust to local coordinates
        local_x = cx - in_x_min
        local_y = cy - in_y_min

        # Draw 3x3 box
        rect = plt.Rectangle(
            (local_x - 1.5, local_y - 1.5), 3, 3,
            fill=False, edgecolor=colors[box_count % len(colors)], linewidth=1, alpha=0.8
        )
        ax1.add_patch(rect)
        box_count += 1

ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Right: Output image
ax2 = axes[1]
ax2.imshow(output_region, cmap='gray', vmin=0, vmax=output_region.max())
ax2.set_title(f'Output Image (3x3 MEAN)\nSize: {view_size}x{view_size} of {TARGET_SIZE[0]}x{TARGET_SIZE[1]}', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "28_sequential_tracking_visualization.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 28_sequential_tracking_visualization.png")
plt.close()

# 7. Create detailed pixel-level visualization (smaller region)
print("\n[Step 7] Creating detailed pixel-level visualization...")

# Use 10x10 output pixels for detailed view
detail_size = 10
detail_out_y_start = center_out_y
detail_out_y_end = center_out_y + detail_size
detail_out_x_start = center_out_x
detail_out_x_end = center_out_x + detail_size

# Get centers for this region
detail_centers_x = centers_x[detail_out_y_start:detail_out_y_end, detail_out_x_start:detail_out_x_end]
detail_centers_y = centers_y[detail_out_y_start:detail_out_y_end, detail_out_x_start:detail_out_x_end]

# Calculate input bounds with margin
margin = 5
in_x1 = max(0, detail_centers_x.min() - margin)
in_x2 = min(warped.shape[1], detail_centers_x.max() + margin)
in_y1 = max(0, detail_centers_y.min() - margin)
in_y2 = min(warped.shape[0], detail_centers_y.max() + margin)

# Extract detailed regions
detail_input = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
detail_output = result[detail_out_y_start:detail_out_y_end, detail_out_x_start:detail_out_x_end]

fig, axes = plt.subplots(1, 2, figsize=(24, 12))

# Left: Input with 3x3 boxes and pixel values
ax1 = axes[0]
ax1.imshow(detail_input, cmap='gray', vmin=0, vmax=detail_input.max())
ax1.set_title(f'Input Image - 3x3 Boxes\nRegion: y=[{in_y1}:{in_y2}], x=[{in_x1}:{in_x2}]', fontsize=12)

# Draw ALL 3x3 boxes with colors matching output pixels
colors = plt.cm.rainbow(np.linspace(0, 1, detail_size * detail_size))
for oj in range(detail_size):
    for oi in range(detail_size):
        out_j = detail_out_y_start + oj
        out_i = detail_out_x_start + oi

        cx = centers_x[out_j, out_i]
        cy = centers_y[out_j, out_i]

        # Adjust to local coordinates
        local_x = cx - in_x1
        local_y = cy - in_y1

        # Draw 3x3 box
        color_idx = oj * detail_size + oi
        rect = plt.Rectangle(
            (local_x - 1.5, local_y - 1.5), 3, 3,
            fill=False, edgecolor=colors[color_idx], linewidth=2, alpha=0.9
        )
        ax1.add_patch(rect)

        # Mark center pixel
        ax1.plot(local_x, local_y, 'o', markersize=3,
                color=colors[color_idx], markeredgecolor='white', markeredgewidth=0.5)

ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Right: Output with pixel values
ax2 = axes[1]
ax2.imshow(detail_output, cmap='gray', vmin=0, vmax=detail_output.max())
ax2.set_title(f'Output Image (3x3 MEAN)\nEach pixel = MEAN of 9 input pixels', fontsize=12)

# Add pixel values
for oj in range(detail_size):
    for oi in range(detail_size):
        val = detail_output[oj, oi]
        color_idx = oj * detail_size + oi
        ax2.text(oi, oj, f'{val:.0f}', ha='center', va='center',
                fontsize=6, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=colors[color_idx], alpha=0.6))

ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "29_sequential_tracking_detail.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 29_sequential_tracking_detail.png")
plt.close()

# 8. Show pitch analysis
print("\n[Step 8] Analyzing actual pitch distribution...")

# Calculate pitch differences (x direction)
pitch_x = np.diff(centers_x, axis=1).flatten()
pitch_y = np.diff(centers_y, axis=0).flatten()

print(f"  Horizontal pitch (X direction):")
print(f"    Min: {pitch_x.min():.2f}")
print(f"    Max: {pitch_x.max():.2f}")
print(f"    Mean: {pitch_x.mean():.2f}")
print(f"    Std: {pitch_x.std():.2f}")

print(f"  Vertical pitch (Y direction, rows):")
# For row-to-row we look at same column
pitch_rows = []
for i in range(min(100, centers_y.shape[1])):
    col_y = centers_y[:, i]
    pitch_rows.extend(np.diff(col_y))
pitch_rows = np.array(pitch_rows)
print(f"    Min: {pitch_rows.min():.2f}")
print(f"    Max: {pitch_rows.max():.2f}")
print(f"    Mean: {pitch_rows.mean():.2f}")

print("\n" + "=" * 60)
print("Sequential Tracking Complete!")
print(f"Output: {result.shape[1]} x {result.shape[0]} pixels")
print("=" * 60)
