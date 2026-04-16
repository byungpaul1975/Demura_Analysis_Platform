# -*- coding: utf-8 -*-
"""
Run Strict 3x3 Algorithm - 0 or 1 pixel gap only
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper, AreaSumResizer

OUTPUT_DIR = "output"
TARGET_SIZE = (2412, 2288)

print("=" * 60)
print("Strict 3x3 Algorithm (0 or 1 pixel gap only)")
print("=" * 60)

# 1. Load and process image
print("\n[Step 1] Loading image...")
image = tifffile.imread("data/G32_cal.tif")
print(f"  Image size: {image.shape[1]} x {image.shape[0]}")

# 2. ROI Detection
print("\n[Step 2] ROI Detection...")
detector = AdaptiveROIDetector()
roi = detector.detect_by_crossing(image)

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

# 4. 2D Grid Tracking Resize
print("\n[Step 4] 2D Grid Tracking Resize (Zigzag)...")
resizer = AreaSumResizer(show_progress=True)
result, centers_x, centers_y = resizer.resize_2d_grid_tracking(warped, TARGET_SIZE)
print(f"  Result size: {result.shape[1]} x {result.shape[0]}")

# 5. Save result
print("\n[Step 5] Saving results...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "31_strict_3x3_mean.tif"),
                 result_16bit, compression='lzw')
print(f"  Saved: 31_strict_3x3_mean.tif")

# 6. Verify pitch distribution
print("\n[Step 6] Verifying pitch distribution...")
pitches_x = np.diff(centers_x, axis=1).flatten()
print(f"  Horizontal pitch distribution:")
print(f"    Min: {pitches_x.min()}")
print(f"    Max: {pitches_x.max()}")
print(f"    Mean: {pitches_x.mean():.2f}")

# Count 3s and 4s
count_3 = np.sum(pitches_x == 3)
count_4 = np.sum(pitches_x == 4)
count_other = np.sum((pitches_x != 3) & (pitches_x != 4))
total = len(pitches_x)
print(f"    Pitch=3 (0 gap): {count_3} ({100*count_3/total:.1f}%)")
print(f"    Pitch=4 (1 gap): {count_4} ({100*count_4/total:.1f}%)")
print(f"    Other: {count_other} ({100*count_other/total:.1f}%)")

# 7. Create visualization with pixel values
print("\n[Step 7] Creating visualization...")

# Show 12x12 output pixels for detailed view
view_size = 12
center_out_y = result.shape[0] // 2
center_out_x = result.shape[1] // 2

out_y_start = center_out_y
out_y_end = center_out_y + view_size
out_x_start = center_out_x
out_x_end = center_out_x + view_size

# Get centers for this region
region_centers_x = centers_x[out_y_start:out_y_end, out_x_start:out_x_end]
region_centers_y = centers_y[out_y_start:out_y_end, out_x_start:out_x_end]

# Calculate input bounds
margin = 3
in_x1 = max(0, region_centers_x.min() - margin)
in_x2 = min(warped.shape[1], region_centers_x.max() + margin)
in_y1 = max(0, region_centers_y.min() - margin)
in_y2 = min(warped.shape[0], region_centers_y.max() + margin)

# Extract regions
input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[out_y_start:out_y_end, out_x_start:out_x_end]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(28, 14))

# Left: Input with pixel values and 3x3 boxes
ax1 = axes[0]
ax1.imshow(input_region, cmap='gray', vmin=0, vmax=input_region.max())

# Add pixel values
in_h, in_w = input_region.shape
for iy in range(in_h):
    for ix in range(in_w):
        val = input_region[iy, ix]
        if val > input_region.max() * 0.7:
            color = 'yellow'
        elif val > input_region.max() * 0.5:
            color = 'white'
        else:
            color = 'gray'
        ax1.text(ix, iy, f'{val:.0f}', ha='center', va='center',
                fontsize=5, color=color, fontweight='bold')

# Draw 3x3 boxes
colors = plt.cm.rainbow(np.linspace(0, 1, view_size * view_size))
for oj in range(view_size):
    for oi in range(view_size):
        out_j = out_y_start + oj
        out_i = out_x_start + oi

        cx = centers_x[out_j, out_i]
        cy = centers_y[out_j, out_i]

        local_x = cx - in_x1
        local_y = cy - in_y1

        color_idx = oj * view_size + oi
        rect = plt.Rectangle(
            (local_x - 1.5, local_y - 1.5), 3, 3,
            fill=False, edgecolor=colors[color_idx], linewidth=2, alpha=0.9
        )
        ax1.add_patch(rect)

        # Mark center with dot
        ax1.plot(local_x, local_y, 'o', markersize=4,
                color=colors[color_idx], markeredgecolor='black', markeredgewidth=0.5)

ax1.set_title(f'Input Image with 3x3 Boxes\n{view_size}×{view_size} output → {in_h}×{in_w} input region\nBoxes have 0 or 1 pixel gap', fontsize=12)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Right: Output with pixel values
ax2 = axes[1]
ax2.imshow(output_region, cmap='gray', vmin=0, vmax=output_region.max())

for oj in range(view_size):
    for oi in range(view_size):
        val = output_region[oj, oi]
        color_idx = oj * view_size + oi
        ax2.text(oi, oj, f'{val:.0f}', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=colors[color_idx], alpha=0.6))

ax2.set_title(f'Output Image (3x3 MEAN)\n{view_size}×{view_size} pixels\nEach pixel = MEAN of 9 input pixels', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "31_strict_3x3_visualization.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 31_strict_3x3_visualization.png")
plt.close()

# 8. Show how boxes connect without gaps
print("\n[Step 8] Creating gap verification image...")

# Smaller region for clearer view
detail_size = 6
detail_centers_x = centers_x[center_out_y:center_out_y+detail_size, center_out_x:center_out_x+detail_size]
detail_centers_y = centers_y[center_out_y:center_out_y+detail_size, center_out_x:center_out_x+detail_size]

in_x1 = max(0, detail_centers_x.min() - 2)
in_x2 = min(warped.shape[1], detail_centers_x.max() + 4)
in_y1 = max(0, detail_centers_y.min() - 2)
in_y2 = min(warped.shape[0], detail_centers_y.max() + 4)

detail_input = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)

fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(detail_input, cmap='gray', vmin=0, vmax=detail_input.max())

# Add pixel values
for iy in range(detail_input.shape[0]):
    for ix in range(detail_input.shape[1]):
        val = detail_input[iy, ix]
        if val > detail_input.max() * 0.7:
            color = 'yellow'
        elif val > detail_input.max() * 0.5:
            color = 'white'
        else:
            color = 'gray'
        ax.text(ix, iy, f'{val:.0f}', ha='center', va='center',
               fontsize=6, color=color, fontweight='bold')

# Draw 3x3 boxes with pitch labels
colors = plt.cm.rainbow(np.linspace(0, 1, detail_size * detail_size))
for oj in range(detail_size):
    for oi in range(detail_size):
        out_j = center_out_y + oj
        out_i = center_out_x + oi

        cx = centers_x[out_j, out_i]
        cy = centers_y[out_j, out_i]

        local_x = cx - in_x1
        local_y = cy - in_y1

        color_idx = oj * detail_size + oi
        rect = plt.Rectangle(
            (local_x - 1.5, local_y - 1.5), 3, 3,
            fill=False, edgecolor=colors[color_idx], linewidth=2.5, alpha=0.95
        )
        ax.add_patch(rect)

        # Show pitch to next box
        if oi < detail_size - 1:
            next_cx = centers_x[out_j, out_i + 1]
            pitch = next_cx - cx
            mid_x = (local_x + (next_cx - in_x1)) / 2
            ax.text(mid_x, local_y - 2.5, f'gap:{pitch-3}',
                   fontsize=8, color='red', ha='center', fontweight='bold')

ax.set_title(f'Gap Verification: 3x3 boxes with 0 or 1 pixel gap\nRed labels show gap size (pitch - 3)', fontsize=14)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "32_gap_verification.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 32_gap_verification.png")
plt.close()

print("\n" + "=" * 60)
print("Strict 3x3 Algorithm Complete!")
print(f"Output: {result.shape[1]} × {result.shape[0]} pixels")
print(f"Pitch distribution: {100*count_3/total:.1f}% pitch=3, {100*count_4/total:.1f}% pitch=4")
print("=" * 60)
