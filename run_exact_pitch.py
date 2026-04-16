# -*- coding: utf-8 -*-
"""
Exact Pitch Algorithm - Use actual 3.88 pitch without rounding
Find bright center within fractional search window
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os
from numba import jit, prange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper, AreaSumResizer

OUTPUT_DIR = "output"
TARGET_SIZE = (2412, 2288)

print("=" * 60)
print("Exact Pitch Algorithm (3.88 pitch)")
print("=" * 60)

# 1. Load and warp image
print("\n[Step 1] Loading and processing image...")
image = tifffile.imread("data/G32_cal.tif")
detector = AdaptiveROIDetector()
roi = detector.detect_by_crossing(image)

warper = PerspectiveWarper()
src_corners = np.array([
    detector._cross_points['TL'],
    detector._cross_points['TR'],
    detector._cross_points['BR'],
    detector._cross_points['BL']
], dtype=np.float32)
warp_result = warper.warp(image, src_corners)
warped = warp_result.image.astype(np.float64)
print(f"  Warped size: {warped.shape[1]} x {warped.shape[0]}")

# Calculate actual pitch
pitch_x = warped.shape[1] / TARGET_SIZE[0]
pitch_y = warped.shape[0] / TARGET_SIZE[1]
print(f"  Actual pitch: X={pitch_x:.4f}, Y={pitch_y:.4f}")

# 2. Exact pitch algorithm
print("\n[Step 2] Running exact pitch algorithm...")

@jit(nopython=True, cache=True)
def exact_pitch_resize(img, target_w, target_h, pitch_x, pitch_y):
    """
    Use exact fractional pitch to find bright pixel centers.

    For each output pixel (i, j):
    - Calculate expected center position: (i * pitch_x, j * pitch_y)
    - Search in 3x3 region around that position for brightest pixel
    - Extract 3x3 centered on brightest pixel
    - Output = MEAN of 3x3
    """
    h, w = img.shape
    result = np.zeros((target_h, target_w), dtype=np.float64)
    centers_x = np.zeros((target_h, target_w), dtype=np.int32)
    centers_y = np.zeros((target_h, target_w), dtype=np.int32)

    for j in range(target_h):
        # Expected y position (fractional)
        expected_y = j * pitch_y

        for i in range(target_w):
            # Expected x position (fractional)
            expected_x = i * pitch_x

            # Search region: 3x3 around expected position
            # But allow +-1 pixel tolerance for zigzag
            y_center = int(round(expected_y))
            x_center = int(round(expected_x))

            # Search in 5x5 region to account for zigzag offset
            best_val = -1.0
            best_y = y_center
            best_x = x_center

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    sy = y_center + dy
                    sx = x_center + dx
                    if 0 <= sy < h and 0 <= sx < w:
                        val = img[sy, sx]
                        if val > best_val:
                            best_val = val
                            best_y = sy
                            best_x = sx

            centers_x[j, i] = best_x
            centers_y[j, i] = best_y

            # Extract 3x3 and calculate MEAN
            y1 = max(0, best_y - 1)
            y2 = min(h, best_y + 2)
            x1 = max(0, best_x - 1)
            x2 = min(w, best_x + 2)

            total = 0.0
            count = 0
            for py in range(y1, y2):
                for px in range(x1, x2):
                    total += img[py, px]
                    count += 1

            if count > 0:
                result[j, i] = total / count

    return result, centers_x, centers_y

import time
start_time = time.time()
result, centers_x, centers_y = exact_pitch_resize(warped, TARGET_SIZE[0], TARGET_SIZE[1], pitch_x, pitch_y)
elapsed = time.time() - start_time
print(f"  Completed in {elapsed:.2f} seconds")
print(f"  Result size: {result.shape[1]} x {result.shape[0]}")

# 3. Save result
print("\n[Step 3] Saving result...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "34_exact_pitch_mean.tif"),
                 result_16bit, compression='lzw')
print(f"  Saved: 34_exact_pitch_mean.tif")

# 4. Verify centers are on bright pixels
print("\n[Step 4] Verifying centers are on bright pixels...")

# Check center region
view_size = 12
center_out_y = result.shape[0] // 2
center_out_x = result.shape[1] // 2

region_centers_x = centers_x[center_out_y:center_out_y+view_size, center_out_x:center_out_x+view_size]
region_centers_y = centers_y[center_out_y:center_out_y+view_size, center_out_x:center_out_x+view_size]

# Get center brightness values
center_values = []
for j in range(view_size):
    for i in range(view_size):
        cx = centers_x[center_out_y + j, center_out_x + i]
        cy = centers_y[center_out_y + j, center_out_x + i]
        center_values.append(warped[cy, cx])

center_values = np.array(center_values)
print(f"  Center pixel brightness:")
print(f"    Min: {center_values.min():.0f}")
print(f"    Max: {center_values.max():.0f}")
print(f"    Mean: {center_values.mean():.0f}")

# 5. Create visualization
print("\n[Step 5] Creating visualization...")

margin = 3
in_x1 = max(0, region_centers_x.min() - margin)
in_x2 = min(warped.shape[1], region_centers_x.max() + margin)
in_y1 = max(0, region_centers_y.min() - margin)
in_y2 = min(warped.shape[0], region_centers_y.max() + margin)

input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[center_out_y:center_out_y+view_size, center_out_x:center_out_x+view_size]

fig, axes = plt.subplots(1, 2, figsize=(28, 14))

# Left: Input with 3x3 boxes
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
        out_j = center_out_y + oj
        out_i = center_out_x + oi

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

        # Mark center
        ax1.plot(local_x, local_y, 'o', markersize=4,
                color=colors[color_idx], markeredgecolor='black', markeredgewidth=0.5)

ax1.set_title(f'Input Image with 3x3 Boxes (Exact Pitch {pitch_x:.3f})\n{view_size}x{view_size} output pixels', fontsize=12)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Right: Output
ax2 = axes[1]
ax2.imshow(output_region, cmap='gray', vmin=0, vmax=output_region.max())

for oj in range(view_size):
    for oi in range(view_size):
        val = output_region[oj, oi]
        color_idx = oj * view_size + oi
        ax2.text(oi, oj, f'{val:.0f}', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor=colors[color_idx], alpha=0.6))

ax2.set_title(f'Output Image (3x3 MEAN)\n{view_size}x{view_size} pixels', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "34_exact_pitch_visualization.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 34_exact_pitch_visualization.png")
plt.close()

# 6. Check for missed bright pixels
print("\n[Step 6] Checking for missed bright pixels...")

# Find bright pixels using local max
from scipy.ndimage import maximum_filter
local_max = maximum_filter(warped, size=3)
is_local_max = (warped == local_max) & (warped > np.percentile(warped, 70))

# Check how many are covered by our 3x3 boxes
covered_mask = np.zeros_like(is_local_max, dtype=bool)

# Mark all 3x3 regions as covered
for j in range(TARGET_SIZE[1]):
    for i in range(TARGET_SIZE[0]):
        cx, cy = centers_x[j, i], centers_y[j, i]
        y1 = max(0, cy - 1)
        y2 = min(warped.shape[0], cy + 2)
        x1 = max(0, cx - 1)
        x2 = min(warped.shape[1], cx + 2)
        covered_mask[y1:y2, x1:x2] = True

# Count missed bright pixels
total_bright = np.sum(is_local_max)
covered_bright = np.sum(is_local_max & covered_mask)
missed_bright = np.sum(is_local_max & ~covered_mask)

print(f"  Total bright pixels (local max): {total_bright}")
print(f"  Covered by 3x3 boxes: {covered_bright} ({100*covered_bright/total_bright:.1f}%)")
print(f"  Missed: {missed_bright} ({100*missed_bright/total_bright:.1f}%)")

print("\n" + "=" * 60)
print("Exact Pitch Algorithm Complete!")
print(f"Output: {result.shape[1]} x {result.shape[0]} pixels")
print("=" * 60)
