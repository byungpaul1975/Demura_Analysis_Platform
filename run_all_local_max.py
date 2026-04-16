# -*- coding: utf-8 -*-
"""
All Local Maximum Algorithm
1. Find ALL local maximum (bright centers) in input image
2. For each local maximum, calculate which output pixel it belongs to
3. For each output pixel, take the brightest local maximum as center
4. Extract 3x3 around that center and calculate MEAN
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os
from scipy.ndimage import maximum_filter
from numba import jit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper

OUTPUT_DIR = "output"
TARGET_SIZE = (2412, 2288)

print("=" * 60)
print("All Local Maximum Algorithm")
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
h, w = warped.shape
print(f"  Warped size: {w} x {h}")

# Calculate pitch
pitch_x = w / TARGET_SIZE[0]
pitch_y = h / TARGET_SIZE[1]
print(f"  Pitch: X={pitch_x:.4f}, Y={pitch_y:.4f}")

# 2. Find ALL local maximums
print("\n[Step 2] Finding all local maximums...")
local_max = maximum_filter(warped, size=3)
is_local_max = (warped == local_max)

# Also require minimum brightness (top 50% instead of 70%)
threshold = np.percentile(warped, 50)
is_bright_max = is_local_max & (warped > threshold)

max_y, max_x = np.where(is_bright_max)
max_values = warped[max_y, max_x]
print(f"  Found {len(max_y)} local maximums")

# 3. Assign each local maximum to its corresponding output pixel
print("\n[Step 3] Assigning local maximums to output pixels...")

# For each local max, calculate which output pixel it belongs to
out_i = (max_x / pitch_x).astype(np.int32)
out_j = (max_y / pitch_y).astype(np.int32)

# Clamp to valid range
out_i = np.clip(out_i, 0, TARGET_SIZE[0] - 1)
out_j = np.clip(out_j, 0, TARGET_SIZE[1] - 1)

# For each output pixel, find the brightest local max
result = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float64)
centers_x = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.int32)
centers_y = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.int32)
has_center = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=bool)

print("  Assigning brightest local max to each output pixel...")
for idx in range(len(max_y)):
    oi = out_i[idx]
    oj = out_j[idx]
    val = max_values[idx]

    if not has_center[oj, oi] or val > warped[centers_y[oj, oi], centers_x[oj, oi]]:
        centers_x[oj, oi] = max_x[idx]
        centers_y[oj, oi] = max_y[idx]
        has_center[oj, oi] = True

# Count how many output pixels have centers assigned
assigned_count = np.sum(has_center)
total_output = TARGET_SIZE[0] * TARGET_SIZE[1]
print(f"  Output pixels with assigned centers: {assigned_count}/{total_output} ({100*assigned_count/total_output:.1f}%)")

# 4. For pixels without centers, use expected position and search nearby
print("\n[Step 4] Filling in missing centers...")
missing_count = 0

for j in range(TARGET_SIZE[1]):
    for i in range(TARGET_SIZE[0]):
        if not has_center[j, i]:
            # Use expected position
            expected_x = int(round(i * pitch_x))
            expected_y = int(round(j * pitch_y))

            # Search in 5x5 region for brightest pixel
            best_val = -1.0
            best_x = expected_x
            best_y = expected_y

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    sy = expected_y + dy
                    sx = expected_x + dx
                    if 0 <= sy < h and 0 <= sx < w:
                        val = warped[sy, sx]
                        if val > best_val:
                            best_val = val
                            best_x = sx
                            best_y = sy

            centers_x[j, i] = best_x
            centers_y[j, i] = best_y
            missing_count += 1

print(f"  Filled {missing_count} missing centers")

# 5. Calculate 3x3 MEAN for each output pixel
print("\n[Step 5] Calculating 3x3 MEAN...")

@jit(nopython=True, cache=True)
def calculate_means(warped, centers_x, centers_y, result):
    h, w = warped.shape
    target_h, target_w = result.shape

    for j in range(target_h):
        for i in range(target_w):
            cx = centers_x[j, i]
            cy = centers_y[j, i]

            y1 = max(0, cy - 1)
            y2 = min(h, cy + 2)
            x1 = max(0, cx - 1)
            x2 = min(w, cx + 2)

            total = 0.0
            count = 0
            for py in range(y1, y2):
                for px in range(x1, x2):
                    total += warped[py, px]
                    count += 1

            if count > 0:
                result[j, i] = total / count

    return result

result = calculate_means(warped, centers_x, centers_y, result)
print(f"  Result size: {result.shape[1]} x {result.shape[0]}")

# 6. Save result
print("\n[Step 6] Saving result...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "35_all_local_max_mean.tif"),
                 result_16bit, compression='lzw')
print(f"  Saved: 35_all_local_max_mean.tif")

# 7. Check coverage
print("\n[Step 7] Checking bright pixel coverage...")
covered_mask = np.zeros((h, w), dtype=bool)

for j in range(TARGET_SIZE[1]):
    for i in range(TARGET_SIZE[0]):
        cx, cy = centers_x[j, i], centers_y[j, i]
        y1 = max(0, cy - 1)
        y2 = min(h, cy + 2)
        x1 = max(0, cx - 1)
        x2 = min(w, cx + 2)
        covered_mask[y1:y2, x1:x2] = True

total_bright = np.sum(is_bright_max)
covered_bright = np.sum(is_bright_max & covered_mask)
missed_bright = np.sum(is_bright_max & ~covered_mask)

print(f"  Total bright pixels: {total_bright}")
print(f"  Covered: {covered_bright} ({100*covered_bright/total_bright:.1f}%)")
print(f"  Missed: {missed_bright} ({100*missed_bright/total_bright:.1f}%)")

# 8. Create visualization
print("\n[Step 8] Creating visualization...")

view_size = 12
center_out_y = TARGET_SIZE[1] // 2
center_out_x = TARGET_SIZE[0] // 2

region_centers_x = centers_x[center_out_y:center_out_y+view_size, center_out_x:center_out_x+view_size]
region_centers_y = centers_y[center_out_y:center_out_y+view_size, center_out_x:center_out_x+view_size]

margin = 3
in_x1 = max(0, region_centers_x.min() - margin)
in_x2 = min(w, region_centers_x.max() + margin)
in_y1 = max(0, region_centers_y.min() - margin)
in_y2 = min(h, region_centers_y.max() + margin)

input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[center_out_y:center_out_y+view_size, center_out_x:center_out_x+view_size]

fig, axes = plt.subplots(1, 2, figsize=(28, 14))

# Left: Input with 3x3 boxes
ax1 = axes[0]
ax1.imshow(input_region, cmap='gray', vmin=0, vmax=input_region.max())

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

colors = plt.cm.rainbow(np.linspace(0, 1, view_size * view_size))
for oj in range(view_size):
    for oi in range(view_size):
        cx = centers_x[center_out_y + oj, center_out_x + oi]
        cy = centers_y[center_out_y + oj, center_out_x + oi]

        local_x = cx - in_x1
        local_y = cy - in_y1

        color_idx = oj * view_size + oi
        rect = plt.Rectangle(
            (local_x - 1.5, local_y - 1.5), 3, 3,
            fill=False, edgecolor=colors[color_idx], linewidth=2, alpha=0.9
        )
        ax1.add_patch(rect)

        ax1.plot(local_x, local_y, 'o', markersize=4,
                color=colors[color_idx], markeredgecolor='black', markeredgewidth=0.5)

ax1.set_title(f'Input with 3x3 Boxes (All Local Max Method)\n{view_size}x{view_size} output pixels', fontsize=12)
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

ax2.set_title(f'Output (3x3 MEAN)\n{view_size}x{view_size} pixels', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "35_all_local_max_visualization.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 35_all_local_max_visualization.png")
plt.close()

print("\n" + "=" * 60)
print("All Local Maximum Algorithm Complete!")
print(f"Output: {result.shape[1]} x {result.shape[0]} pixels")
print(f"Coverage: {100*covered_bright/total_bright:.1f}%")
print("=" * 60)
