# -*- coding: utf-8 -*-
"""
Vertical Tracking Algorithm
- Track bright pixels VERTICALLY (column by column)
- Vertical pitch is more consistent (3.7~4.0)
- For each column, find bright centers going down
- Extract 3x3 centered on each bright pixel -> MEAN
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os
from numba import jit, prange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper

OUTPUT_DIR = "output"
TARGET_SIZE = (2412, 2288)  # (width, height)

print("=" * 60)
print("Vertical Tracking Algorithm")
print("Track bright pixels along COLUMNS (vertical direction)")
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

# 2. Vertical tracking algorithm
print("\n[Step 2] Running vertical tracking algorithm...")

@jit(nopython=True, cache=True)
def find_brightest_in_region(img, cy, cx, search_y, search_x, h, w):
    """Find brightest pixel in search region around (cx, cy)"""
    best_val = -1.0
    best_y = cy
    best_x = cx

    for dy in range(-search_y, search_y + 1):
        for dx in range(-search_x, search_x + 1):
            sy = cy + dy
            sx = cx + dx
            if 0 <= sy < h and 0 <= sx < w:
                val = img[sy, sx]
                if val > best_val:
                    best_val = val
                    best_y = sy
                    best_x = sx

    return best_x, best_y, best_val

@jit(nopython=True, cache=True)
def vertical_tracking_resize(img, target_w, target_h, pitch_x, pitch_y):
    """
    Track bright pixels VERTICALLY along each column.

    Algorithm:
    1. For each column i (0 to target_w-1):
       - Start at top of column (x = i * pitch_x)
       - Track downward with pitch 3.7~4.0
       - At each position, find brightest pixel in small region
    2. Extract 3x3 centered on each bright pixel
    3. Output = MEAN of 3x3
    """
    h, w = img.shape
    result = np.zeros((target_h, target_w), dtype=np.float64)
    centers_x = np.zeros((target_h, target_w), dtype=np.int32)
    centers_y = np.zeros((target_h, target_w), dtype=np.int32)

    pitch_min = 3
    pitch_max = 5

    # Process each column
    for i in range(target_w):
        # Expected x position for this column
        col_x = int(round(i * pitch_x))

        # Track vertically down this column
        # First row: search near top
        expected_y = int(round(0 * pitch_y))
        cx, cy, _ = find_brightest_in_region(img, expected_y, col_x, 2, 2, h, w)
        centers_x[0, i] = cx
        centers_y[0, i] = cy

        # Subsequent rows: search below previous position
        for j in range(1, target_h):
            prev_y = centers_y[j - 1, i]
            prev_x = centers_x[j - 1, i]

            # Search for next bright pixel at pitch 3-5 below
            best_val = -1.0
            best_y = prev_y + 4
            best_x = prev_x

            # Try pitch 3, 4, 5
            for pitch in range(pitch_min, pitch_max + 1):
                expected_y = prev_y + pitch
                if expected_y >= h:
                    continue

                # Search in small region around expected position
                # Allow horizontal movement of +-1 for zigzag
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        sy = expected_y + dy
                        sx = prev_x + dx
                        if 0 <= sy < h and 0 <= sx < w:
                            val = img[sy, sx]
                            if val > best_val:
                                best_val = val
                                best_y = sy
                                best_x = sx

            centers_x[j, i] = best_x
            centers_y[j, i] = best_y

    # Calculate 3x3 MEAN for each center
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
                    total += img[py, px]
                    count += 1

            if count > 0:
                result[j, i] = total / count

    return result, centers_x, centers_y

import time
start_time = time.time()
result, centers_x, centers_y = vertical_tracking_resize(warped, TARGET_SIZE[0], TARGET_SIZE[1], pitch_x, pitch_y)
elapsed = time.time() - start_time
print(f"  Completed in {elapsed:.2f} seconds")
print(f"  Result size: {result.shape[1]} x {result.shape[0]}")

# 3. Save result
print("\n[Step 3] Saving result...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "36_vertical_tracking_mean.tif"),
                 result_16bit, compression='lzw')
print(f"  Saved: 36_vertical_tracking_mean.tif")

# 4. Analyze vertical pitch
print("\n[Step 4] Analyzing vertical pitch...")
v_pitches = []
for i in range(min(100, TARGET_SIZE[0])):
    col_y = centers_y[:, i]
    v_pitches.extend(np.diff(col_y))

v_pitches = np.array(v_pitches)
print(f"  Vertical pitch statistics:")
print(f"    Min: {v_pitches.min()}")
print(f"    Max: {v_pitches.max()}")
print(f"    Mean: {v_pitches.mean():.2f}")

# Count distribution
for p in range(2, 7):
    count = np.sum(v_pitches == p)
    pct = 100 * count / len(v_pitches)
    print(f"    Pitch={p}: {count} ({pct:.1f}%)")

# 5. Create visualization - 100x100 input region and corresponding output
print("\n[Step 5] Creating visualization (100x100 input region)...")

# Center of output
out_center_y = TARGET_SIZE[1] // 2
out_center_x = TARGET_SIZE[0] // 2

# Find how many output pixels fit in 100x100 input region
# 100 / pitch ~ 26 output pixels
view_out_size = 25  # ~25x25 output pixels correspond to ~100x100 input

out_y_start = out_center_y
out_y_end = out_center_y + view_out_size
out_x_start = out_center_x
out_x_end = out_center_x + view_out_size

# Get corresponding input region
region_centers_x = centers_x[out_y_start:out_y_end, out_x_start:out_x_end]
region_centers_y = centers_y[out_y_start:out_y_end, out_x_start:out_x_end]

# Calculate input region bounds (should be ~100x100)
margin = 2
in_x1 = max(0, region_centers_x.min() - margin)
in_x2 = min(w, region_centers_x.max() + margin + 2)
in_y1 = max(0, region_centers_y.min() - margin)
in_y2 = min(h, region_centers_y.max() + margin + 2)

# Adjust to get exactly 100x100 or close
in_size = max(in_x2 - in_x1, in_y2 - in_y1)
if in_size < 100:
    # Expand to 100
    expand = (100 - in_size) // 2
    in_x1 = max(0, in_x1 - expand)
    in_y1 = max(0, in_y1 - expand)
    in_x2 = min(w, in_x1 + 100)
    in_y2 = min(h, in_y1 + 100)

print(f"  Input region: [{in_y1}:{in_y2}, {in_x1}:{in_x2}] = {in_y2-in_y1}x{in_x2-in_x1} pixels")
print(f"  Output region: [{out_y_start}:{out_y_end}, {out_x_start}:{out_x_end}] = {view_out_size}x{view_out_size} pixels")

input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[out_y_start:out_y_end, out_x_start:out_x_end]

# Create large figure
fig, axes = plt.subplots(1, 2, figsize=(32, 16))

# Left: Input 100x100 with pixel values and 3x3 boxes
ax1 = axes[0]
ax1.imshow(input_region, cmap='gray', vmin=0, vmax=input_region.max())

# Add pixel values to input
in_h, in_w = input_region.shape
for iy in range(in_h):
    for ix in range(in_w):
        val = input_region[iy, ix]
        if val > input_region.max() * 0.75:
            color = 'yellow'
        elif val > input_region.max() * 0.5:
            color = 'white'
        else:
            color = 'gray'
        ax1.text(ix, iy, f'{val:.0f}', ha='center', va='center',
                fontsize=3, color=color, fontweight='bold')

# Draw 3x3 boxes for all output pixels in view
colors = plt.cm.rainbow(np.linspace(0, 1, view_out_size * view_out_size))
for oj in range(view_out_size):
    for oi in range(view_out_size):
        out_j = out_y_start + oj
        out_i = out_x_start + oi

        cx = centers_x[out_j, out_i]
        cy = centers_y[out_j, out_i]

        # Check if center is within our input region
        local_x = cx - in_x1
        local_y = cy - in_y1

        if 0 <= local_x < in_w and 0 <= local_y < in_h:
            color_idx = oj * view_out_size + oi
            rect = plt.Rectangle(
                (local_x - 1.5, local_y - 1.5), 3, 3,
                fill=False, edgecolor=colors[color_idx], linewidth=1.5, alpha=0.9
            )
            ax1.add_patch(rect)

ax1.set_title(f'Input Image (Center ~100x100 region)\nSize: {in_h}x{in_w} pixels', fontsize=14)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Right: Output with pixel values
ax2 = axes[1]
ax2.imshow(output_region, cmap='gray', vmin=0, vmax=output_region.max())

# Add pixel values to output
for oj in range(view_out_size):
    for oi in range(view_out_size):
        val = output_region[oj, oi]
        color_idx = oj * view_out_size + oi
        ax2.text(oi, oj, f'{val:.0f}', ha='center', va='center',
                fontsize=5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.05', facecolor=colors[color_idx], alpha=0.5))

ax2.set_title(f'Output Image (3x3 MEAN)\nSize: {view_out_size}x{view_out_size} pixels', fontsize=14)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "36_vertical_tracking_100x100.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 36_vertical_tracking_100x100.png")
plt.close()

# 6. Check center brightness
print("\n[Step 6] Checking center pixel brightness...")
center_vals = []
for oj in range(view_out_size):
    for oi in range(view_out_size):
        out_j = out_y_start + oj
        out_i = out_x_start + oi
        cx = centers_x[out_j, out_i]
        cy = centers_y[out_j, out_i]
        center_vals.append(warped[cy, cx])

center_vals = np.array(center_vals)
print(f"  Center pixel brightness in view region:")
print(f"    Min: {center_vals.min():.0f}")
print(f"    Max: {center_vals.max():.0f}")
print(f"    Mean: {center_vals.mean():.0f}")
print(f"    % above 70th percentile: {100*np.sum(center_vals > np.percentile(warped, 70))/len(center_vals):.1f}%")

print("\n" + "=" * 60)
print("Vertical Tracking Algorithm Complete!")
print(f"Output: {result.shape[1]} x {result.shape[0]} pixels")
print("=" * 60)
