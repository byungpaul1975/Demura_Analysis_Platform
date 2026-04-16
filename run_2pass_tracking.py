# -*- coding: utf-8 -*-
"""
2-Pass Algorithm: Vertical first, then Horizontal zigzag
1. Find bright pixels along VERTICAL direction (consistent 3.7~4.0 pitch)
2. Then track HORIZONTAL zigzag pattern (also 3.7~4.0 pitch)
3. Extract 3x3 centered on each found position -> MEAN
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os
from numba import jit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper

OUTPUT_DIR = "output"
TARGET_SIZE = (2412, 2288)  # (width, height)

print("=" * 60)
print("2-Pass Algorithm: Vertical + Horizontal Zigzag")
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

pitch_x = w / TARGET_SIZE[0]
pitch_y = h / TARGET_SIZE[1]
print(f"  Pitch: X={pitch_x:.4f}, Y={pitch_y:.4f}")

# 2. Two-pass algorithm
print("\n[Step 2] Running 2-pass algorithm...")

@jit(nopython=True, cache=True)
def two_pass_tracking(img, target_w, target_h, pitch_x, pitch_y):
    """
    2-Pass Algorithm:

    Pass 1 (Vertical): For each column, track bright pixels vertically
                       with pitch 3-5. This gives us row positions.

    Pass 2 (Horizontal): For each row, track bright pixels horizontally
                         with pitch 3-5, starting from Pass 1 positions.
                         Account for zigzag offset between rows.

    This ensures both vertical AND horizontal bright pixels are captured.
    """
    h, w = img.shape
    result = np.zeros((target_h, target_w), dtype=np.float64)
    centers_x = np.zeros((target_h, target_w), dtype=np.int32)
    centers_y = np.zeros((target_h, target_w), dtype=np.int32)

    pitch_min = 3
    pitch_max = 5

    # ============ PASS 1: Vertical tracking for Y positions ============
    # For each output column, track down to find Y positions
    row_y_positions = np.zeros((target_h, target_w), dtype=np.int32)

    for i in range(target_w):
        col_x = int(round(i * pitch_x))
        col_x = max(0, min(w - 1, col_x))

        # First row
        best_val = -1.0
        best_y = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                sy = 0 + dy
                sx = col_x + dx
                if 0 <= sy < h and 0 <= sx < w:
                    val = img[sy, sx]
                    if val > best_val:
                        best_val = val
                        best_y = sy
        row_y_positions[0, i] = best_y

        # Track down
        for j in range(1, target_h):
            prev_y = row_y_positions[j - 1, i]

            best_val = -1.0
            best_y = prev_y + 4

            for pitch in range(pitch_min, pitch_max + 1):
                expected_y = prev_y + pitch
                if expected_y >= h:
                    continue
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        sy = expected_y + dy
                        sx = col_x + dx
                        if 0 <= sy < h and 0 <= sx < w:
                            val = img[sy, sx]
                            if val > best_val:
                                best_val = val
                                best_y = sy

            row_y_positions[j, i] = best_y

    # ============ PASS 2: Horizontal tracking for X positions ============
    # For each row, track horizontally with zigzag consideration

    for j in range(target_h):
        # Get average Y position for this row from Pass 1
        row_y = int(np.mean(row_y_positions[j, :]))

        # First column: search near expected position
        expected_x = int(round(0 * pitch_x))
        best_val = -1.0
        best_x = expected_x
        best_y = row_y

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                sy = row_y + dy
                sx = expected_x + dx
                if 0 <= sy < h and 0 <= sx < w:
                    val = img[sy, sx]
                    if val > best_val:
                        best_val = val
                        best_x = sx
                        best_y = sy

        centers_x[j, 0] = best_x
        centers_y[j, 0] = best_y

        # Track horizontally with zigzag
        for i in range(1, target_w):
            prev_x = centers_x[j, i - 1]
            prev_y = centers_y[j, i - 1]

            # Also consider Y from Pass 1 for this column
            pass1_y = row_y_positions[j, i]

            best_val = -1.0
            best_x = prev_x + 4
            best_y = prev_y

            # Search at pitch 3, 4, 5 from previous X
            for pitch in range(pitch_min, pitch_max + 1):
                expected_x = prev_x + pitch
                if expected_x >= w:
                    continue

                # Search around expected position
                # Allow Y to vary (zigzag) but prefer Pass 1's Y hint
                for dy in range(-2, 3):
                    for dx in range(-1, 2):
                        sy = pass1_y + dy  # Use Pass 1's Y as reference
                        sx = expected_x + dx
                        if 0 <= sy < h and 0 <= sx < w:
                            val = img[sy, sx]
                            if val > best_val:
                                best_val = val
                                best_x = sx
                                best_y = sy

            centers_x[j, i] = best_x
            centers_y[j, i] = best_y

    # ============ Calculate 3x3 MEAN ============
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
result, centers_x, centers_y = two_pass_tracking(warped, TARGET_SIZE[0], TARGET_SIZE[1], pitch_x, pitch_y)
elapsed = time.time() - start_time
print(f"  Completed in {elapsed:.2f} seconds")
print(f"  Result size: {result.shape[1]} x {result.shape[0]}")

# 3. Save result
print("\n[Step 3] Saving result...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "37_2pass_vertical_horizontal.tif"),
                 result_16bit, compression='lzw')
print(f"  Saved: 37_2pass_vertical_horizontal.tif")

# 4. Analyze pitch distribution
print("\n[Step 4] Analyzing pitch distribution...")

# Horizontal pitch
h_pitches = np.diff(centers_x, axis=1).flatten()
print(f"  Horizontal pitch:")
print(f"    Min: {h_pitches.min()}, Max: {h_pitches.max()}, Mean: {h_pitches.mean():.2f}")
for p in range(2, 7):
    pct = 100 * np.sum(h_pitches == p) / len(h_pitches)
    print(f"    Pitch={p}: {pct:.1f}%")

# Vertical pitch
v_pitches = np.diff(centers_y, axis=0).flatten()
print(f"  Vertical pitch:")
print(f"    Min: {v_pitches.min()}, Max: {v_pitches.max()}, Mean: {v_pitches.mean():.2f}")
for p in range(2, 7):
    pct = 100 * np.sum(v_pitches == p) / len(v_pitches)
    print(f"    Pitch={p}: {pct:.1f}%")

# 5. Create visualization - 100x100 input region
print("\n[Step 5] Creating 100x100 visualization...")

# Get center region
out_center_y = TARGET_SIZE[1] // 2
out_center_x = TARGET_SIZE[0] // 2

# Calculate how many output pixels for ~100x100 input
view_out_size = 25  # ~25x25 output = ~100x100 input

out_y_start = out_center_y
out_y_end = out_center_y + view_out_size
out_x_start = out_center_x
out_x_end = out_center_x + view_out_size

# Get centers for this region
region_centers_x = centers_x[out_y_start:out_y_end, out_x_start:out_x_end]
region_centers_y = centers_y[out_y_start:out_y_end, out_x_start:out_x_end]

# Input region bounds (aim for ~100x100)
margin = 2
in_x1 = max(0, region_centers_x.min() - margin)
in_x2 = min(w, region_centers_x.max() + margin + 2)
in_y1 = max(0, region_centers_y.min() - margin)
in_y2 = min(h, region_centers_y.max() + margin + 2)

# Ensure ~100x100
target_in_size = 100
if (in_x2 - in_x1) < target_in_size:
    expand = (target_in_size - (in_x2 - in_x1)) // 2
    in_x1 = max(0, in_x1 - expand)
    in_x2 = min(w, in_x2 + expand)
if (in_y2 - in_y1) < target_in_size:
    expand = (target_in_size - (in_y2 - in_y1)) // 2
    in_y1 = max(0, in_y1 - expand)
    in_y2 = min(h, in_y2 + expand)

print(f"  Input region: {in_y2-in_y1}x{in_x2-in_x1} pixels")
print(f"  Output region: {view_out_size}x{view_out_size} pixels")

input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[out_y_start:out_y_end, out_x_start:out_x_end]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(32, 16))

# Left: Input ~100x100 with pixel values and 3x3 boxes
ax1 = axes[0]
ax1.imshow(input_region, cmap='gray', vmin=0, vmax=input_region.max())

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

# Draw 3x3 boxes
colors = plt.cm.rainbow(np.linspace(0, 1, view_out_size * view_out_size))
for oj in range(view_out_size):
    for oi in range(view_out_size):
        out_j = out_y_start + oj
        out_i = out_x_start + oi

        cx = centers_x[out_j, out_i]
        cy = centers_y[out_j, out_i]

        local_x = cx - in_x1
        local_y = cy - in_y1

        if 0 <= local_x < in_w and 0 <= local_y < in_h:
            color_idx = oj * view_out_size + oi
            rect = plt.Rectangle(
                (local_x - 1.5, local_y - 1.5), 3, 3,
                fill=False, edgecolor=colors[color_idx], linewidth=1.5, alpha=0.9
            )
            ax1.add_patch(rect)

ax1.set_title(f'Input Image (~100x100 center region)\nSize: {in_h}x{in_w} pixels\n2-Pass: Vertical + Horizontal Zigzag', fontsize=14)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Right: Output with pixel values
ax2 = axes[1]
ax2.imshow(output_region, cmap='gray', vmin=0, vmax=output_region.max())

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
plt.savefig(os.path.join(OUTPUT_DIR, "37_2pass_100x100.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 37_2pass_100x100.png")
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
        if 0 <= cy < h and 0 <= cx < w:
            center_vals.append(warped[cy, cx])

center_vals = np.array(center_vals)
threshold_70 = np.percentile(warped, 70)
print(f"  Center brightness (view region):")
print(f"    Min: {center_vals.min():.0f}")
print(f"    Max: {center_vals.max():.0f}")
print(f"    Mean: {center_vals.mean():.0f}")
print(f"    % above 70th percentile ({threshold_70:.0f}): {100*np.sum(center_vals > threshold_70)/len(center_vals):.1f}%")

print("\n" + "=" * 60)
print("2-Pass Algorithm Complete!")
print(f"Output: {result.shape[1]} x {result.shape[0]} pixels")
print("=" * 60)
