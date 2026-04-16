# -*- coding: utf-8 -*-
"""
Row-by-Row Algorithm with Horizontal Zigzag
1. Track rows vertically (pitch 3-5)
2. Within each row, track horizontally with zigzag (pitch 3-5)
3. Each 3x3 -> 1 pixel MEAN
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
print("Row-by-Row Algorithm with Horizontal Zigzag")
print("=" * 60)

# 1. Load and warp
print("\n[Step 1] Loading image...")
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
print(f"  Warped: {w} x {h}")

pitch_x = w / TARGET_SIZE[0]
pitch_y = h / TARGET_SIZE[1]
print(f"  Pitch: X={pitch_x:.4f}, Y={pitch_y:.4f}")

# 2. Row-by-row algorithm
print("\n[Step 2] Running row-by-row algorithm...")

@jit(nopython=True, cache=True)
def row_by_row_zigzag(img, target_w, target_h, pitch_x, pitch_y):
    """
    Algorithm:
    1. For row 0: Find first pixel at (0,0) region, then track horizontally
    2. For row j: Start position is based on row j-1's first pixel + vertical pitch
       Then track horizontally with zigzag pattern

    Key insight: Each row's starting X position may be offset (zigzag)
    from the previous row by +-1 pixel.
    """
    h, w = img.shape
    result = np.zeros((target_h, target_w), dtype=np.float64)
    centers_x = np.zeros((target_h, target_w), dtype=np.int32)
    centers_y = np.zeros((target_h, target_w), dtype=np.int32)

    pitch_min = 3
    pitch_max = 5

    # Row 0: Initialize
    # Find first bright pixel near (0, 0)
    best_val = -1.0
    best_x = 0
    best_y = 0
    for sy in range(min(5, h)):
        for sx in range(min(5, w)):
            val = img[sy, sx]
            if val > best_val:
                best_val = val
                best_x = sx
                best_y = sy

    centers_x[0, 0] = best_x
    centers_y[0, 0] = best_y

    # Track row 0 horizontally
    for i in range(1, target_w):
        prev_x = centers_x[0, i - 1]
        prev_y = centers_y[0, i - 1]

        best_val = -1.0
        best_x = prev_x + 4
        best_y = prev_y

        for pitch in range(pitch_min, pitch_max + 1):
            expected_x = prev_x + pitch
            if expected_x >= w:
                continue
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    sy = prev_y + dy
                    sx = expected_x + dx
                    if 0 <= sy < h and 0 <= sx < w:
                        val = img[sy, sx]
                        if val > best_val:
                            best_val = val
                            best_x = sx
                            best_y = sy

        centers_x[0, i] = best_x
        centers_y[0, i] = best_y

    # Subsequent rows: start from previous row's positions + vertical pitch
    for j in range(1, target_h):
        # Previous row's first pixel
        prev_row_first_x = centers_x[j - 1, 0]
        prev_row_first_y = centers_y[j - 1, 0]

        # Find first pixel of this row: vertical pitch from previous row
        best_val = -1.0
        best_x = prev_row_first_x
        best_y = prev_row_first_y + 4

        for pitch in range(pitch_min, pitch_max + 1):
            expected_y = prev_row_first_y + pitch
            if expected_y >= h:
                continue
            # Allow horizontal zigzag offset
            for dy in range(-1, 2):
                for dx in range(-2, 3):  # Wider x search for zigzag
                    sy = expected_y + dy
                    sx = prev_row_first_x + dx
                    if 0 <= sy < h and 0 <= sx < w:
                        val = img[sy, sx]
                        if val > best_val:
                            best_val = val
                            best_x = sx
                            best_y = sy

        centers_x[j, 0] = best_x
        centers_y[j, 0] = best_y

        # Track this row horizontally
        for i in range(1, target_w):
            prev_x = centers_x[j, i - 1]
            prev_y = centers_y[j, i - 1]

            best_val = -1.0
            best_x = prev_x + 4
            best_y = prev_y

            for pitch in range(pitch_min, pitch_max + 1):
                expected_x = prev_x + pitch
                if expected_x >= w:
                    continue
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        sy = prev_y + dy
                        sx = expected_x + dx
                        if 0 <= sy < h and 0 <= sx < w:
                            val = img[sy, sx]
                            if val > best_val:
                                best_val = val
                                best_x = sx
                                best_y = sy

            centers_x[j, i] = best_x
            centers_y[j, i] = best_y

    # Calculate 3x3 MEAN
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
result, centers_x, centers_y = row_by_row_zigzag(warped, TARGET_SIZE[0], TARGET_SIZE[1], pitch_x, pitch_y)
elapsed = time.time() - start_time
print(f"  Completed in {elapsed:.2f} seconds")
print(f"  Result: {result.shape[1]} x {result.shape[0]}")

# 3. Save
print("\n[Step 3] Saving result...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "38_row_zigzag.tif"), result_16bit, compression='lzw')
print(f"  Saved: 38_row_zigzag.tif")

# 4. Pitch analysis
print("\n[Step 4] Pitch analysis...")
h_pitch = np.diff(centers_x, axis=1).flatten()
v_pitch = np.diff(centers_y, axis=0).flatten()

print(f"  Horizontal: Min={h_pitch.min()}, Max={h_pitch.max()}, Mean={h_pitch.mean():.2f}")
for p in [3, 4, 5]:
    print(f"    Pitch={p}: {100*np.sum(h_pitch==p)/len(h_pitch):.1f}%")

print(f"  Vertical: Min={v_pitch.min()}, Max={v_pitch.max()}, Mean={v_pitch.mean():.2f}")
for p in [3, 4, 5]:
    print(f"    Pitch={p}: {100*np.sum(v_pitch==p)/len(v_pitch):.1f}%")

# 5. Visualization: 100x100 input
print("\n[Step 5] Creating 100x100 visualization...")

out_cy = TARGET_SIZE[1] // 2
out_cx = TARGET_SIZE[0] // 2
view_out = 25  # 25x25 output -> ~100x100 input

# Get center region
region_cx = centers_x[out_cy:out_cy+view_out, out_cx:out_cx+view_out]
region_cy = centers_y[out_cy:out_cy+view_out, out_cx:out_cx+view_out]

# Calculate input bounds to be ~100x100
in_x1 = region_cx.min() - 2
in_y1 = region_cy.min() - 2
in_x2 = in_x1 + 100
in_y2 = in_y1 + 100

# Clamp
in_x1 = max(0, in_x1)
in_y1 = max(0, in_y1)
in_x2 = min(w, in_x2)
in_y2 = min(h, in_y2)

print(f"  Input: [{in_y1}:{in_y2}, {in_x1}:{in_x2}] = {in_y2-in_y1}x{in_x2-in_x1}")
print(f"  Output: {view_out}x{view_out}")

input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[out_cy:out_cy+view_out, out_cx:out_cx+view_out]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(32, 16))

# Left: Input with pixel values and boxes
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
colors = plt.cm.rainbow(np.linspace(0, 1, view_out * view_out))
for oj in range(view_out):
    for oi in range(view_out):
        cx = centers_x[out_cy + oj, out_cx + oi]
        cy = centers_y[out_cy + oj, out_cx + oi]

        lx = cx - in_x1
        ly = cy - in_y1

        if 0 <= lx < in_w and 0 <= ly < in_h:
            ci = oj * view_out + oi
            rect = plt.Rectangle(
                (lx - 1.5, ly - 1.5), 3, 3,
                fill=False, edgecolor=colors[ci], linewidth=1.5, alpha=0.9
            )
            ax1.add_patch(rect)

ax1.set_title(f'Input Image (100x100 center)\nSize: {in_h}x{in_w} pixels', fontsize=14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Right: Output with values
ax2 = axes[1]
ax2.imshow(output_region, cmap='gray', vmin=0, vmax=output_region.max())

for oj in range(view_out):
    for oi in range(view_out):
        val = output_region[oj, oi]
        ci = oj * view_out + oi
        ax2.text(oi, oj, f'{val:.0f}', ha='center', va='center',
                fontsize=5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.05', facecolor=colors[ci], alpha=0.5))

ax2.set_title(f'Output Image (3x3 MEAN)\nSize: {view_out}x{view_out} pixels', fontsize=14)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "38_row_zigzag_100x100.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 38_row_zigzag_100x100.png")
plt.close()

# 6. Center brightness check
print("\n[Step 6] Center brightness...")
cvals = []
for oj in range(view_out):
    for oi in range(view_out):
        cx = centers_x[out_cy + oj, out_cx + oi]
        cy = centers_y[out_cy + oj, out_cx + oi]
        if 0 <= cy < h and 0 <= cx < w:
            cvals.append(warped[cy, cx])

cvals = np.array(cvals)
th70 = np.percentile(warped, 70)
print(f"  Min: {cvals.min():.0f}, Max: {cvals.max():.0f}, Mean: {cvals.mean():.0f}")
print(f"  % above 70th percentile: {100*np.sum(cvals > th70)/len(cvals):.1f}%")

print("\n" + "=" * 60)
print("Row-by-Row Zigzag Complete!")
print(f"Output: {result.shape[1]} x {result.shape[0]}")
print("=" * 60)
