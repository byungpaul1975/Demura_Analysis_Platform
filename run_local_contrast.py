# -*- coding: utf-8 -*-
"""
Local Contrast Algorithm
- Find 3x3 region with HIGH LOCAL CONTRAST (not global brightness)
- Local contrast = center pixel vs surrounding pixels in pitch range
- Track vertically with pitch 3.7~4.0
- Track horizontally with pitch 3.7~4.0 (zigzag)
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
print("Local Contrast Algorithm")
print("Find 3x3 with HIGH LOCAL CONTRAST in pitch range")
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

# 2. Local contrast algorithm
print("\n[Step 2] Running local contrast algorithm...")

@jit(nopython=True, cache=True)
def calc_local_contrast(img, cy, cx, h, w):
    """
    Calculate local contrast for a 3x3 region centered at (cx, cy).
    Contrast = center 3x3 mean - surrounding ring mean
    Higher contrast = brighter center compared to neighbors
    """
    # Center 3x3 mean
    center_sum = 0.0
    center_count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            sy = cy + dy
            sx = cx + dx
            if 0 <= sy < h and 0 <= sx < w:
                center_sum += img[sy, sx]
                center_count += 1

    if center_count == 0:
        return 0.0

    center_mean = center_sum / center_count

    # Surrounding ring (5x5 minus center 3x3)
    ring_sum = 0.0
    ring_count = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            # Skip center 3x3
            if -1 <= dy <= 1 and -1 <= dx <= 1:
                continue
            sy = cy + dy
            sx = cx + dx
            if 0 <= sy < h and 0 <= sx < w:
                ring_sum += img[sy, sx]
                ring_count += 1

    if ring_count == 0:
        return center_mean

    ring_mean = ring_sum / ring_count

    # Contrast = how much brighter is center vs surrounding
    return center_mean - ring_mean

@jit(nopython=True, cache=True)
def find_best_contrast_in_range(img, base_y, base_x, pitch_min, pitch_max, direction, h, w):
    """
    Find the position with best local contrast within pitch range.
    direction: 'h' for horizontal, 'v' for vertical

    Returns: (best_x, best_y, best_contrast)
    """
    best_contrast = -1e9
    best_x = base_x
    best_y = base_y

    # Search within pitch range
    for pitch in range(pitch_min, pitch_max + 1):
        if direction == 0:  # vertical
            expected_y = base_y + pitch
            expected_x = base_x
        else:  # horizontal
            expected_x = base_x + pitch
            expected_y = base_y

        # Search in small region around expected position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                sy = expected_y + dy
                sx = expected_x + dx

                if 0 <= sy < h - 1 and 0 <= sx < w - 1:
                    contrast = calc_local_contrast(img, sy, sx, h, w)
                    if contrast > best_contrast:
                        best_contrast = contrast
                        best_x = sx
                        best_y = sy

    return best_x, best_y, best_contrast

@jit(nopython=True, cache=True)
def local_contrast_tracking(img, target_w, target_h, pitch_x, pitch_y):
    """
    Algorithm:
    1. Vertical tracking: Find positions with best LOCAL CONTRAST at pitch 3-5
    2. Horizontal tracking: For each row, track with best LOCAL CONTRAST at pitch 3-5

    Local contrast = 3x3 center brightness - surrounding ring brightness
    This finds display pixels that stand out from their background.
    """
    h, w = img.shape
    result = np.zeros((target_h, target_w), dtype=np.float64)
    centers_x = np.zeros((target_h, target_w), dtype=np.int32)
    centers_y = np.zeros((target_h, target_w), dtype=np.int32)

    pitch_min = 3
    pitch_max = 5

    # Row 0: Initialize with best contrast in top-left region
    best_contrast = -1e9
    best_x = 2
    best_y = 2
    for sy in range(1, min(6, h - 1)):
        for sx in range(1, min(6, w - 1)):
            contrast = calc_local_contrast(img, sy, sx, h, w)
            if contrast > best_contrast:
                best_contrast = contrast
                best_x = sx
                best_y = sy

    centers_x[0, 0] = best_x
    centers_y[0, 0] = best_y

    # Track row 0 horizontally with local contrast
    for i in range(1, target_w):
        prev_x = centers_x[0, i - 1]
        prev_y = centers_y[0, i - 1]

        # Find best contrast in horizontal pitch range
        bx, by, _ = find_best_contrast_in_range(img, prev_y, prev_x, pitch_min, pitch_max, 1, h, w)
        centers_x[0, i] = bx
        centers_y[0, i] = by

    # Subsequent rows
    for j in range(1, target_h):
        # Find row start: vertical pitch from previous row's first pixel
        prev_x = centers_x[j - 1, 0]
        prev_y = centers_y[j - 1, 0]

        # Find best contrast in vertical pitch range, allowing horizontal zigzag
        best_contrast = -1e9
        best_x = prev_x
        best_y = prev_y + 4

        for pitch in range(pitch_min, pitch_max + 1):
            expected_y = prev_y + pitch
            # Allow wider x search for zigzag offset
            for dy in range(-1, 2):
                for dx in range(-2, 3):
                    sy = expected_y + dy
                    sx = prev_x + dx
                    if 0 <= sy < h - 1 and 0 <= sx < w - 1:
                        contrast = calc_local_contrast(img, sy, sx, h, w)
                        if contrast > best_contrast:
                            best_contrast = contrast
                            best_x = sx
                            best_y = sy

        centers_x[j, 0] = best_x
        centers_y[j, 0] = best_y

        # Track this row horizontally
        for i in range(1, target_w):
            prev_x = centers_x[j, i - 1]
            prev_y = centers_y[j, i - 1]

            # Find best contrast in horizontal pitch range
            bx, by, _ = find_best_contrast_in_range(img, prev_y, prev_x, pitch_min, pitch_max, 1, h, w)
            centers_x[j, i] = bx
            centers_y[j, i] = by

    # Calculate 3x3 MEAN for output
    for j in range(target_h):
        for i in range(target_w):
            cx = centers_x[j, i]
            cy = centers_y[j, i]

            total = 0.0
            count = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    sy = cy + dy
                    sx = cx + dx
                    if 0 <= sy < h and 0 <= sx < w:
                        total += img[sy, sx]
                        count += 1

            if count > 0:
                result[j, i] = total / count

    return result, centers_x, centers_y

import time
start_time = time.time()
result, centers_x, centers_y = local_contrast_tracking(warped, TARGET_SIZE[0], TARGET_SIZE[1], pitch_x, pitch_y)
elapsed = time.time() - start_time
print(f"  Completed in {elapsed:.2f} seconds")
print(f"  Result: {result.shape[1]} x {result.shape[0]}")

# 3. Save
print("\n[Step 3] Saving result...")
result_16bit = np.clip(result, 0, 65535).astype(np.uint16)
tifffile.imwrite(os.path.join(OUTPUT_DIR, "39_local_contrast.tif"), result_16bit, compression='lzw')
print(f"  Saved: 39_local_contrast.tif")

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

# 5. Verify LOCAL contrast at centers
print("\n[Step 5] Checking LOCAL contrast at centers...")

# Sample some centers and check their local contrast
sample_contrasts = []
out_cy = TARGET_SIZE[1] // 2
out_cx = TARGET_SIZE[0] // 2
for oj in range(25):
    for oi in range(25):
        cx = centers_x[out_cy + oj, out_cx + oi]
        cy = centers_y[out_cy + oj, out_cx + oi]

        # Calculate local contrast
        center_val = warped[cy, cx] if 0 <= cy < h and 0 <= cx < w else 0

        # Surrounding mean (3x3 neighbors excluding center)
        surr_sum = 0.0
        surr_count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                sy = cy + dy
                sx = cx + dx
                if 0 <= sy < h and 0 <= sx < w:
                    surr_sum += warped[sy, sx]
                    surr_count += 1

        surr_mean = surr_sum / surr_count if surr_count > 0 else 0
        local_contrast = center_val - surr_mean
        sample_contrasts.append(local_contrast)

sample_contrasts = np.array(sample_contrasts)
print(f"  Local contrast (center - neighbors):")
print(f"    Min: {sample_contrasts.min():.0f}")
print(f"    Max: {sample_contrasts.max():.0f}")
print(f"    Mean: {sample_contrasts.mean():.0f}")
print(f"    % positive (center brighter): {100*np.sum(sample_contrasts > 0)/len(sample_contrasts):.1f}%")

# 6. Visualization: 30x30 input
print("\n[Step 6] Creating 30x30 visualization...")

view_out = 8  # ~30/4 = 7-8 output pixels

# Get center region
region_cx = centers_x[out_cy:out_cy+view_out, out_cx:out_cx+view_out]
region_cy = centers_y[out_cy:out_cy+view_out, out_cx:out_cx+view_out]

# Input bounds ~30x30
in_x1 = region_cx.min() - 2
in_y1 = region_cy.min() - 2
in_x2 = in_x1 + 30
in_y2 = in_y1 + 30

in_x1 = max(0, in_x1)
in_y1 = max(0, in_y1)
in_x2 = min(w, in_x2)
in_y2 = min(h, in_y2)

print(f"  Input: {in_y2-in_y1}x{in_x2-in_x1}")
print(f"  Output: {view_out}x{view_out}")

input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[out_cy:out_cy+view_out, out_cx:out_cx+view_out]

# Colors for boxes
colors = plt.cm.rainbow(np.linspace(0, 1, view_out * view_out))

# Image 1: Input with values and boxes
fig1, ax1 = plt.subplots(figsize=(16, 16))
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
                fontsize=12, color=color, fontweight='bold')

# Draw boxes
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
                fill=False, edgecolor=colors[ci], linewidth=2, alpha=0.9
            )
            ax1.add_patch(rect)

ax1.set_title(f'Input Image (30x30)\nLocal Contrast Method', fontsize=16)
ax1.set_xlabel('X', fontsize=14)
ax1.set_ylabel('Y', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "39_input_30x30.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 39_input_30x30.png")
plt.close()

# Image 2: Output
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.imshow(output_region, cmap='gray', vmin=0, vmax=output_region.max())

for oj in range(view_out):
    for oi in range(view_out):
        val = output_region[oj, oi]
        ci = oj * view_out + oi
        ax2.text(oi, oj, f'{val:.0f}', ha='center', va='center',
                fontsize=20, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=colors[ci], alpha=0.6))

ax2.set_title(f'Output Image (3x3 MEAN)\n{view_out}x{view_out} pixels', fontsize=16)
ax2.set_xlabel('X', fontsize=14)
ax2.set_ylabel('Y', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "39_output_8x8.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 39_output_8x8.png")
plt.close()

print("\n" + "=" * 60)
print("Local Contrast Algorithm Complete!")
print(f"Output: {result.shape[1]} x {result.shape[0]}")
print("=" * 60)
