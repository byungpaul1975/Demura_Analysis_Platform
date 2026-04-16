# -*- coding: utf-8 -*-
"""
Verify No Skipping - Check that all bright pixels are captured
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper, AreaSumResizer
import cv2

OUTPUT_DIR = "output"
TARGET_SIZE = (2412, 2288)
PITCH_RANGE = (3.7, 4.0)

print("Loading and processing...")

# Load and warp image
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
warped = warp_result.image

# Run sequential tracking
resizer = AreaSumResizer(show_progress=True)
result, centers_x, centers_y = resizer.resize_sequential_tracking(
    warped, TARGET_SIZE, pitch_range=PITCH_RANGE
)

print(f"Result: {result.shape[1]} x {result.shape[0]}")

# Verify: Show a small region with ALL boxes and pixel values
# Create visualization showing input with pixel-level detail

# Center region - 20x20 output pixels
view_size = 15
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

# Extract input region
input_region = warped[in_y1:in_y2, in_x1:in_x2].astype(np.float32)
output_region = result[out_y_start:out_y_end, out_x_start:out_x_end]

# Create detailed figure with pixel values
fig, axes = plt.subplots(1, 2, figsize=(28, 14))

# Left: Input with pixel values and boxes
ax1 = axes[0]
ax1.imshow(input_region, cmap='gray', vmin=0, vmax=input_region.max())

# Add pixel values to input image
in_h, in_w = input_region.shape
for iy in range(in_h):
    for ix in range(in_w):
        val = input_region[iy, ix]
        # Color based on brightness
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

        # Convert to local coordinates
        local_x = cx - in_x1
        local_y = cy - in_y1

        # Draw 3x3 box at pixel boundaries
        color_idx = oj * view_size + oi
        rect = plt.Rectangle(
            (local_x - 1.5, local_y - 1.5), 3, 3,
            fill=False, edgecolor=colors[color_idx], linewidth=2, alpha=0.9
        )
        ax1.add_patch(rect)

ax1.set_title(f'Input Image with Pixel Values\n{view_size}x{view_size} output pixels → {in_h}x{in_w} input region', fontsize=12)
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

ax2.set_title(f'Output Image (3x3 MEAN)\n{view_size}x{view_size} pixels, values are MEAN of 9 input pixels', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "30_pixel_values_verification.png"), dpi=150, bbox_inches='tight')
print(f"Saved: 30_pixel_values_verification.png")
plt.close()

# Check for potential missed bright pixels
print("\n=== Verifying No Skipping ===")

# For each row, check if pitch between consecutive centers is within range
skipped_count = 0
total_pairs = 0

for j in range(result.shape[0]):
    for i in range(1, result.shape[1]):
        prev_x = centers_x[j, i-1]
        curr_x = centers_x[j, i]
        pitch = curr_x - prev_x

        total_pairs += 1
        if pitch < 3 or pitch > 5:
            skipped_count += 1

print(f"Total horizontal pixel pairs checked: {total_pairs}")
print(f"Pairs with pitch outside 3-5 range: {skipped_count}")
print(f"Percentage within range: {100*(1 - skipped_count/total_pairs):.2f}%")

# Show distribution
pitches = np.diff(centers_x, axis=1).flatten()
print(f"\nPitch distribution:")
print(f"  Min: {pitches.min()}")
print(f"  Max: {pitches.max()}")
print(f"  Mean: {pitches.mean():.2f}")
print(f"  Std: {pitches.std():.2f}")

# Pitch histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pitches, bins=range(pitches.min(), pitches.max()+2), edgecolor='black', alpha=0.7)
ax.axvline(x=3.7, color='red', linestyle='--', label='Pitch min (3.7)')
ax.axvline(x=4.0, color='red', linestyle='--', label='Pitch max (4.0)')
ax.set_xlabel('Pitch (pixels)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Horizontal Pitch Between Consecutive Centers')
ax.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "30_pitch_distribution.png"), dpi=150, bbox_inches='tight')
print(f"Saved: 30_pitch_distribution.png")
plt.close()

print("\n=== Done ===")
