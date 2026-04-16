# -*- coding: utf-8 -*-
"""
Bright Pixel Detection Approach - Find ALL bright pixels first
"""
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import os
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import AdaptiveROIDetector, PerspectiveWarper, AreaSumResizer

OUTPUT_DIR = "output"
TARGET_SIZE = (2412, 2288)

print("=" * 60)
print("Bright Pixel Detection Approach")
print("Find ALL bright pixels -> Group into 3x3 -> MEAN")
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
warped = warp_result.image.astype(np.float32)
print(f"  Warped size: {warped.shape[1]} x {warped.shape[0]}")

# 2. Find bright pixels
print("\n[Step 2] Finding bright pixels...")

# Use local maximum detection
# A pixel is bright if it's the local maximum in its 3x3 neighborhood
from scipy.ndimage import maximum_filter

# Local maximum detection
local_max = maximum_filter(warped, size=3)
is_local_max = (warped == local_max)

# Also require minimum brightness
threshold = np.percentile(warped, 70)  # Top 30% brightest
is_bright = warped > threshold

# Combine
bright_pixels = is_local_max & is_bright

bright_y, bright_x = np.where(bright_pixels)
print(f"  Found {len(bright_y)} bright pixel candidates")

# 3. Analyze spacing
print("\n[Step 3] Analyzing bright pixel spacing...")

# Look at center region
h, w = warped.shape
center_y, center_x = h // 2, w // 2
region_size = 100

# Get bright pixels in center region
mask = (bright_y >= center_y) & (bright_y < center_y + region_size) & \
       (bright_x >= center_x) & (bright_x < center_x + region_size)

region_bright_y = bright_y[mask]
region_bright_x = bright_x[mask]

print(f"  Bright pixels in center {region_size}x{region_size} region: {len(region_bright_y)}")
print(f"  Expected for target {TARGET_SIZE[0]}x{TARGET_SIZE[1]}: ~{(region_size/warped.shape[0]*TARGET_SIZE[1]) * (region_size/warped.shape[1]*TARGET_SIZE[0]):.0f}")

# 4. Visualize bright pixel detection
print("\n[Step 4] Creating visualization...")

# Smaller region for clear visualization
view_size = 50
view_y1 = center_y
view_y2 = center_y + view_size
view_x1 = center_x
view_x2 = center_x + view_size

view_region = warped[view_y1:view_y2, view_x1:view_x2]
view_bright = bright_pixels[view_y1:view_y2, view_x1:view_x2]

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Left: Original with bright pixels marked
ax1 = axes[0]
ax1.imshow(view_region, cmap='gray', vmin=0, vmax=view_region.max())

# Mark bright pixels
bright_local_y, bright_local_x = np.where(view_bright)
ax1.scatter(bright_local_x, bright_local_y, c='red', s=30, marker='x', linewidths=1)

ax1.set_title(f'Input Image with Bright Pixels (Red X)\nCenter {view_size}x{view_size} region', fontsize=12)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Right: With pixel values
ax2 = axes[1]
ax2.imshow(view_region, cmap='gray', vmin=0, vmax=view_region.max())

for iy in range(view_size):
    for ix in range(view_size):
        val = view_region[iy, ix]
        if view_bright[iy, ix]:
            color = 'red'
            fontweight = 'bold'
        elif val > view_region.max() * 0.7:
            color = 'yellow'
            fontweight = 'bold'
        elif val > view_region.max() * 0.5:
            color = 'white'
            fontweight = 'normal'
        else:
            color = 'gray'
            fontweight = 'normal'
        ax2.text(ix, iy, f'{val:.0f}', ha='center', va='center',
                fontsize=4, color=color, fontweight=fontweight)

ax2.set_title(f'Pixel Values (Red = Local Maximum)\nBright pixels marked with red values', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "33_bright_pixel_detection.png"), dpi=150, bbox_inches='tight')
print(f"  Saved: 33_bright_pixel_detection.png")
plt.close()

# 5. Calculate average spacing between bright pixels
print("\n[Step 5] Calculating bright pixel spacing...")

# Sort by y, then by x for each row
sorted_idx = np.lexsort((region_bright_x, region_bright_y))
sorted_y = region_bright_y[sorted_idx]
sorted_x = region_bright_x[sorted_idx]

# Find horizontal spacing within same row (±1 pixel y tolerance)
h_spacings = []
for i in range(len(sorted_y) - 1):
    if abs(sorted_y[i+1] - sorted_y[i]) <= 1:  # Same row
        dx = sorted_x[i+1] - sorted_x[i]
        if 2 <= dx <= 6:  # Reasonable spacing
            h_spacings.append(dx)

if h_spacings:
    h_spacings = np.array(h_spacings)
    print(f"  Horizontal spacing statistics:")
    print(f"    Min: {h_spacings.min()}")
    print(f"    Max: {h_spacings.max()}")
    print(f"    Mean: {h_spacings.mean():.2f}")
    print(f"    Most common: {np.bincount(h_spacings).argmax()}")

# 6. Estimate actual pitch from warped image size and target
actual_pitch_x = warped.shape[1] / TARGET_SIZE[0]
actual_pitch_y = warped.shape[0] / TARGET_SIZE[1]
print(f"\n  Calculated pitch from image dimensions:")
print(f"    Pitch X: {actual_pitch_x:.3f} pixels")
print(f"    Pitch Y: {actual_pitch_y:.3f} pixels")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
