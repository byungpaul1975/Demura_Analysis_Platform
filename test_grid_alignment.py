# -*- coding: utf-8 -*-
"""
Test Adaptive ROI with Display Pixel Grid Alignment
Camera pixels: 3x3 to 4x4 camera pixels = 1 display pixel
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.roi_detector import AdaptiveROIDetector

# Load image
img = cv2.imread(str(Path('data/G32_cal.tif')), cv2.IMREAD_UNCHANGED)
print(f'Image: {img.shape}, dtype: {img.dtype}')
print('='*70)

# Test with different display pixel pitch values
pitch_values = [(3.0, 3.0), (3.5, 3.5), (4.0, 4.0)]

results = {}

for pitch in pitch_values:
    print(f'\nDisplay Pixel Pitch: {pitch[0]}x{pitch[1]} camera px/display px')
    print('-'*50)

    detector = AdaptiveROIDetector(
        initial_threshold=200,
        corner_region_size=60,
        display_pixel_pitch=pitch,
        align_to_display_grid=True
    )

    # Detect with grid alignment
    roi = detector.detect_with_grid_alignment(img)

    if roi:
        info = detector.get_display_pixel_info(roi)
        results[pitch] = {'roi': roi, 'info': info, 'detector': detector}

        print(f'  Display Resolution: {info["display_width_pixels"]} x {info["display_height_pixels"]}')
        print(f'  Camera ROI: {info["camera_width_pixels"]:.1f} x {info["camera_height_pixels"]:.1f} px')
        print(f'  Total Display Pixels: {info["total_display_pixels"]:,}')
        print(f'  Corners (TL, TR, BR, BL):')
        for name, corner in zip(['TL', 'TR', 'BR', 'BL'], roi.corners):
            print(f'    {name}: ({corner[0]:.2f}, {corner[1]:.2f})')

# Compare results
print('\n' + '='*70)
print('COMPARISON: Display Pixel Grid Alignment')
print('='*70)

print(f'\n{"Pitch":<12} {"Display W":<12} {"Display H":<12} {"Camera W":<12} {"Camera H":<12}')
print('-'*60)
for pitch, data in results.items():
    info = data['info']
    print(f'{pitch[0]}x{pitch[1]:<8} {info["display_width_pixels"]:<12} {info["display_height_pixels"]:<12} '
          f'{info["camera_width_pixels"]:<12.1f} {info["camera_height_pixels"]:<12.1f}')

# Visualize edge regions with display pixel grid overlay
print('\n' + '='*70)
print('EDGE VISUALIZATION WITH DISPLAY PIXEL GRID')
print('='*70)

# Use 3.5 pitch as reference
pitch = (3.5, 3.5)
detector = results[pitch]['detector']
roi = results[pitch]['roi']
corners = roi.corners

# Calculate edge centers
edge_centers = {
    'Top': ((corners[0] + corners[1]) / 2).astype(int),
    'Right': ((corners[1] + corners[2]) / 2).astype(int),
    'Bottom': ((corners[2] + corners[3]) / 2).astype(int),
    'Left': ((corners[3] + corners[0]) / 2).astype(int)
}

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

half = 15  # 30x30 region

for idx, (name, center) in enumerate(edge_centers.items()):
    cx, cy = center[0], center[1]

    y1, y2 = max(0, cy - half), min(img.shape[0], cy + half)
    x1, x2 = max(0, cx - half), min(img.shape[1], cx + half)
    region = img[y1:y2, x1:x2].copy()

    ax = axes[idx]
    ax.imshow(region, cmap='gray', interpolation='nearest')
    ax.set_title(f'{name} Edge - 30x30 pixels\nCenter: ({cx}, {cy})', fontsize=12)

    # Overlay display pixel grid
    pitch_x, pitch_y = pitch

    # Calculate grid offset based on edge center position
    x_offset = cx % pitch_x
    y_offset = cy % pitch_y

    # Draw grid lines (aligned to display pixel boundaries)
    start_x = -x_offset
    start_y = -y_offset

    for i in np.arange(start_x + half, region.shape[1], pitch_x):
        if 0 <= i < region.shape[1]:
            ax.axvline(x=i, color='yellow', linestyle='-', linewidth=1, alpha=0.8)
    for i in np.arange(start_y + half, region.shape[0], pitch_y):
        if 0 <= i < region.shape[0]:
            ax.axhline(y=i, color='yellow', linestyle='-', linewidth=1, alpha=0.8)

    # Draw negative direction
    for i in np.arange(start_x + half, 0, -pitch_x):
        if 0 <= i < region.shape[1]:
            ax.axvline(x=i, color='yellow', linestyle='-', linewidth=1, alpha=0.8)
    for i in np.arange(start_y + half, 0, -pitch_y):
        if 0 <= i < region.shape[0]:
            ax.axhline(y=i, color='yellow', linestyle='-', linewidth=1, alpha=0.8)

    # Mark center
    ax.axhline(y=half, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=half, color='red', linestyle='--', linewidth=1.5)
    ax.plot(half, half, 'r+', markersize=15, markeredgewidth=2)

    ax.set_xlabel('Camera X (pixels)')
    ax.set_ylabel('Camera Y (pixels)')

    # Add grid info
    ax.text(0.02, 0.98, f'Grid: {pitch_x}x{pitch_y} cam px/disp px',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(f'Edge Regions with Display Pixel Grid (pitch={pitch[0]}x{pitch[1]})\n'
             f'Yellow lines = Display pixel boundaries', fontsize=14)
plt.tight_layout()
plt.savefig('output/edge_30x30_display_grid.png', dpi=200, bbox_inches='tight')
print('Saved: output/edge_30x30_display_grid.png')

# Show display pixel info
print('\n' + '='*70)
print('FINAL DISPLAY PIXEL INFO (pitch=3.5x3.5)')
print('='*70)
info = results[(3.5, 3.5)]['info']
print(f'  Display Resolution: {info["display_width_pixels"]} x {info["display_height_pixels"]} pixels')
print(f'  Camera ROI Size: {info["camera_width_pixels"]:.1f} x {info["camera_height_pixels"]:.1f} px')
print(f'  Horizontal Pitch: {info["horizontal_pitch"]} camera px/display px')
print(f'  Vertical Pitch: {info["vertical_pitch"]} camera px/display px')
print(f'  Total Display Pixels: {info["total_display_pixels"]:,}')
