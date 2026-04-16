# -*- coding: utf-8 -*-
"""
Test X/Y Crossing Method for ROI Detection
Validates detect_by_crossing() method in AdaptiveROIDetector
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from src.roi_detector import AdaptiveROIDetector

# Load image
img = cv2.imread(str(Path('data/G32_cal.tif')), cv2.IMREAD_UNCHANGED)
print(f'Image: {img.shape}, dtype: {img.dtype}')

# Display pixel pitch
DISPLAY_PITCH = 3.8
EDGE_OFFSET = 2

# Create detector
detector = AdaptiveROIDetector(display_pixel_pitch=(DISPLAY_PITCH, DISPLAY_PITCH))

# Detect ROI using X/Y crossing method
print('\n' + '='*70)
print('DETECTING ROI USING X/Y CROSSING METHOD')
print('='*70)

roi = detector.detect_by_crossing(img, edge_offset=EDGE_OFFSET, search_radius=30)

if roi is None:
    print('ERROR: ROI detection failed')
    exit(1)

# Get cross points and edge info
cross_points = detector.get_cross_points()
edge_infos = detector.get_edge_infos()

corner_names = ['TL', 'TR', 'BR', 'BL']

print(f'\n{"Corner":<8} {"ROI Edge (x, y)":<25} {"Cross Point (x, y)":<25} {"Offset"}')
print('-'*80)
for idx, corner_name in enumerate(corner_names):
    ep = roi.corners[idx]
    cp = cross_points.get(corner_name, np.array([0, 0]))
    offset_x = ep[0] - cp[0]
    offset_y = ep[1] - cp[1]
    print(f'{corner_name:<8} ({ep[0]:>8.1f}, {ep[1]:>8.1f})    ({cp[0]:>8.1f}, {cp[1]:>8.1f})    ({offset_x:+.0f}, {offset_y:+.0f})')

# Display pixel info
display_info = detector.get_display_pixel_info(roi)
print(f'\nROI Dimensions:')
print(f'  Camera pixels: {roi.width:.1f} x {roi.height:.1f}')
print(f'  Display pixels: {display_info["display_width_pixels"]} x {display_info["display_height_pixels"]}')
print(f'  Pixel pitch: {DISPLAY_PITCH}')
print(f'  Angle: {roi.angle:.2f} degrees')


# Visualize results
print('\n' + '='*70)
print('VISUALIZING RESULTS')
print('='*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

half = 15

for idx, corner_name in enumerate(corner_names):
    roi_edge = roi.corners[idx]
    cross_point = cross_points[corner_name]

    cx, cy = int(roi_edge[0]), int(roi_edge[1])

    y1 = max(0, cy - half)
    y2 = min(img.shape[0], cy + half)
    x1 = max(0, cx - half)
    x2 = min(img.shape[1], cx + half)

    region = img[y1:y2, x1:x2].copy()

    ax = axes[idx]
    ax.imshow(region, cmap='gray', interpolation='nearest')
    ax.set_title(f'{corner_name} Edge Point\nROI: ({cx}, {cy})', fontsize=14)

    # Draw display pixel grid
    for gx in np.arange(0, 30, DISPLAY_PITCH):
        ax.axvline(x=gx, color='yellow', linestyle='-', linewidth=0.5, alpha=0.5)
    for gy in np.arange(0, 30, DISPLAY_PITCH):
        ax.axhline(y=gy, color='yellow', linestyle='-', linewidth=0.5, alpha=0.5)

    # Calculate relative positions within 30x30 view
    cross_rel_x = cross_point[0] - roi_edge[0] + half
    cross_rel_y = cross_point[1] - roi_edge[1] + half

    # Draw X edge line (vertical line at edge_x)
    ax.axvline(x=cross_rel_x, color='lime', linestyle='-', linewidth=2, alpha=0.8, label='X edge')

    # Draw Y edge line (horizontal line at edge_y)
    ax.axhline(y=cross_rel_y, color='cyan', linestyle='-', linewidth=2, alpha=0.8, label='Y edge')

    # Mark the crossing point with a STAR
    ax.plot(cross_rel_x, cross_rel_y, 'w*', markersize=25, markeredgecolor='black',
            markeredgewidth=1.5, label='X/Y Cross')

    # Mark the ROI edge point (red cross at center)
    ax.axhline(y=half, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=half, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(half, half, 'r+', markersize=20, markeredgewidth=3)

    ax.text(0.02, 0.98,
            f'ROI Edge: ({cx}, {cy})\n'
            f'Cross point: ({cross_point[0]:.0f}, {cross_point[1]:.0f})\n'
            f'Green=X edge, Cyan=Y edge\n'
            f'White star=Cross point',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Camera X (pixels)')
    ax.set_ylabel('Camera Y (pixels)')
    ax.set_xlim(0, 30)
    ax.set_ylim(30, 0)

plt.suptitle('AdaptiveROIDetector.detect_by_crossing() Results\n'
             'White star = Cross point of X & Y edges, Red cross = ROI edge (2px outward)',
             fontsize=14)
plt.tight_layout()
plt.savefig('output/test_crossing_method.png', dpi=200, bbox_inches='tight')
print('Saved: output/test_crossing_method.png')


# Full image with ROI overlay
fig2, ax2 = plt.subplots(1, 1, figsize=(16, 12))

# Normalize image for display
img_display = (img / img.max() * 255).astype(np.uint8)
ax2.imshow(img_display, cmap='gray')

# Draw ROI polygon
roi_polygon = np.vstack([roi.corners, roi.corners[0]])
ax2.plot(roi_polygon[:, 0], roi_polygon[:, 1], 'r-', linewidth=2, label='ROI')

# Draw corner markers
colors = ['cyan', 'magenta', 'yellow', 'lime']
for idx, corner_name in enumerate(corner_names):
    corner = roi.corners[idx]
    cross = cross_points[corner_name]

    # Cross point (star)
    ax2.plot(cross[0], cross[1], '*', color=colors[idx], markersize=15,
             markeredgecolor='white', markeredgewidth=1)

    # ROI edge (plus)
    ax2.plot(corner[0], corner[1], '+', color='red', markersize=12, markeredgewidth=2)

    ax2.text(corner[0]+30, corner[1], corner_name, fontsize=12, color='white',
             fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

ax2.set_title(f'ROI Detection using X/Y Crossing Method\n'
              f'Size: {display_info["display_width_pixels"]} x {display_info["display_height_pixels"]} display pixels')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('output/test_crossing_method_full.png', dpi=150, bbox_inches='tight')
print('Saved: output/test_crossing_method_full.png')

print('\n' + '='*70)
print('TEST COMPLETED SUCCESSFULLY')
print('='*70)
