# -*- coding: utf-8 -*-
"""Test adaptive ROI detector and compare with global threshold"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.roi_detector import ROIDetector, AdaptiveROIDetector

# Load image
img = cv2.imread(str(Path('data/G32_cal.tif')), cv2.IMREAD_UNCHANGED)
print(f'Image: {img.shape}, dtype: {img.dtype}')
print('='*70)

# Test 1: Global threshold detection
print('GLOBAL THRESHOLD DETECTION (threshold=460)')
print('-'*70)
global_detector = ROIDetector(threshold=460)
global_roi = global_detector.detect(img)

if global_roi:
    print('Corners:')
    for name, corner in zip(['TL', 'TR', 'BR', 'BL'], global_roi.corners):
        print(f'  {name}: ({corner[0]:.2f}, {corner[1]:.2f})')
    print(f'Area: {global_roi.area:.0f}')
    print(f'Width: {global_roi.width:.2f}, Height: {global_roi.height:.2f}')
    print(f'Angle: {global_roi.angle:.4f} deg')

print()
print('='*70)

# Test 2: Adaptive per-corner detection
print('ADAPTIVE PER-CORNER THRESHOLD DETECTION')
print('-'*70)
adaptive_detector = AdaptiveROIDetector(
    initial_threshold=200,
    corner_region_size=60,
    threshold_range=(50, 600),
    threshold_step=20
)
adaptive_roi = adaptive_detector.detect_adaptive(img)

if adaptive_roi:
    print('Corners (refined):')
    for name, corner in zip(['TL', 'TR', 'BR', 'BL'], adaptive_roi.corners):
        print(f'  {name}: ({corner[0]:.2f}, {corner[1]:.2f})')
    print(f'Area: {adaptive_roi.area:.0f}')
    print(f'Width: {adaptive_roi.width:.2f}, Height: {adaptive_roi.height:.2f}')
    print(f'Angle: {adaptive_roi.angle:.4f} deg')

    print()
    print('Per-corner thresholds:')
    thresholds = adaptive_detector.get_corner_thresholds()
    for name, thresh in thresholds.items():
        print(f'  {name}: {thresh}')
    print(f'  Average: {adaptive_detector.get_average_threshold():.0f}')

    print()
    print('Corner Metrics:')
    metrics = adaptive_detector.get_corner_metrics()
    for name, m in metrics.items():
        print(f'  {name}: thresh={m.optimal_threshold}, grad_max={m.gradient_max:.1f}, '
              f'sharpness={m.edge_sharpness:.1f}, contrast={m.edge_contrast:.1f}, '
              f'confidence={m.confidence:.3f}')

# Compare corners
print()
print('='*70)
print('CORNER POSITION COMPARISON')
print('-'*70)
print(f'{"Corner":<8} {"Global (x,y)":<20} {"Adaptive (x,y)":<20} {"Diff (px)"}')
print('-'*70)
for i, name in enumerate(['TL', 'TR', 'BR', 'BL']):
    g = global_roi.corners[i]
    a = adaptive_roi.corners[i]
    diff = np.linalg.norm(a - g)
    print(f'{name:<8} ({g[0]:.1f}, {g[1]:.1f}){" "*6}({a[0]:.1f}, {a[1]:.1f}){" "*6}{diff:.2f}')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Global detection
ax1 = axes[0]
img_vis1 = (img / img.max() * 255).astype(np.uint8)
img_vis1 = cv2.cvtColor(img_vis1, cv2.COLOR_GRAY2RGB)
corners1 = global_roi.corners.astype(np.int32)
cv2.polylines(img_vis1, [corners1], True, (0, 255, 0), 3)
for i, (name, corner) in enumerate(zip(['TL', 'TR', 'BR', 'BL'], corners1)):
    cv2.circle(img_vis1, tuple(corner), 30, (255, 0, 0), -1)
ax1.imshow(img_vis1)
ax1.set_title(f'Global Threshold (460)\nArea: {global_roi.area:.0f}')

# Adaptive detection
ax2 = axes[1]
img_vis2 = (img / img.max() * 255).astype(np.uint8)
img_vis2 = cv2.cvtColor(img_vis2, cv2.COLOR_GRAY2RGB)
corners2 = adaptive_roi.corners.astype(np.int32)
cv2.polylines(img_vis2, [corners2], True, (0, 255, 255), 3)
for i, (name, corner) in enumerate(zip(['TL', 'TR', 'BR', 'BL'], corners2)):
    thresh = thresholds[name]
    cv2.circle(img_vis2, tuple(corner), 30, (255, 0, 255), -1)
ax2.imshow(img_vis2)
ax2.set_title(f'Adaptive Per-Corner\nTL={thresholds["TL"]}, TR={thresholds["TR"]}, BR={thresholds["BR"]}, BL={thresholds["BL"]}')

plt.tight_layout()
plt.savefig('output/adaptive_vs_global_comparison.png', dpi=150, bbox_inches='tight')
print()
print('Saved: output/adaptive_vs_global_comparison.png')

# Detail view of corners
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))

half = 60
for idx, name in enumerate(['TL', 'TR', 'BR', 'BL']):
    # Global corner
    g = global_roi.corners[idx].astype(int)
    y1, y2 = max(0, g[1]-half), min(img.shape[0], g[1]+half)
    x1, x2 = max(0, g[0]-half), min(img.shape[1], g[0]+half)
    region_g = img[y1:y2, x1:x2]

    ax_g = axes2[0, idx]
    ax_g.imshow(region_g, cmap='gray')
    ax_g.axhline(y=half, color='r', linestyle='--', linewidth=1)
    ax_g.axvline(x=half, color='r', linestyle='--', linewidth=1)
    ax_g.set_title(f'Global {name}\n({g[0]}, {g[1]})')

    # Adaptive corner
    a = adaptive_roi.corners[idx].astype(int)
    y1, y2 = max(0, a[1]-half), min(img.shape[0], a[1]+half)
    x1, x2 = max(0, a[0]-half), min(img.shape[1], a[0]+half)
    region_a = img[y1:y2, x1:x2]

    ax_a = axes2[1, idx]
    ax_a.imshow(region_a, cmap='gray')
    ax_a.axhline(y=half, color='cyan', linestyle='--', linewidth=1)
    ax_a.axvline(x=half, color='cyan', linestyle='--', linewidth=1)
    thresh = thresholds[name]
    ax_a.set_title(f'Adaptive {name} (thresh={thresh})\n({a[0]}, {a[1]})')

plt.suptitle('Corner Comparison: Global vs Adaptive', fontsize=14)
plt.tight_layout()
plt.savefig('output/adaptive_corner_details.png', dpi=150, bbox_inches='tight')
print('Saved: output/adaptive_corner_details.png')
