# -*- coding: utf-8 -*-
"""
Edge Corner Detection with Star Marks
TL, TR, BR, BL 각 코너에 대한 cross point (star mark) 시각화
Star mark 중심 30x30 픽셀 영역 표시
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 작업 디렉토리 설정
work_dir = Path(r'c:\Users\byungpaul\Desktop\AI_Project\20260304_ROI_algorithm')
data_dir = work_dir / 'data'
output_dir = work_dir / 'output'
output_dir.mkdir(exist_ok=True)

# src 모듈 경로 추가
sys.path.insert(0, str(work_dir / 'src'))
from importlib import import_module
roi_detector_module = import_module('2_roi_detector')
AdaptiveROIDetector = roi_detector_module.AdaptiveROIDetector

print("=" * 60)
print("Edge Corner Detection with Star Marks (30x30 pixel view)")
print("=" * 60)

# Step 1: Load image
print("\n[Step 1] Loading image...")
img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
print(f"Image: {img.shape[1]}x{img.shape[0]}, dtype: {img.dtype}")

# Step 2: Detect ROI using crossing method
print("\n[Step 2] Detecting ROI using X/Y crossing method...")
detector = AdaptiveROIDetector(
    initial_threshold=200,
    morph_kernel_size=51,
    corner_region_size=60
)

roi = detector.detect_by_crossing(img, edge_offset=2, search_radius=30)

if roi is None:
    print("ROI detection failed!")
    sys.exit(1)

print(f"ROI detected!")
print(f"  Width: {roi.width:.1f} px")
print(f"  Height: {roi.height:.1f} px")
print(f"  Tilt: {roi.angle:.4f} deg")

# Get cross points (star marks)
cross_points = detector.get_cross_points()

print("\n[Step 3] Corner positions (pixel coordinates):")
corner_names = ['TL', 'TR', 'BR', 'BL']
for name in corner_names:
    corner = roi.corners[corner_names.index(name)]
    cross = cross_points.get(name, corner)
    print(f"  {name}: Cross Point ({cross[0]:.0f}, {cross[1]:.0f}) px, ROI Corner ({corner[0]:.0f}, {corner[1]:.0f}) px")

# Step 4: Create 30x30 pixel view for each corner - all in one image
print("\n[Step 4] Creating 30x30 pixel views for all corners...")

# Create figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

# Colors for each corner (RGB for matplotlib)
corner_colors_rgb = {
    'TL': 'blue',
    'TR': 'green',
    'BR': 'red',
    'BL': 'cyan'
}

half_size = 15  # 30x30 means 15 pixels on each side from center

for i, name in enumerate(corner_names):
    cross = cross_points.get(name, roi.corners[i])
    cx, cy = int(cross[0]), int(cross[1])

    # Extract 30x30 region centered on cross point
    x1 = cx - half_size
    x2 = cx + half_size
    y1 = cy - half_size
    y2 = cy + half_size

    # Handle boundary conditions
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - img.shape[1])
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - img.shape[0])

    x1_clip = max(0, x1)
    x2_clip = min(img.shape[1], x2)
    y1_clip = max(0, y1)
    y2_clip = min(img.shape[0], y2)

    region = img[y1_clip:y2_clip, x1_clip:x2_clip].copy()

    # Pad if necessary
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        region = np.pad(region, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Normalize for display
    region_norm = ((region - region.min()) / (region.max() - region.min() + 1e-6) * 255).astype(np.uint8)

    ax = axes[i]

    # Display image with pixel coordinates (shift by 0.5 to align to pixel centers)
    im = ax.imshow(region_norm, cmap='gray', extent=[x1-0.5, x2-0.5, y2-0.5, y1-0.5])

    # Draw star mark at cross point (center of last edge pixel)
    star_size = 0.4  # pixels (small star mark)
    color = corner_colors_rgb[name]

    # Draw small cross mark (+ shape)
    ax.plot([cx - star_size, cx + star_size], [cy, cy], color=color, linewidth=1.2)  # horizontal
    ax.plot([cx, cx], [cy - star_size, cy + star_size], color=color, linewidth=1.2)  # vertical

    # Mark center point (small dot)
    ax.plot(cx, cy, 'o', color=color, markersize=2)

    # Set title with pixel coordinates
    ax.set_title(f'{name} Corner\nStar Mark: ({cx}, {cy}) px\n30x30 pixel view', fontsize=12, fontweight='bold')

    # Set axis labels with pixel coordinates
    ax.set_xlabel('X (pixels)', fontsize=10)
    ax.set_ylabel('Y (pixels)', fontsize=10)

    # Set ticks to show actual pixel coordinates
    ax.set_xticks(np.arange(x1, x2+1, 5))
    ax.set_yticks(np.arange(y1, y2+1, 5))

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel Value', fontsize=9)

plt.suptitle('Edge Corner Detection - 30x30 Pixel View with Star Marks\n(All 4 corners in one image)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the combined image
output_path = output_dir / '09_edge_corners_30x30_combined.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.show()

print("\n" + "=" * 60)
print("Edge Corner Detection Complete!")
print("=" * 60)
print(f"\nCorner positions (pixel coordinates):")
for name in corner_names:
    cross = cross_points.get(name, roi.corners[corner_names.index(name)])
    print(f"  {name}: Star Mark at ({int(cross[0])}, {int(cross[1])}) px")
print(f"\nOutput file: {output_path}")
