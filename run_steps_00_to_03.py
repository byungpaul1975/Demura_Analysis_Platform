# -*- coding: utf-8 -*-
"""
ROI Algorithm Step 00~03 Execution
1_utils.py -> 2_roi_detector.py -> 3_image_cropper.py -> 4_perspective_warper.py
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

# Import modules
utils_module = import_module('1_utils')
roi_detector_module = import_module('2_roi_detector')
image_cropper_module = import_module('3_image_cropper')
perspective_warper_module = import_module('4_perspective_warper')

print("=" * 70)
print("ROI Algorithm: Step 00 ~ 03 Execution")
print("=" * 70)

# =============================================================================
# Step 00: Utils - Load Image
# =============================================================================
print("\n" + "=" * 70)
print("[Step 00] 1_utils.py - Image Load")
print("=" * 70)

img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

print(f"Image Path: {img_path}")
print(f"Image Shape: {img.shape[1]} x {img.shape[0]} (W x H)")
print(f"Image Dtype: {img.dtype}")
print(f"Image Range: [{img.min()}, {img.max()}]")

# =============================================================================
# Step 01: ROI Detection
# =============================================================================
print("\n" + "=" * 70)
print("[Step 01] 2_roi_detector.py - ROI Detection")
print("=" * 70)

AdaptiveROIDetector = roi_detector_module.AdaptiveROIDetector
adaptive_detector = AdaptiveROIDetector(
    initial_threshold=200,
    morph_kernel_size=51,
    corner_region_size=60
)
roi = adaptive_detector.detect_by_crossing(img, edge_offset=2, search_radius=30)
cross_points = adaptive_detector.get_cross_points()

print(f"ROI Detection (Crossing Method):")
print(f"  Width: {roi.width:.1f} pixels")
print(f"  Height: {roi.height:.1f} pixels")
print(f"  Tilt Angle: {roi.angle:.4f} degrees")

corner_names = ['TL', 'TR', 'BR', 'BL']
print(f"\nStar Mark Positions (Cross Points):")
for name in corner_names:
    cross = cross_points.get(name)
    print(f"  {name}: ({int(cross[0])}, {int(cross[1])}) px")

# =============================================================================
# Step 02: ROI Crop based on Star Marks
# =============================================================================
print("\n" + "=" * 70)
print("[Step 02] 3_image_cropper.py - ROI Crop (Star Mark based)")
print("=" * 70)

tl = cross_points['TL']
tr = cross_points['TR']
br = cross_points['BR']
bl = cross_points['BL']

roi_x_min = int(min(tl[0], bl[0]))
roi_x_max = int(max(tr[0], br[0]))
roi_y_min = int(min(tl[1], tr[1]))
roi_y_max = int(max(bl[1], br[1]))

print(f"ROI Bounding Box (Star Mark based):")
print(f"  X: {roi_x_min} ~ {roi_x_max}")
print(f"  Y: {roi_y_min} ~ {roi_y_max}")
print(f"  Size: {roi_x_max - roi_x_min} x {roi_y_max - roi_y_min} pixels")

# =============================================================================
# Step 03: Perspective Warp
# =============================================================================
print("\n" + "=" * 70)
print("[Step 03] 4_perspective_warper.py - Perspective Warp")
print("=" * 70)

PerspectiveWarper = perspective_warper_module.PerspectiveWarper

# Source corners from star marks (cross points)
src_corners = np.array([
    cross_points['TL'],
    cross_points['TR'],
    cross_points['BR'],
    cross_points['BL']
], dtype=np.float32)

# Initialize warper with INTER_NEAREST to preserve pixel values
warper = PerspectiveWarper(interpolation=cv2.INTER_NEAREST)

# Option 1: Warp to calculated size (from ROI dimensions)
warp_result = warper.warp(img, src_corners)

print(f"Source Corners (Star Marks):")
for i, name in enumerate(corner_names):
    print(f"  {name}: ({src_corners[i][0]:.1f}, {src_corners[i][1]:.1f})")

print(f"\nDestination Corners:")
for i, name in enumerate(corner_names):
    print(f"  {name}: ({warp_result.dst_corners[i][0]:.1f}, {warp_result.dst_corners[i][1]:.1f})")

print(f"\nWarp Result:")
print(f"  Original Size: {img.shape[1]} x {img.shape[0]} pixels")
print(f"  Warped Size: {warp_result.width} x {warp_result.height} pixels")
print(f"  Interpolation: INTER_NEAREST (preserves pixel values)")

print(f"\nTransform Matrix:")
print(warp_result.transform_matrix)

print(f"\nWarped Image Stats:")
print(f"  Dtype: {warp_result.image.dtype}")
print(f"  Range: [{warp_result.image.min()}, {warp_result.image.max()}]")
print(f"  Mean: {warp_result.image.mean():.2f}")

# Save warped image
warped_output_path = output_dir / '13_warped_image.tif'
cv2.imwrite(str(warped_output_path), warp_result.image)
print(f"\nWarped Image Saved: {warped_output_path}")

# =============================================================================
# Visualization
# =============================================================================
print("\n" + "=" * 70)
print("Creating Visualization...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 8-bit conversion for display
img_8bit = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
warped_8bit = ((warp_result.image - warp_result.image.min()) /
               (warp_result.image.max() - warp_result.image.min()) * 255).astype(np.uint8)

corner_colors = {'TL': 'blue', 'TR': 'green', 'BR': 'red', 'BL': 'cyan'}

# Plot 1: Original Image with Star Marks
ax1 = axes[0, 0]
ax1.imshow(img_8bit, cmap='gray')
for i, name in enumerate(corner_names):
    cross = cross_points[name]
    ax1.plot(cross[0], cross[1], '+', color=corner_colors[name], markersize=15, markeredgewidth=3)
    ax1.plot(cross[0], cross[1], 'o', color=corner_colors[name], markersize=8, label=name)
# Draw ROI polygon
for i in range(4):
    pt1 = src_corners[i]
    pt2 = src_corners[(i+1) % 4]
    ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'yellow', linewidth=2)
ax1.legend(loc='upper right')
ax1.set_title(f'[Step 01] Original + Star Marks\n{img.shape[1]}x{img.shape[0]}', fontsize=12)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Plot 2: Zoomed Star Marks (TL corner)
ax2 = axes[0, 1]
zoom_size = 50
tl_x, tl_y = int(tl[0]), int(tl[1])
region = img_8bit[max(0,tl_y-zoom_size):tl_y+zoom_size, max(0,tl_x-zoom_size):tl_x+zoom_size]
ax2.imshow(region, cmap='gray', extent=[tl_x-zoom_size, tl_x+zoom_size, tl_y+zoom_size, tl_y-zoom_size])
ax2.plot(tl_x, tl_y, '+', color='blue', markersize=20, markeredgewidth=2)
ax2.axhline(y=tl_y, color='blue', linestyle='--', alpha=0.5)
ax2.axvline(x=tl_x, color='blue', linestyle='--', alpha=0.5)
ax2.set_title(f'[Step 02] TL Star Mark Detail\nPosition: ({tl_x}, {tl_y}) px', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

# Plot 3: Warped Image
ax3 = axes[1, 0]
ax3.imshow(warped_8bit, cmap='gray')
# Draw destination corners
for i, name in enumerate(corner_names):
    dst = warp_result.dst_corners[i]
    ax3.plot(dst[0], dst[1], 'o', color=corner_colors[name], markersize=8)
ax3.set_title(f'[Step 03] Warped Image\n{warp_result.width}x{warp_result.height}', fontsize=12)
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')

# Plot 4: Before/After Comparison
ax4 = axes[1, 1]
# Show warped image histogram
ax4.hist(warped_8bit.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7, label='Warped')
ax4.hist(img_8bit.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.5, label='Original')
ax4.set_title('[Step 03] Pixel Value Distribution\n(Original vs Warped)', fontsize=12)
ax4.set_xlabel('Pixel Value')
ax4.set_ylabel('Frequency')
ax4.set_yscale('log')
ax4.legend()

plt.suptitle('ROI Algorithm: Step 00 ~ 03 (Warp) Execution Results', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_path = output_dir / '13_steps_00_to_03_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.show()

print("\n" + "=" * 70)
print("Step 00 ~ 03 Execution Complete!")
print("=" * 70)
print(f"\nOutput Files:")
print(f"  - {warped_output_path}")
print(f"  - {output_path}")
