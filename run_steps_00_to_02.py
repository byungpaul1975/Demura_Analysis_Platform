# -*- coding: utf-8 -*-
"""
ROI Algorithm Step 00~02 Execution
1_utils.py -> 2_roi_detector.py -> 3_image_cropper.py
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

print("=" * 70)
print("ROI Algorithm: Step 00 ~ 02 Execution")
print("=" * 70)

# =============================================================================
# Step 00: Utils - Load Image
# =============================================================================
print("\n" + "=" * 70)
print("[Step 00] 1_utils.py - Image Load & Basic Functions")
print("=" * 70)

img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

print(f"Image Path: {img_path}")
print(f"Image Shape: {img.shape} (H x W)")
print(f"Image Dtype: {img.dtype}")
print(f"Image Range: [{img.min()}, {img.max()}]")
print(f"Image Mean: {img.mean():.2f}")

# Normalize using utils
normalized = utils_module.normalize_image(img.astype(np.float32))
print(f"\nNormalized Range: [{normalized.min():.4f}, {normalized.max():.4f}]")

# =============================================================================
# Step 01: ROI Detection
# =============================================================================
print("\n" + "=" * 70)
print("[Step 01] 2_roi_detector.py - ROI Detection")
print("=" * 70)

# Basic ROI Detection
ROIDetector = roi_detector_module.ROIDetector
detector = ROIDetector(threshold=460, morph_kernel_size=51)
roi = detector.detect(img)

print(f"ROI Detection Result:")
print(f"  Area: {roi.area:.0f} pixels")
print(f"  Width: {roi.width:.1f} pixels")
print(f"  Height: {roi.height:.1f} pixels")
print(f"  Tilt Angle: {roi.angle:.4f} degrees")
print(f"\nCorner Positions:")
corner_names = ['TL', 'TR', 'BR', 'BL']
for i, name in enumerate(corner_names):
    print(f"  {name}: ({roi.corners[i][0]:.1f}, {roi.corners[i][1]:.1f})")

# Adaptive ROI Detection with crossing method
AdaptiveROIDetector = roi_detector_module.AdaptiveROIDetector
adaptive_detector = AdaptiveROIDetector(
    initial_threshold=200,
    morph_kernel_size=51,
    corner_region_size=60
)
roi_crossing = adaptive_detector.detect_by_crossing(img, edge_offset=2, search_radius=30)
cross_points = adaptive_detector.get_cross_points()

print(f"\nAdaptive Detection (Crossing Method):")
print(f"  Width: {roi_crossing.width:.1f} pixels")
print(f"  Height: {roi_crossing.height:.1f} pixels")
print(f"  Tilt Angle: {roi_crossing.angle:.4f} degrees")
print(f"\nCross Points (Star Marks):")
for name in corner_names:
    cross = cross_points.get(name)
    print(f"  {name}: ({int(cross[0])}, {int(cross[1])}) px")

# =============================================================================
# Step 02: Image Cropping
# =============================================================================
print("\n" + "=" * 70)
print("[Step 02] 3_image_cropper.py - Image Cropping")
print("=" * 70)

ImageCropper = image_cropper_module.ImageCropper
CropRegion = image_cropper_module.CropRegion

# Create crop region based on detected ROI (with margin)
margin = 50
x_min = int(min(roi.corners[:, 0])) - margin
x_max = int(max(roi.corners[:, 0])) + margin
y_min = int(min(roi.corners[:, 1])) - margin
y_max = int(max(roi.corners[:, 1])) + margin

cropper = ImageCropper()
cropper.set_region(x_min, y_min, x_max, y_max)

crop_info = cropper.get_info()
print(f"Crop Region:")
print(f"  X: {crop_info['x_start']} ~ {crop_info['x_end']}")
print(f"  Y: {crop_info['y_start']} ~ {crop_info['y_end']}")
print(f"  Size: {crop_info['width']} x {crop_info['height']} pixels")

# Crop the image
cropped_img = cropper.crop(img)
print(f"\nOriginal Image: {img.shape[1]} x {img.shape[0]} pixels")
print(f"Cropped Image: {cropped_img.shape[1]} x {cropped_img.shape[0]} pixels")

# Save cropped image
cropped_output_path = output_dir / '11_cropped_image.tif'
cv2.imwrite(str(cropped_output_path), cropped_img)
print(f"Cropped Image Saved: {cropped_output_path}")

# =============================================================================
# ROI Crop based on Star Marks (Cross Points)
# =============================================================================
print("\n" + "=" * 70)
print("[Step 02-2] ROI Crop based on Star Marks")
print("=" * 70)

# Get exact ROI boundaries from cross points
tl = cross_points['TL']
tr = cross_points['TR']
br = cross_points['BR']
bl = cross_points['BL']

# Calculate bounding box from star marks
roi_x_min = int(min(tl[0], bl[0]))
roi_x_max = int(max(tr[0], br[0]))
roi_y_min = int(min(tl[1], tr[1]))
roi_y_max = int(max(bl[1], br[1]))

print(f"Star Mark based ROI:")
print(f"  TL: ({int(tl[0])}, {int(tl[1])}) px")
print(f"  TR: ({int(tr[0])}, {int(tr[1])}) px")
print(f"  BR: ({int(br[0])}, {int(br[1])}) px")
print(f"  BL: ({int(bl[0])}, {int(bl[1])}) px")
print(f"\nROI Bounding Box:")
print(f"  X: {roi_x_min} ~ {roi_x_max}")
print(f"  Y: {roi_y_min} ~ {roi_y_max}")
print(f"  Size: {roi_x_max - roi_x_min} x {roi_y_max - roi_y_min} pixels")

# Crop ROI based on star marks
roi_cropped_img = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max].copy()
print(f"\nROI Cropped Image: {roi_cropped_img.shape[1]} x {roi_cropped_img.shape[0]} pixels")

# Save ROI cropped image
roi_cropped_output_path = output_dir / '12_roi_cropped_by_star_marks.tif'
cv2.imwrite(str(roi_cropped_output_path), roi_cropped_img)
print(f"ROI Cropped Image Saved: {roi_cropped_output_path}")

# =============================================================================
# Visualization
# =============================================================================
print("\n" + "=" * 70)
print("Creating Visualization...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 8-bit conversion for display
img_8bit = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
cropped_8bit = ((cropped_img - cropped_img.min()) / (cropped_img.max() - cropped_img.min()) * 255).astype(np.uint8)

# Plot 1: Original Image with ROI
ax1 = axes[0, 0]
ax1.imshow(img_8bit, cmap='gray')
pts = roi.corners.astype(np.int32)
for i in range(4):
    pt1 = pts[i]
    pt2 = pts[(i+1) % 4]
    ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g-', linewidth=2)
ax1.set_title(f'[Step 00] Original Image\n{img.shape[1]}x{img.shape[0]}, {img.dtype}', fontsize=12)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Plot 2: ROI Detection Result
ax2 = axes[0, 1]
ax2.imshow(img_8bit, cmap='gray')
# Draw ROI polygon
for i in range(4):
    pt1 = pts[i]
    pt2 = pts[(i+1) % 4]
    ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'lime', linewidth=2)
# Draw corners with labels
corner_colors = {'TL': 'blue', 'TR': 'green', 'BR': 'red', 'BL': 'cyan'}
for i, name in enumerate(corner_names):
    corner = roi.corners[i]
    cross = cross_points.get(name)
    ax2.plot(corner[0], corner[1], 'o', color=corner_colors[name], markersize=8, label=f'{name}')
    # Draw small cross at cross point
    ax2.plot(cross[0], cross[1], '+', color=corner_colors[name], markersize=10, markeredgewidth=2)
ax2.legend(loc='upper right')
ax2.set_title(f'[Step 01] ROI Detection\nTilt: {roi.angle:.4f}°, Area: {roi.area:.0f} px', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

# Plot 3: Crop Region
ax3 = axes[1, 0]
ax3.imshow(img_8bit, cmap='gray')
# Draw crop rectangle
rect_x = [x_min, x_max, x_max, x_min, x_min]
rect_y = [y_min, y_min, y_max, y_max, y_min]
ax3.plot(rect_x, rect_y, 'r-', linewidth=2, label='Crop Region')
ax3.fill(rect_x, rect_y, 'red', alpha=0.1)
ax3.legend(loc='upper right')
ax3.set_title(f'[Step 02] Crop Region\n({x_min}, {y_min}) to ({x_max}, {y_max})', fontsize=12)
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')

# Plot 4: Cropped Image
ax4 = axes[1, 1]
ax4.imshow(cropped_8bit, cmap='gray')
# Adjust corner positions for cropped image
for i, name in enumerate(corner_names):
    corner = roi.corners[i]
    ax4.plot(corner[0] - x_min, corner[1] - y_min, 'o', color=corner_colors[name], markersize=8)
ax4.set_title(f'[Step 02] Cropped Image\n{cropped_img.shape[1]}x{cropped_img.shape[0]} pixels', fontsize=12)
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')

plt.suptitle('ROI Algorithm: Step 00 ~ 02 Execution Results', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_path = output_dir / '10_steps_00_to_02_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.show()

print("\n" + "=" * 70)
print("Step 00 ~ 02 Execution Complete!")
print("=" * 70)
