# -*- coding: utf-8 -*-
"""
ROI Algorithm Step 00~04 Execution
1_utils -> 2_roi_detector -> 3_image_cropper -> 4_perspective_warper -> 5_area_sum_resizer
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
roi_detector_module = import_module('2_roi_detector')
perspective_warper_module = import_module('4_perspective_warper')
area_sum_resizer_module = import_module('5_area_sum_resizer')

print("=" * 70)
print("ROI Algorithm: Step 00 ~ 04 Execution")
print("=" * 70)

# =============================================================================
# Step 00: Load Image
# =============================================================================
print("\n" + "=" * 70)
print("[Step 00] Image Load")
print("=" * 70)

img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
print(f"Original: {img.shape[1]} x {img.shape[0]} pixels, {img.dtype}")

# =============================================================================
# Step 01: ROI Detection
# =============================================================================
print("\n" + "=" * 70)
print("[Step 01] ROI Detection (Crossing Method)")
print("=" * 70)

AdaptiveROIDetector = roi_detector_module.AdaptiveROIDetector
detector = AdaptiveROIDetector(initial_threshold=200, morph_kernel_size=51, corner_region_size=60)
roi = detector.detect_by_crossing(img, edge_offset=2, search_radius=30)
cross_points = detector.get_cross_points()

corner_names = ['TL', 'TR', 'BR', 'BL']
print(f"ROI Size: {roi.width:.1f} x {roi.height:.1f} pixels")
print(f"Tilt: {roi.angle:.4f} degrees")
for name in corner_names:
    cross = cross_points[name]
    print(f"  {name}: ({int(cross[0])}, {int(cross[1])}) px")

# =============================================================================
# Step 03: Perspective Warp
# =============================================================================
print("\n" + "=" * 70)
print("[Step 03] Perspective Warp")
print("=" * 70)

PerspectiveWarper = perspective_warper_module.PerspectiveWarper
src_corners = np.array([cross_points['TL'], cross_points['TR'],
                        cross_points['BR'], cross_points['BL']], dtype=np.float32)

warper = PerspectiveWarper(interpolation=cv2.INTER_NEAREST)
warp_result = warper.warp(img, src_corners)

print(f"Warped Size: {warp_result.width} x {warp_result.height} pixels")
print(f"Interpolation: INTER_NEAREST")

# =============================================================================
# Step 04: Area Sum Resize (Display Pixel Mode)
# =============================================================================
print("\n" + "=" * 70)
print("[Step 04] Area Sum Resizer (Display Pixel Mode)")
print("=" * 70)

AreaSumResizer = area_sum_resizer_module.AreaSumResizer

# Target display resolution (2412 x 2288 for this display)
target_width = 2412
target_height = 2288

print(f"Input Size: {warp_result.width} x {warp_result.height} pixels")
print(f"Target Size: {target_width} x {target_height} pixels")
print(f"Expected Scale: {warp_result.width/target_width:.4f} x {warp_result.height/target_height:.4f}")

resizer = AreaSumResizer(show_progress=True)

# Use display pixel mode: find brightest 2x2 center, sum 4x4 pixels
resize_result = resizer.resize_display_pixel(
    warp_result.image,
    (target_width, target_height),
    pitch_range=(3.7, 4.0),
    bright_kernel_size=2,
    sum_kernel_size=4
)

print(f"\nResize Complete:")
print(f"  Output Size: {resize_result.image.shape[1]} x {resize_result.image.shape[0]} pixels")
print(f"  Scale X: {resize_result.scale_x:.4f}")
print(f"  Scale Y: {resize_result.scale_y:.4f}")
print(f"  Pixels per output: {resize_result.pixels_per_output:.2f}")
print(f"  Output Range: [{resize_result.image.min():.2f}, {resize_result.image.max():.2f}]")

# Save resized image (convert to uint16)
resized_uint16 = resize_result.image.astype(np.uint16)
resized_output_path = output_dir / '14_area_sum_resized.tif'
cv2.imwrite(str(resized_output_path), resized_uint16)
print(f"\nResized Image Saved: {resized_output_path}")

# =============================================================================
# Visualization
# =============================================================================
print("\n" + "=" * 70)
print("Creating Visualization...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Normalize for display
warped_8bit = ((warp_result.image - warp_result.image.min()) /
               (warp_result.image.max() - warp_result.image.min()) * 255).astype(np.uint8)
resized_8bit = ((resize_result.image - resize_result.image.min()) /
                (resize_result.image.max() - resize_result.image.min()) * 255).astype(np.uint8)

# Plot 1: Warped Image (Before)
ax1 = axes[0, 0]
ax1.imshow(warped_8bit, cmap='gray')
ax1.set_title(f'[Step 03] Warped Image\n{warp_result.width} x {warp_result.height} pixels', fontsize=12)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

# Plot 2: Resized Image (After)
ax2 = axes[0, 1]
ax2.imshow(resized_8bit, cmap='gray')
ax2.set_title(f'[Step 04] Area Sum Resized\n{target_width} x {target_height} pixels', fontsize=12)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

# Plot 3: Zoom comparison - top-left corner
ax3 = axes[1, 0]
zoom_size = 100
ax3.imshow(warped_8bit[0:zoom_size*4, 0:zoom_size*4], cmap='gray')
ax3.set_title(f'Warped - Top-Left Corner (400x400)\nScale: 1:1', fontsize=12)
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')

# Plot 4: Zoom comparison - resized top-left corner
ax4 = axes[1, 1]
ax4.imshow(resized_8bit[0:zoom_size, 0:zoom_size], cmap='gray')
ax4.set_title(f'Resized - Top-Left Corner (100x100)\nScale: {resize_result.scale_x:.2f}x downsampled', fontsize=12)
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')

plt.suptitle(f'ROI Algorithm: Step 04 - Area Sum Resizer\nInput: {warp_result.width}x{warp_result.height} → Output: {target_width}x{target_height}',
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_path = output_dir / '14_steps_00_to_04_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.show()

print("\n" + "=" * 70)
print("Step 00 ~ 04 Execution Complete!")
print("=" * 70)
print(f"\nPipeline Summary:")
print(f"  Original:  {img.shape[1]} x {img.shape[0]} pixels")
print(f"  Warped:    {warp_result.width} x {warp_result.height} pixels")
print(f"  Resized:   {target_width} x {target_height} pixels")
print(f"\nOutput Files:")
print(f"  - {resized_output_path}")
print(f"  - {output_path}")
