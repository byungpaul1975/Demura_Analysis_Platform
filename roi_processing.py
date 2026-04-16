# -*- coding: utf-8 -*-
"""
Display Panel ROI Processing Script - Version 4

Processing Steps:
0. Crop original image first (1700,300) to (11700,9900)
1. Load image and check basic information
2. Detect ROI (display area)
3. Detect display boundary and extract corners
4. Correct tilt using warp
5. Resize to display resolution
6. Normalize to 16bit and save final result
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import os

# =============================================================================
# Step 0: Crop original image
# =============================================================================
print("=" * 60)
print("Step 0: Crop original image")
print("=" * 60)

work_dir = Path(r'c:\Users\byungpaul\Desktop\AI_Project\20260304_ROI_algorithm')
data_dir = work_dir / 'data'
output_dir = work_dir / 'output'
output_dir.mkdir(exist_ok=True)

img_path = data_dir / 'G32_cal.tif'
img_original = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

print(f"Original image path: {img_path}")
print(f"Original image shape: {img_original.shape}")
print(f"Original image dtype: {img_original.dtype}")

# Crop coordinates (x, y) format: start (1700, 300) to end (11700, 9900)
x_start, y_start = 1700, 300
x_end, y_end = 11700, 9900

# In numpy/cv2, array indexing is [row, col] = [y, x]
img = img_original[y_start:y_end, x_start:x_end]

print(f"\nCrop region: ({x_start}, {y_start}) to ({x_end}, {y_end})")
print(f"Cropped image shape: {img.shape}")
print(f"Cropped size: {img.shape[1]} x {img.shape[0]} pixels")

# Save cropped image
cv2.imwrite(str(output_dir / '00_cropped_image.tif'), img)
print(f"Saved: {output_dir / '00_cropped_image.tif'}")

# Visualization
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_original, cmap='gray')
plt.title(f'Original Image ({img_original.shape[1]}x{img_original.shape[0]})')
# Draw crop rectangle
rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                       fill=False, edgecolor='red', linewidth=2)
plt.gca().add_patch(rect)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.title(f'Cropped Image ({img.shape[1]}x{img.shape[0]})')
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / '00_crop_comparison.png', dpi=150)
print(f"Saved: {output_dir / '00_crop_comparison.png'}")
plt.close()

# =============================================================================
# Step 1: Load image and check basic information
# =============================================================================
print("\n" + "=" * 60)
print("Step 1: Check cropped image information")
print("=" * 60)

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image min: {img.min()}, max: {img.max()}")
print(f"Image mean: {img.mean():.2f}")

# Calculate histogram for analysis
hist, bins = np.histogram(img.ravel(), bins=4096, range=(0, 4096))
print(f"Background peak (0-100 range): {np.argmax(hist[:100])}")
print(f"Display peak (estimation): {np.argmax(hist[100:]) + 100}")

# Image histogram check
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Cropped Image ({img.shape[1]}x{img.shape[0]})')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0, img.max()), color='gray', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.yscale('log')

plt.tight_layout()
plt.savefig(output_dir / '01_cropped_image_info.png', dpi=150)
print(f"Saved: {output_dir / '01_cropped_image_info.png'}")
plt.close()

# =============================================================================
# Step 2: Detect ROI (display area)
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Detect ROI (display area)")
print("=" * 60)

# Normalize image (convert to 8bit for processing)
img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

# Use a threshold that separates background from display
thresh_low = 50  # Capture the display region
binary = (img > thresh_low).astype(np.uint8) * 255

print(f"Using threshold: {thresh_low}")

# Apply strong morphological operations to get main display region
kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)
binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel_large)

# Fill holes
binary_filled = ndimage.binary_fill_holes(binary_cleaned).astype(np.uint8) * 255

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_normalized, cmap='gray')
plt.title('Normalized Image (8bit)')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title(f'Binary Threshold (>{thresh_low})')

plt.subplot(2, 3, 3)
plt.imshow(binary_cleaned, cmap='gray')
plt.title('After Morphology')

plt.subplot(2, 3, 4)
plt.imshow(binary_filled, cmap='gray')
plt.title('Holes Filled')

# Show histogram with threshold line
plt.subplot(2, 3, 5)
plt.hist(img.ravel(), bins=256, range=(0, 500), color='gray', alpha=0.7)
plt.axvline(x=thresh_low, color='r', linestyle='--', label=f'Threshold={thresh_low}')
plt.title('Histogram (zoom 0-500)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig(output_dir / '02_roi_detection.png', dpi=150)
print(f"Saved: {output_dir / '02_roi_detection.png'}")
plt.close()

# =============================================================================
# Step 3: Detect display boundary and extract corners
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: Detect display boundary and extract corners")
print("=" * 60)

# Contour detection on filled binary image
contours, hierarchy = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found: {len(contours)}")

# Find the largest contour (should be the display)
contour_areas = [(cv2.contourArea(cnt), cnt) for cnt in contours]
contour_areas.sort(key=lambda x: x[0], reverse=True)

print(f"\nTop 5 contours by area:")
for i, (area, cnt) in enumerate(contour_areas[:5]):
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"  {i+1}. Area: {area:.0f}, Bounding: {w}x{h}")

# Select the largest contour
largest_contour = contour_areas[0][1]
contour_area = contour_areas[0][0]
print(f"\nSelected largest contour area: {contour_area:.0f}")

# Calculate expected display area
image_area = img.shape[0] * img.shape[1]
print(f"Cropped image area: {image_area}")
print(f"Contour covers: {contour_area/image_area*100:.1f}% of cropped image")

# Get minimum area rectangle for the selected contour
rect = cv2.minAreaRect(largest_contour)
corners = cv2.boxPoints(rect)
corners = corners.astype(np.float32)

print(f"MinAreaRect center: ({rect[0][0]:.1f}, {rect[0][1]:.1f})")
print(f"MinAreaRect size: {rect[1][0]:.1f} x {rect[1][1]:.1f}")
print(f"MinAreaRect angle: {rect[2]:.2f} degrees")

# Sort corners (top-left, top-right, bottom-right, bottom-left)
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect

ordered_corners = order_corners(corners)
print(f"Ordered corners (TL, TR, BR, BL):\n{ordered_corners}")

# Calculate width and height
width = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
height = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
print(f"Display size from corners: {width:.0f} x {height:.0f} pixels")

# Corner visualization
img_with_corners = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)

# Draw selected contour in green
cv2.drawContours(img_with_corners, [largest_contour], -1, (0, 255, 0), 3)

# Draw corner rectangle
box = np.intp(ordered_corners)
cv2.drawContours(img_with_corners, [box], 0, (255, 0, 255), 5)

# Draw corners
corner_labels = ['TL', 'TR', 'BR', 'BL']
colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 255, 0)]
for i, corner in enumerate(ordered_corners):
    cv2.circle(img_with_corners, (int(corner[0]), int(corner[1])), 20, colors[i], -1)
    cv2.putText(img_with_corners, corner_labels[i], (int(corner[0])+25, int(corner[1])+25),
                cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 4)

plt.figure(figsize=(14, 12))
plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
plt.title('Detected Display Boundary and Corners\n(Green=Contour, Magenta=Bounding Box)')
plt.savefig(output_dir / '03_corner_detection.png', dpi=150)
print(f"Saved: {output_dir / '03_corner_detection.png'}")
plt.close()

# Calculate tilt angle
top_edge = ordered_corners[1] - ordered_corners[0]
angle = np.degrees(np.arctan2(top_edge[1], top_edge[0]))
print(f"Detected tilt angle: {angle:.2f} degrees")

# =============================================================================
# Step 4: Correct tilt using warp (preserve pixel data)
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Correct tilt using warp (preserve pixel data)")
print("=" * 60)

# Calculate actual size of display area
width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])

dst_width = int(max(width_top, width_bottom))
dst_height = int(max(height_left, height_right))

print(f"Detected display size in camera pixels: {dst_width} x {dst_height}")

# Camera:Display ratio (3.8:1)
camera_to_display_ratio = 3.8
estimated_display_width = dst_width / camera_to_display_ratio
estimated_display_height = dst_height / camera_to_display_ratio
print(f"Estimated display resolution: {estimated_display_width:.0f} x {estimated_display_height:.0f}")

# Perform perspective transform (use INTER_NEAREST to preserve pixel values)
dst_corners = np.array([
    [0, 0],
    [dst_width - 1, 0],
    [dst_width - 1, dst_height - 1],
    [0, dst_height - 1]
], dtype="float32")

# Calculate transform matrix
M = cv2.getPerspectiveTransform(ordered_corners, dst_corners)

# Use INTER_NEAREST to warp without mixing pixel values
warped = cv2.warpPerspective(img, M, (dst_width, dst_height), flags=cv2.INTER_NEAREST)

print(f"Warped image shape: {warped.shape}")
print(f"Warped image dtype: {warped.dtype}")
print(f"Warped image min: {warped.min()}, max: {warped.max()}")

# Save intermediate result
warped_output = warped.astype(np.uint16) if warped.dtype != np.uint16 else warped
cv2.imwrite(str(output_dir / '04_warped_image.tif'), warped_output)
print(f"Saved: {output_dir / '04_warped_image.tif'}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Cropped Image ({img.shape[1]}x{img.shape[0]})')
# Draw ROI rectangle on original
for i in range(4):
    pt1 = tuple(ordered_corners[i].astype(int))
    pt2 = tuple(ordered_corners[(i+1)%4].astype(int))
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=2)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(warped, cmap='gray')
plt.title(f'Warped Image ({warped.shape[1]}x{warped.shape[0]})')
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / '04_warp_comparison.png', dpi=150)
print(f"Saved: {output_dir / '04_warp_comparison.png'}")
plt.close()

# =============================================================================
# Step 5: Resize to display resolution (preserve pixel data)
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: Resize to display resolution")
print("=" * 60)

# Target display resolution - FIXED to 2412 x 2288
display_width = 2412   # Fixed width
display_height = 2288  # Fixed height

print(f"Target display resolution: {display_width} x {display_height}")
print(f"(Fixed resolution as specified)")

# Area-based resize with fractional pixel summation (optimized with numba-like vectorization)
def area_sum_resize(img, target_size):
    """
    Resize by summing all camera pixels corresponding to each display pixel.
    Handles fractional boundaries by weighting pixels proportionally.

    Each display pixel = sum of all corresponding camera pixels (with fractional weights at boundaries)
    """
    h, w = img.shape
    new_h, new_w = target_size

    # Calculate scale factors
    scale_y = h / new_h
    scale_x = w / new_w

    print(f"Scale factors: {scale_x:.4f} x {scale_y:.4f} camera pixels per display pixel")
    print(f"(Each display pixel sums approximately {scale_x:.2f} x {scale_y:.2f} = {scale_x * scale_y:.2f} camera pixels)")

    # Use float64 for precision during summation
    img_float = img.astype(np.float64)
    result = np.zeros((new_h, new_w), dtype=np.float64)

    # Progress tracking
    total_pixels = new_h * new_w
    progress_step = total_pixels // 10

    pixel_count = 0
    for j in range(new_h):
        # Calculate y boundaries (fractional)
        y_start = j * scale_y
        y_end = (j + 1) * scale_y

        # Integer boundaries for y
        y_start_int = int(np.floor(y_start))
        y_end_int = min(int(np.ceil(y_end)), h)

        # Pre-calculate y weights for this row
        y_weights = np.zeros(y_end_int - y_start_int)
        for idx, yy in enumerate(range(y_start_int, y_end_int)):
            y_weight_start = max(0.0, y_start - yy)
            y_weight_end = max(0.0, yy + 1 - y_end)
            y_weights[idx] = 1.0 - y_weight_start - y_weight_end

        for i in range(new_w):
            # Calculate x boundaries (fractional)
            x_start = i * scale_x
            x_end = (i + 1) * scale_x

            # Integer boundaries for x
            x_start_int = int(np.floor(x_start))
            x_end_int = min(int(np.ceil(x_end)), w)

            # Pre-calculate x weights
            x_weights = np.zeros(x_end_int - x_start_int)
            for idx, xx in enumerate(range(x_start_int, x_end_int)):
                x_weight_start = max(0.0, x_start - xx)
                x_weight_end = max(0.0, xx + 1 - x_end)
                x_weights[idx] = 1.0 - x_weight_start - x_weight_end

            # Extract the block and compute weighted sum
            block = img_float[y_start_int:y_end_int, x_start_int:x_end_int]

            # Create weight matrix (outer product of y_weights and x_weights)
            weight_matrix = np.outer(y_weights, x_weights)

            # Compute weighted sum
            result[j, i] = np.sum(block * weight_matrix)

            pixel_count += 1

        # Progress update every 10%
        if j % (new_h // 10) == 0:
            print(f"  Progress: {j}/{new_h} rows ({100*j/new_h:.0f}%)")

    print(f"  Progress: {new_h}/{new_h} rows (100%)")
    return result

print("Resizing using area sum method (summing all camera pixels with fractional weights)...")
resized = area_sum_resize(warped, (display_height, display_width))

print(f"Resized image shape: {resized.shape}")
print(f"Resized image dtype: {resized.dtype}")
print(f"Resized image min: {resized.min()}, max: {resized.max()}")

# Save intermediate result
cv2.imwrite(str(output_dir / '05_resized_image.tif'), resized.astype(np.uint16))
print(f"Saved: {output_dir / '05_resized_image.tif'}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(warped, cmap='gray')
plt.title(f'Warped Image ({warped.shape[1]}x{warped.shape[0]})')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(resized, cmap='gray')
plt.title(f'Resized Image ({resized.shape[1]}x{resized.shape[0]})')
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / '05_resize_comparison.png', dpi=150)
print(f"Saved: {output_dir / '05_resize_comparison.png'}")
plt.close()

# =============================================================================
# Step 6: Normalize to 16bit and save final result
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: Normalize to 16bit and save final result")
print("=" * 60)

# Normalize to 16bit (0-65535 range)
min_val = resized.min()
max_val = resized.max()
normalized_16bit = ((resized - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

print(f"Original range: [{min_val}, {max_val}]")
print(f"Normalized range: [{normalized_16bit.min()}, {normalized_16bit.max()}]")

# Save final result
cv2.imwrite(str(output_dir / '06_final_normalized.tif'), normalized_16bit)
print(f"Saved: {output_dir / '06_final_normalized.tif'}")

# Also save as PNG (convert to 8bit)
normalized_8bit = (normalized_16bit / 256).astype(np.uint8)
cv2.imwrite(str(output_dir / '06_final_normalized_preview.png'), normalized_8bit)
print(f"Saved: {output_dir / '06_final_normalized_preview.png'}")

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Cropped ({img.shape[1]}x{img.shape[0]})')
# Draw ROI
for i in range(4):
    pt1 = tuple(ordered_corners[i].astype(int))
    pt2 = tuple(ordered_corners[(i+1)%4].astype(int))
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=2)
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(warped, cmap='gray')
plt.title(f'Warped ({warped.shape[1]}x{warped.shape[0]})')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(normalized_16bit, cmap='gray')
plt.title(f'Final Normalized 16bit ({normalized_16bit.shape[1]}x{normalized_16bit.shape[0]})')
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / '06_final_comparison.png', dpi=150)
print(f"Saved: {output_dir / '06_final_comparison.png'}")
plt.close()

# =============================================================================
# Additional: Pixel value analysis
# =============================================================================
print("\n" + "=" * 60)
print("Additional: Pixel value analysis")
print("=" * 60)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(resized.ravel(), bins=256, range=(0, resized.max()), color='blue', alpha=0.7)
plt.title('Pixel Distribution (Before Normalization)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(normalized_16bit.ravel(), bins=256, color='green', alpha=0.7)
plt.title('Pixel Distribution (After 16bit Normalization)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(output_dir / '07_pixel_analysis.png', dpi=150)
print(f"Saved: {output_dir / '07_pixel_analysis.png'}")
plt.close()

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 60)
print("Processing Complete - Final Summary")
print("=" * 60)
print(f"\nOriginal image: {img_original.shape[1]} x {img_original.shape[0]} pixels")
print(f"Crop region: ({x_start}, {y_start}) to ({x_end}, {y_end})")
print(f"Cropped image: {img.shape[1]} x {img.shape[0]} pixels")
print(f"Detected ROI size: {dst_width} x {dst_height} camera pixels")
print(f"Detected tilt angle: {angle:.2f} degrees")
print(f"Warped image: {warped.shape[1]} x {warped.shape[0]} pixels")
print(f"Final display resolution: {resized.shape[1]} x {resized.shape[0]} pixels")
print(f"Camera:Display ratio used: {camera_to_display_ratio}:1")
print(f"Actual ratio achieved: {dst_width/resized.shape[1]:.2f}:1 (width), {dst_height/resized.shape[0]:.2f}:1 (height)")

print(f"\nSaved files:")
print(f"  0. {output_dir / '00_cropped_image.tif'} - Cropped original")
print(f"  0. {output_dir / '00_crop_comparison.png'} - Crop visualization")
print(f"  1. {output_dir / '01_cropped_image_info.png'} - Cropped image info")
print(f"  2. {output_dir / '02_roi_detection.png'} - ROI detection")
print(f"  3. {output_dir / '03_corner_detection.png'} - Corner detection")
print(f"  4. {output_dir / '04_warped_image.tif'} - Warped image")
print(f"  5. {output_dir / '04_warp_comparison.png'} - Warp comparison")
print(f"  6. {output_dir / '05_resized_image.tif'} - Resized image")
print(f"  7. {output_dir / '05_resize_comparison.png'} - Resize comparison")
print(f"  8. {output_dir / '06_final_normalized.tif'} - Final normalized 16bit")
print(f"  9. {output_dir / '06_final_normalized_preview.png'} - Final preview (8bit)")
print(f"  10. {output_dir / '06_final_comparison.png'} - Final comparison")
print(f"  11. {output_dir / '07_pixel_analysis.png'} - Pixel analysis")
