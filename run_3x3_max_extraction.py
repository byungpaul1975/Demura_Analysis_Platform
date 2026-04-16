# -*- coding: utf-8 -*-
"""
Brightest Pixel Detection with 3x3 Box Grouping

1. Find brightest pixel in each pitch region
   - Horizontal pitch: 7.4 ~ 8.0 pixels
   - Vertical pitch: 3.7 ~ 4.0 pixels
2. Center 3x3 box around the brightest pixel
3. Visualize with boxes
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Working directory setup
work_dir = Path(r'c:\Users\byungpaul\Desktop\AI_Project\20260304_ROI_algorithm')
data_dir = work_dir / 'data'
output_dir = work_dir / 'output'
output_dir.mkdir(exist_ok=True)

# Add src module path
sys.path.insert(0, str(work_dir / 'src'))

# Target output resolution
TARGET_WIDTH = 2412
TARGET_HEIGHT = 2288

# Pitch values (camera pixels per display pixel)
PITCH_H = 7.7  # Horizontal: 7.4 ~ 8.0 (use middle value)
PITCH_V = 3.85  # Vertical: 3.7 ~ 4.0 (use middle value)

print("=" * 60)
print("Brightest Pixel Detection with 3x3 Box Grouping")
print("=" * 60)
print(f"Horizontal pitch: {PITCH_H} pixels")
print(f"Vertical pitch: {PITCH_V} pixels")

print("\n" + "=" * 60)
print("Step 1: Load Image")
print("=" * 60)

img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

print(f"Image path: {img_path}")
print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image min: {img.min()}, max: {img.max()}")

print("\n" + "=" * 60)
print("Step 2: ROI Detection")
print("=" * 60)

from importlib import import_module
roi_detector = import_module('2_roi_detector')
ROIDetector = roi_detector.ROIDetector

detector = ROIDetector()
roi_result = detector.detect(img)

if roi_result:
    print(f"ROI detected: area={roi_result.area:.0f}, size={roi_result.width:.1f}x{roi_result.height:.1f}")
    corners = {
        'top_left': roi_result.corners[0],
        'top_right': roi_result.corners[1],
        'bottom_right': roi_result.corners[2],
        'bottom_left': roi_result.corners[3]
    }
else:
    print("ROI detection failed")
    raise Exception("ROI detection failed")

print("\nDetected corners:")
for name, point in corners.items():
    print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")

print("\n" + "=" * 60)
print("Step 3: Perspective Warp (Preserve Original Pitch)")
print("=" * 60)

# Calculate warped size based on pitch to match target output
# warped_width = TARGET_WIDTH * PITCH_H
# warped_height = TARGET_HEIGHT * PITCH_V

warped_width = int(TARGET_WIDTH * PITCH_H)  # ~18572
warped_height = int(TARGET_HEIGHT * PITCH_V)  # ~8809

print(f"Target output resolution: {TARGET_WIDTH} x {TARGET_HEIGHT}")
print(f"Calculated warped size: {warped_width} x {warped_height}")

src_pts = np.array([
    corners['top_left'],
    corners['top_right'],
    corners['bottom_right'],
    corners['bottom_left']
], dtype=np.float32)

dst_pts = np.array([
    [0, 0],
    [warped_width - 1, 0],
    [warped_width - 1, warped_height - 1],
    [0, warped_height - 1]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply Warp
warped = cv2.warpPerspective(img, M, (warped_width, warped_height),
                              flags=cv2.INTER_LINEAR)

print(f"\nWarped image shape: {warped.shape}")
print(f"Warped image dtype: {warped.dtype}")
print(f"Warped min: {warped.min()}, max: {warped.max()}")

print("\n" + "=" * 60)
print("Step 4: Find Brightest Pixel in Each Pitch Region")
print("=" * 60)

def find_brightest_and_group_3x3(img, pitch_h, pitch_v, target_w, target_h):
    """
    Find brightest pixel in each pitch region, then group 3x3 around it.

    Args:
        img: Input warped image
        pitch_h: Horizontal pitch (7.4~8.0)
        pitch_v: Vertical pitch (3.7~4.0)
        target_w: Target output width
        target_h: Target output height

    Returns:
        output: Output image (max values from 3x3 around brightest)
        brightest_positions: (y, x) of brightest pixel for each output pixel
        box_data: Data for visualization
    """
    h, w = img.shape

    print(f"Input image: {w} x {h}")
    print(f"Pitch: H={pitch_h:.2f}, V={pitch_v:.2f}")
    print(f"Target output: {target_w} x {target_h}")

    output = np.zeros((target_h, target_w), dtype=img.dtype)
    brightest_positions = np.zeros((target_h, target_w, 2), dtype=np.float32)
    box_data = []

    for out_y in range(target_h):
        row_data = []
        for out_x in range(target_w):
            # Calculate the pitch region for this output pixel
            y_center = (out_y + 0.5) * pitch_v
            x_center = (out_x + 0.5) * pitch_h

            # Define search region (full pitch area)
            y_start = int(out_y * pitch_v)
            y_end = int((out_y + 1) * pitch_v)
            x_start = int(out_x * pitch_h)
            x_end = int((out_x + 1) * pitch_h)

            # Clip to image bounds
            y_start = max(0, y_start)
            y_end = min(h, y_end)
            x_start = max(0, x_start)
            x_end = min(w, x_end)

            # Extract region and find brightest pixel
            region = img[y_start:y_end, x_start:x_end]

            if region.size > 0:
                # Find brightest pixel position in region
                local_max_idx = np.unravel_index(np.argmax(region), region.shape)
                bright_y = y_start + local_max_idx[0]
                bright_x = x_start + local_max_idx[1]

                brightest_positions[out_y, out_x] = [bright_y, bright_x]

                # Now extract 3x3 box centered on brightest pixel
                box_y_start = max(0, bright_y - 1)
                box_y_end = min(h, bright_y + 2)
                box_x_start = max(0, bright_x - 1)
                box_x_end = min(w, bright_x + 2)

                box_3x3 = img[box_y_start:box_y_end, box_x_start:box_x_end]

                # Output is max of 3x3 (which should be the center, the brightest)
                output[out_y, out_x] = np.max(box_3x3)

                row_data.append({
                    'pitch_region': (y_start, y_end, x_start, x_end),
                    'brightest_pos': (bright_y, bright_x),
                    'brightest_val': img[bright_y, bright_x],
                    'box_3x3_region': (box_y_start, box_y_end, box_x_start, box_x_end),
                    'box_3x3': box_3x3.copy(),
                    'output_val': output[out_y, out_x]
                })
            else:
                row_data.append(None)

        box_data.append(row_data)

        if out_y % 500 == 0:
            print(f"  Progress: {out_y}/{target_h} rows ({100*out_y/target_h:.1f}%)")

    print(f"  Progress: {target_h}/{target_h} rows (100%)")

    return output, brightest_positions, box_data

output_img, brightest_positions, box_data = find_brightest_and_group_3x3(
    warped, PITCH_H, PITCH_V, TARGET_WIDTH, TARGET_HEIGHT
)

print(f"\nOutput image shape: {output_img.shape}")
print(f"Output image dtype: {output_img.dtype}")
print(f"Output min: {output_img.min()}, max: {output_img.max()}")

print("\n" + "=" * 60)
print("Step 5: Visualization with 3x3 Boxes")
print("=" * 60)

def visualize_brightest_with_3x3_boxes(warped, box_data, output_img,
                                        pitch_h, pitch_v,
                                        sample_rows=5, sample_cols=6,
                                        start_row=0, start_col=0):
    """
    Visualize pitch regions, brightest pixels (star), and 3x3 boxes.
    """
    # Calculate the input region to show
    input_h = int((sample_rows + 1) * pitch_v)
    input_w = int((sample_cols + 1) * pitch_h)
    input_y_start = int(start_row * pitch_v)
    input_x_start = int(start_col * pitch_h)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Left: Input image with pitch regions, brightest stars, and 3x3 boxes
    ax1 = axes[0]
    input_region = warped[input_y_start:input_y_start+input_h,
                          input_x_start:input_x_start+input_w].astype(np.float32)

    im1 = ax1.imshow(input_region, cmap='gray', interpolation='nearest')
    ax1.set_title(f'Input Region\n'
                  f'Pitch: H={pitch_h:.1f}, V={pitch_v:.1f}\n'
                  f'Yellow rect: pitch region, Cyan rect: 3x3 box, Red star: brightest')

    # Draw pitch regions and 3x3 boxes
    for row_idx in range(sample_rows):
        for col_idx in range(sample_cols):
            data = box_data[start_row + row_idx][start_col + col_idx]
            if data is None:
                continue

            pitch_region = data['pitch_region']
            bright_pos = data['brightest_pos']
            box_region = data['box_3x3_region']

            # Convert to local coordinates
            local_pitch_y = pitch_region[0] - input_y_start
            local_pitch_x = pitch_region[2] - input_x_start
            pitch_h_size = pitch_region[3] - pitch_region[2]
            pitch_v_size = pitch_region[1] - pitch_region[0]

            local_bright_y = bright_pos[0] - input_y_start
            local_bright_x = bright_pos[1] - input_x_start

            local_box_y = box_region[0] - input_y_start
            local_box_x = box_region[2] - input_x_start
            box_h_size = box_region[3] - box_region[2]
            box_v_size = box_region[1] - box_region[0]

            # Draw pitch region (yellow dashed)
            pitch_rect = plt.Rectangle((local_pitch_x - 0.5, local_pitch_y - 0.5),
                                        pitch_h_size, pitch_v_size,
                                        fill=False, edgecolor='yellow',
                                        linewidth=1, linestyle='--', alpha=0.7)
            ax1.add_patch(pitch_rect)

            # Draw 3x3 box (cyan solid)
            box_rect = plt.Rectangle((local_box_x - 0.5, local_box_y - 0.5),
                                      box_h_size, box_v_size,
                                      fill=False, edgecolor='cyan',
                                      linewidth=2, alpha=0.9)
            ax1.add_patch(box_rect)

            # Draw brightest pixel star
            ax1.plot(local_bright_x, local_bright_y, marker='*', markersize=12,
                    color='red', markeredgecolor='yellow', markeredgewidth=1)

            # Show brightest value
            ax1.text(local_bright_x, local_bright_y + 1.5,
                    f'{int(data["brightest_val"])}',
                    ha='center', va='top', fontsize=6,
                    color='red', fontweight='bold')

    plt.colorbar(im1, ax=ax1)

    # Right: Output image (corresponding region)
    ax2 = axes[1]
    output_region = output_img[start_row:start_row+sample_rows,
                               start_col:start_col+sample_cols]

    im2 = ax2.imshow(output_region, cmap='gray', interpolation='nearest')
    ax2.set_title(f'Output (Max of 3x3 around brightest)\nSize: {sample_cols} x {sample_rows}')

    # Add value labels to output
    for oy in range(sample_rows):
        for ox in range(sample_cols):
            val = output_region[oy, ox]
            ax2.text(ox, oy, f'{int(val)}', ha='center', va='center',
                    fontsize=8, color='red', fontweight='bold')

    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    return fig

# Create detailed visualization for a sample region
fig_detail = visualize_brightest_with_3x3_boxes(
    warped, box_data, output_img,
    PITCH_H, PITCH_V,
    sample_rows=6, sample_cols=8,
    start_row=0, start_col=0
)
detail_path = output_dir / 'brightest_3x3_box_visualization.png'
fig_detail.savefig(detail_path, dpi=200, bbox_inches='tight')
print(f"Detail visualization saved: {detail_path}")
plt.close(fig_detail)

# Create another visualization at different location (center area)
center_row = TARGET_HEIGHT // 2
center_col = TARGET_WIDTH // 2
fig_center = visualize_brightest_with_3x3_boxes(
    warped, box_data, output_img,
    PITCH_H, PITCH_V,
    sample_rows=6, sample_cols=8,
    start_row=center_row, start_col=center_col
)
center_path = output_dir / 'brightest_3x3_box_center.png'
fig_center.savefig(center_path, dpi=200, bbox_inches='tight')
print(f"Center visualization saved: {center_path}")
plt.close(fig_center)

# Create overall comparison visualization
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14))

# Full input
ax1 = axes2[0, 0]
im1 = ax1.imshow(warped, cmap='gray')
ax1.set_title(f'Input (Warped): {warped.shape[1]} x {warped.shape[0]}')
plt.colorbar(im1, ax=ax1)

# Full output
ax2 = axes2[0, 1]
im2 = ax2.imshow(output_img, cmap='gray')
ax2.set_title(f'Output: {output_img.shape[1]} x {output_img.shape[0]}')
plt.colorbar(im2, ax=ax2)

# Zoomed input with boxes
zoom_rows = 8
zoom_cols = 10
ax3 = axes2[1, 0]
input_zoom_h = int((zoom_rows + 1) * PITCH_V)
input_zoom_w = int((zoom_cols + 1) * PITCH_H)
input_zoom = warped[:input_zoom_h, :input_zoom_w].copy()
im3 = ax3.imshow(input_zoom, cmap='gray')
ax3.set_title(f'Input Zoomed (top-left)\nYellow: pitch region, Cyan: 3x3 box, Red star: brightest')

# Draw boxes
for row_idx in range(zoom_rows):
    for col_idx in range(zoom_cols):
        data = box_data[row_idx][col_idx]
        if data is None:
            continue

        box_region = data['box_3x3_region']
        bright_pos = data['brightest_pos']

        # 3x3 box
        box_rect = plt.Rectangle(
            (box_region[2] - 0.5, box_region[0] - 0.5),
            box_region[3] - box_region[2],
            box_region[1] - box_region[0],
            fill=False, edgecolor='cyan', linewidth=1, alpha=0.8
        )
        ax3.add_patch(box_rect)

        # Brightest star
        ax3.plot(bright_pos[1], bright_pos[0], marker='*', markersize=6,
                color='red', markeredgecolor='yellow', markeredgewidth=0.5)

plt.colorbar(im3, ax=ax3)

# Zoomed output
ax4 = axes2[1, 1]
output_zoom = output_img[:zoom_rows, :zoom_cols]
im4 = ax4.imshow(output_zoom, cmap='gray', interpolation='nearest')
ax4.set_title(f'Output Zoomed (top-left {zoom_cols}x{zoom_rows})')
plt.colorbar(im4, ax=ax4)

plt.tight_layout()
result_path = output_dir / 'brightest_3x3_result.png'
fig2.savefig(result_path, dpi=150)
print(f"Result image saved: {result_path}")
plt.close(fig2)

# Save TIFF files
input_save_path = output_dir / 'brightest_input_warped.tif'
Image.fromarray(warped).save(str(input_save_path))
print(f"Input image saved: {input_save_path}")

output_save_path = output_dir / 'brightest_3x3_output.tif'
Image.fromarray(output_img).save(str(output_save_path))
print(f"Output image saved: {output_save_path}")

print("\n" + "=" * 60)
print("Processing Complete")
print("=" * 60)
print(f"Input (warped) size: {warped.shape[1]} x {warped.shape[0]}")
print(f"Output size: {output_img.shape[1]} x {output_img.shape[0]}")
print(f"Target size: {TARGET_WIDTH} x {TARGET_HEIGHT}")
print(f"\nPitch settings:")
print(f"  - Horizontal pitch: {PITCH_H} pixels (range: 7.4~8.0)")
print(f"  - Vertical pitch: {PITCH_V} pixels (range: 3.7~4.0)")
print(f"\nAlgorithm:")
print(f"  1. Find brightest pixel in each pitch region")
print(f"  2. Center 3x3 box around the brightest pixel")
print(f"  3. Output max value of the 3x3 box")
print(f"\nSaved files:")
print(f"  - {detail_path}")
print(f"  - {center_path}")
print(f"  - {result_path}")
print(f"  - {input_save_path}")
print(f"  - {output_save_path}")

plt.close('all')
