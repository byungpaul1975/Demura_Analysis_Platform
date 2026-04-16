# -*- coding: utf-8 -*-
"""
Center Region Detailed Visualization (100x100)

Show pixel values, selected brightest pixels, and 3x3 boxes
for the center 100x100 region of the panel.
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

sys.path.insert(0, str(work_dir / 'src'))

# Parameters
TARGET_WIDTH = 2412
TARGET_HEIGHT = 2288
PITCH_H = 7.7
PITCH_V = 3.85

# Visualization region (100x100 output pixels from center)
VIEW_SIZE = 100
CENTER_OUT_X = TARGET_WIDTH // 2
CENTER_OUT_Y = TARGET_HEIGHT // 2
START_OUT_X = CENTER_OUT_X - VIEW_SIZE // 2
START_OUT_Y = CENTER_OUT_Y - VIEW_SIZE // 2

print("=" * 70)
print("Center Region Detailed Visualization (100x100)")
print("=" * 70)
print(f"Output region: ({START_OUT_X}, {START_OUT_Y}) to ({START_OUT_X + VIEW_SIZE}, {START_OUT_Y + VIEW_SIZE})")

# Load and process image
print("\nLoading image and detecting ROI...")
img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

from importlib import import_module
roi_detector = import_module('2_roi_detector')
ROIDetector = roi_detector.ROIDetector

detector = ROIDetector()
roi_result = detector.detect(img)
corners = {
    'top_left': roi_result.corners[0],
    'top_right': roi_result.corners[1],
    'bottom_right': roi_result.corners[2],
    'bottom_left': roi_result.corners[3]
}

# Warp image
warped_width = int(TARGET_WIDTH * PITCH_H)
warped_height = int(TARGET_HEIGHT * PITCH_V)

src_pts = np.array([
    corners['top_left'], corners['top_right'],
    corners['bottom_right'], corners['bottom_left']
], dtype=np.float32)

dst_pts = np.array([
    [0, 0], [warped_width - 1, 0],
    [warped_width - 1, warped_height - 1], [0, warped_height - 1]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (warped_width, warped_height), flags=cv2.INTER_LINEAR)

print(f"Warped image: {warped.shape[1]} x {warped.shape[0]}")

# Calculate input region corresponding to output 100x100
input_start_x = int(START_OUT_X * PITCH_H)
input_start_y = int(START_OUT_Y * PITCH_V)
input_width = int(VIEW_SIZE * PITCH_H) + 10  # Extra margin
input_height = int(VIEW_SIZE * PITCH_V) + 10

print(f"Input region: ({input_start_x}, {input_start_y}), size: {input_width} x {input_height}")

# Extract input region
input_region = warped[input_start_y:input_start_y+input_height,
                      input_start_x:input_start_x+input_width].copy()

# Process the region to find brightest pixels and 3x3 boxes
print("\nFinding brightest pixels and 3x3 boxes...")

h, w = warped.shape
box_data = []

for out_y in range(START_OUT_Y, START_OUT_Y + VIEW_SIZE):
    row_data = []
    for out_x in range(START_OUT_X, START_OUT_X + VIEW_SIZE):
        y_start = int(out_y * PITCH_V)
        y_end = int((out_y + 1) * PITCH_V)
        x_start = int(out_x * PITCH_H)
        x_end = int((out_x + 1) * PITCH_H)

        y_start = max(0, y_start)
        y_end = min(h, y_end)
        x_start = max(0, x_start)
        x_end = min(w, x_end)

        region = warped[y_start:y_end, x_start:x_end]

        if region.size > 0:
            local_max_idx = np.unravel_index(np.argmax(region), region.shape)
            bright_y = y_start + local_max_idx[0]
            bright_x = x_start + local_max_idx[1]

            box_y_start = max(0, bright_y - 1)
            box_y_end = min(h, bright_y + 2)
            box_x_start = max(0, bright_x - 1)
            box_x_end = min(w, bright_x + 2)

            row_data.append({
                'pitch_region': (y_start, y_end, x_start, x_end),
                'brightest_pos': (bright_y, bright_x),
                'brightest_val': warped[bright_y, bright_x],
                'box_3x3_region': (box_y_start, box_y_end, box_x_start, box_x_end),
            })
        else:
            row_data.append(None)
    box_data.append(row_data)

print(f"Processed {len(box_data)} x {len(box_data[0])} output pixels")

# Create detailed visualization
print("\nCreating visualization...")

# Select a smaller region for detailed pixel value display (10x10 output pixels)
DETAIL_SIZE = 10
detail_start_row = VIEW_SIZE // 2 - DETAIL_SIZE // 2
detail_start_col = VIEW_SIZE // 2 - DETAIL_SIZE // 2

# Calculate corresponding input region
detail_input_y = int((START_OUT_Y + detail_start_row) * PITCH_V) - input_start_y
detail_input_x = int((START_OUT_X + detail_start_col) * PITCH_H) - input_start_x
detail_input_h = int(DETAIL_SIZE * PITCH_V) + 5
detail_input_w = int(DETAIL_SIZE * PITCH_H) + 5

fig, axes = plt.subplots(1, 2, figsize=(24, 12))

# Left: Input image with pixel values, brightest stars, and 3x3 boxes
ax1 = axes[0]
detail_region = input_region[detail_input_y:detail_input_y+detail_input_h,
                              detail_input_x:detail_input_x+detail_input_w].astype(np.float32)

im1 = ax1.imshow(detail_region, cmap='gray', interpolation='nearest')
ax1.set_title(f'Input Region (Center {DETAIL_SIZE}x{DETAIL_SIZE} output pixels)\n'
              f'Yellow dash: pitch region, Cyan: 3x3 box, Red star: brightest\n'
              f'Numbers: pixel values', fontsize=11)

# Draw for each output pixel in detail region
for row_idx in range(DETAIL_SIZE):
    for col_idx in range(DETAIL_SIZE):
        data = box_data[detail_start_row + row_idx][detail_start_col + col_idx]
        if data is None:
            continue

        pitch_region = data['pitch_region']
        bright_pos = data['brightest_pos']
        box_region = data['box_3x3_region']

        # Convert to local coordinates
        offset_y = int((START_OUT_Y + detail_start_row) * PITCH_V)
        offset_x = int((START_OUT_X + detail_start_col) * PITCH_H)

        local_pitch_y = pitch_region[0] - offset_y
        local_pitch_x = pitch_region[2] - offset_x
        pitch_h_size = pitch_region[3] - pitch_region[2]
        pitch_v_size = pitch_region[1] - pitch_region[0]

        local_bright_y = bright_pos[0] - offset_y
        local_bright_x = bright_pos[1] - offset_x

        local_box_y = box_region[0] - offset_y
        local_box_x = box_region[2] - offset_x
        box_h_size = box_region[3] - box_region[2]
        box_v_size = box_region[1] - box_region[0]

        # Draw pitch region (yellow dashed)
        if 0 <= local_pitch_y < detail_input_h and 0 <= local_pitch_x < detail_input_w:
            pitch_rect = plt.Rectangle((local_pitch_x - 0.5, local_pitch_y - 0.5),
                                        pitch_h_size, pitch_v_size,
                                        fill=False, edgecolor='yellow',
                                        linewidth=1, linestyle='--', alpha=0.8)
            ax1.add_patch(pitch_rect)

        # Draw 3x3 box (cyan solid)
        if 0 <= local_box_y < detail_input_h and 0 <= local_box_x < detail_input_w:
            box_rect = plt.Rectangle((local_box_x - 0.5, local_box_y - 0.5),
                                      box_h_size, box_v_size,
                                      fill=False, edgecolor='cyan',
                                      linewidth=2, alpha=0.9)
            ax1.add_patch(box_rect)

        # Draw brightest pixel star
        if 0 <= local_bright_y < detail_input_h and 0 <= local_bright_x < detail_input_w:
            ax1.plot(local_bright_x, local_bright_y, marker='*', markersize=15,
                    color='red', markeredgecolor='yellow', markeredgewidth=1)

# Show pixel values for every pixel in detail region
for py in range(min(detail_input_h, detail_region.shape[0])):
    for px in range(min(detail_input_w, detail_region.shape[1])):
        val = detail_region[py, px]
        ax1.text(px, py, f'{int(val)}', ha='center', va='center',
                fontsize=5, color='white', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.3))

plt.colorbar(im1, ax=ax1, shrink=0.8)

# Right: Full 100x100 overview with boxes
ax2 = axes[1]
im2 = ax2.imshow(input_region, cmap='gray', interpolation='nearest')
ax2.set_title(f'Full 100x100 Output Region Overview\n'
              f'Red stars: brightest pixels, Cyan boxes: 3x3 regions', fontsize=11)

# Draw all brightest positions and 3x3 boxes
for row_idx in range(VIEW_SIZE):
    for col_idx in range(VIEW_SIZE):
        data = box_data[row_idx][col_idx]
        if data is None:
            continue

        bright_pos = data['brightest_pos']
        box_region = data['box_3x3_region']

        # Convert to local coordinates
        local_bright_y = bright_pos[0] - input_start_y
        local_bright_x = bright_pos[1] - input_start_x

        local_box_y = box_region[0] - input_start_y
        local_box_x = box_region[2] - input_start_x
        box_h_size = box_region[3] - box_region[2]
        box_v_size = box_region[1] - box_region[0]

        # Draw 3x3 box
        if 0 <= local_box_y < input_height and 0 <= local_box_x < input_width:
            box_rect = plt.Rectangle((local_box_x - 0.5, local_box_y - 0.5),
                                      box_h_size, box_v_size,
                                      fill=False, edgecolor='cyan',
                                      linewidth=0.5, alpha=0.6)
            ax2.add_patch(box_rect)

        # Draw brightest star
        if 0 <= local_bright_y < input_height and 0 <= local_bright_x < input_width:
            ax2.plot(local_bright_x, local_bright_y, marker='*', markersize=3,
                    color='red', markeredgecolor='yellow', markeredgewidth=0.3)

# Mark the detail region
detail_rect = plt.Rectangle((detail_input_x - 0.5, detail_input_y - 0.5),
                              detail_input_w, detail_input_h,
                              fill=False, edgecolor='lime',
                              linewidth=2, linestyle='-')
ax2.add_patch(detail_rect)
ax2.text(detail_input_x, detail_input_y - 5, 'Detail Region',
         color='lime', fontsize=10, fontweight='bold')

plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
output_path = output_dir / 'center_100x100_detailed.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()

# Create even more detailed view (5x5 output pixels with all pixel values)
print("\nCreating super-detailed 5x5 view...")

SUPER_DETAIL_SIZE = 5
super_start_row = VIEW_SIZE // 2 - SUPER_DETAIL_SIZE // 2
super_start_col = VIEW_SIZE // 2 - SUPER_DETAIL_SIZE // 2

super_input_y = int((START_OUT_Y + super_start_row) * PITCH_V) - input_start_y
super_input_x = int((START_OUT_X + super_start_col) * PITCH_H) - input_start_x
super_input_h = int(SUPER_DETAIL_SIZE * PITCH_V) + 3
super_input_w = int(SUPER_DETAIL_SIZE * PITCH_H) + 3

fig2, ax = plt.subplots(1, 1, figsize=(20, 12))

super_region = input_region[super_input_y:super_input_y+super_input_h,
                             super_input_x:super_input_x+super_input_w].astype(np.float32)

im = ax.imshow(super_region, cmap='gray', interpolation='nearest')
ax.set_title(f'Super Detail: Center 5x5 Output Pixels\n'
             f'All pixel values shown, Red star=brightest, Cyan box=3x3 region\n'
             f'Yellow dashed=pitch search region', fontsize=12)

# Draw for each output pixel
for row_idx in range(SUPER_DETAIL_SIZE):
    for col_idx in range(SUPER_DETAIL_SIZE):
        data = box_data[super_start_row + row_idx][super_start_col + col_idx]
        if data is None:
            continue

        pitch_region = data['pitch_region']
        bright_pos = data['brightest_pos']
        box_region = data['box_3x3_region']

        offset_y = int((START_OUT_Y + super_start_row) * PITCH_V)
        offset_x = int((START_OUT_X + super_start_col) * PITCH_H)

        local_pitch_y = pitch_region[0] - offset_y
        local_pitch_x = pitch_region[2] - offset_x
        pitch_h_size = pitch_region[3] - pitch_region[2]
        pitch_v_size = pitch_region[1] - pitch_region[0]

        local_bright_y = bright_pos[0] - offset_y
        local_bright_x = bright_pos[1] - offset_x

        local_box_y = box_region[0] - offset_y
        local_box_x = box_region[2] - offset_x
        box_h_size = box_region[3] - box_region[2]
        box_v_size = box_region[1] - box_region[0]

        # Pitch region
        pitch_rect = plt.Rectangle((local_pitch_x - 0.5, local_pitch_y - 0.5),
                                    pitch_h_size, pitch_v_size,
                                    fill=False, edgecolor='yellow',
                                    linewidth=2, linestyle='--', alpha=0.9)
        ax.add_patch(pitch_rect)

        # 3x3 box
        box_rect = plt.Rectangle((local_box_x - 0.5, local_box_y - 0.5),
                                  box_h_size, box_v_size,
                                  fill=False, edgecolor='cyan',
                                  linewidth=3, alpha=1.0)
        ax.add_patch(box_rect)

        # Brightest star
        ax.plot(local_bright_x, local_bright_y, marker='*', markersize=25,
                color='red', markeredgecolor='yellow', markeredgewidth=2)

# Show ALL pixel values
for py in range(super_region.shape[0]):
    for px in range(super_region.shape[1]):
        val = super_region[py, px]
        ax.text(px, py, f'{int(val)}', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.5))

plt.colorbar(im, ax=ax, shrink=0.6)
plt.tight_layout()

super_output_path = output_dir / 'center_5x5_super_detailed.png'
plt.savefig(super_output_path, dpi=200, bbox_inches='tight')
print(f"Saved: {super_output_path}")
plt.close()

print("\n" + "=" * 70)
print("Visualization Complete")
print("=" * 70)
print(f"\nSaved files:")
print(f"  - {output_path}")
print(f"  - {super_output_path}")

plt.close('all')
