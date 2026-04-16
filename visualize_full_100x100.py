# -*- coding: utf-8 -*-
"""
Full 100x100 Center Region Visualization

Show all 100x100 output pixels with:
- Brightest pixels (stars)
- 3x3 boxes
- Pitch regions
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

work_dir = Path(r'c:\Users\byungpaul\Desktop\AI_Project\20260304_ROI_algorithm')
data_dir = work_dir / 'data'
output_dir = work_dir / 'output'
sys.path.insert(0, str(work_dir / 'src'))

TARGET_WIDTH = 2412
TARGET_HEIGHT = 2288
PITCH_H = 7.7
PITCH_V = 3.85

VIEW_SIZE = 100
CENTER_OUT_X = TARGET_WIDTH // 2
CENTER_OUT_Y = TARGET_HEIGHT // 2
START_OUT_X = CENTER_OUT_X - VIEW_SIZE // 2
START_OUT_Y = CENTER_OUT_Y - VIEW_SIZE // 2

print("=" * 70)
print("Full 100x100 Center Region Visualization")
print("=" * 70)

# Load and process
img = cv2.imread(str(data_dir / 'G32_cal.tif'), cv2.IMREAD_UNCHANGED)

from importlib import import_module
ROIDetector = import_module('2_roi_detector').ROIDetector
roi_result = ROIDetector().detect(img)
corners = {
    'top_left': roi_result.corners[0],
    'top_right': roi_result.corners[1],
    'bottom_right': roi_result.corners[2],
    'bottom_left': roi_result.corners[3]
}

warped_width = int(TARGET_WIDTH * PITCH_H)
warped_height = int(TARGET_HEIGHT * PITCH_V)

src_pts = np.array([corners['top_left'], corners['top_right'],
                    corners['bottom_right'], corners['bottom_left']], dtype=np.float32)
dst_pts = np.array([[0, 0], [warped_width - 1, 0],
                    [warped_width - 1, warped_height - 1], [0, warped_height - 1]], dtype=np.float32)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (warped_width, warped_height), flags=cv2.INTER_LINEAR)

print(f"Warped: {warped.shape[1]} x {warped.shape[0]}")

# Extract 100x100 input region
input_start_x = int(START_OUT_X * PITCH_H)
input_start_y = int(START_OUT_Y * PITCH_V)
input_width = int(VIEW_SIZE * PITCH_H) + 10
input_height = int(VIEW_SIZE * PITCH_V) + 10

input_region = warped[input_start_y:input_start_y+input_height,
                      input_start_x:input_start_x+input_width].copy()

print(f"Input region: {input_width} x {input_height}")

# Find brightest and 3x3 boxes for all 100x100
print("Processing 100x100 output pixels...")
h, w = warped.shape
box_data = []
output_values = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.uint16)

for out_y in range(VIEW_SIZE):
    row_data = []
    for out_x in range(VIEW_SIZE):
        abs_out_y = START_OUT_Y + out_y
        abs_out_x = START_OUT_X + out_x

        y_start = int(abs_out_y * PITCH_V)
        y_end = int((abs_out_y + 1) * PITCH_V)
        x_start = int(abs_out_x * PITCH_H)
        x_end = int((abs_out_x + 1) * PITCH_H)

        y_start, y_end = max(0, y_start), min(h, y_end)
        x_start, x_end = max(0, x_start), min(w, x_end)

        region = warped[y_start:y_end, x_start:x_end]

        if region.size > 0:
            local_max_idx = np.unravel_index(np.argmax(region), region.shape)
            bright_y = y_start + local_max_idx[0]
            bright_x = x_start + local_max_idx[1]

            box_y_start = max(0, bright_y - 1)
            box_y_end = min(h, bright_y + 2)
            box_x_start = max(0, bright_x - 1)
            box_x_end = min(w, bright_x + 2)

            output_values[out_y, out_x] = np.max(warped[box_y_start:box_y_end, box_x_start:box_x_end])

            row_data.append({
                'pitch_region': (y_start, y_end, x_start, x_end),
                'brightest_pos': (bright_y, bright_x),
                'brightest_val': warped[bright_y, bright_x],
                'box_3x3_region': (box_y_start, box_y_end, box_x_start, box_x_end),
            })
        else:
            row_data.append(None)
    box_data.append(row_data)

print("Creating visualization...")

# Create high-resolution figure for 100x100
fig, axes = plt.subplots(1, 2, figsize=(32, 16))

# Left: Input image with all 100x100 boxes and stars
ax1 = axes[0]
im1 = ax1.imshow(input_region, cmap='gray', interpolation='nearest')
ax1.set_title(f'Input Region: 100x100 Output Pixels\n'
              f'Size: {input_width}x{input_height} camera pixels\n'
              f'Red star=brightest, Cyan box=3x3', fontsize=14)

# Draw all boxes and stars
for out_y in range(VIEW_SIZE):
    for out_x in range(VIEW_SIZE):
        data = box_data[out_y][out_x]
        if data is None:
            continue

        bright_pos = data['brightest_pos']
        box_region = data['box_3x3_region']

        # Local coordinates
        local_bright_y = bright_pos[0] - input_start_y
        local_bright_x = bright_pos[1] - input_start_x

        local_box_y = box_region[0] - input_start_y
        local_box_x = box_region[2] - input_start_x
        box_h = box_region[3] - box_region[2]
        box_v = box_region[1] - box_region[0]

        # Draw 3x3 box
        if 0 <= local_box_y < input_height and 0 <= local_box_x < input_width:
            rect = plt.Rectangle((local_box_x - 0.5, local_box_y - 0.5),
                                  box_h, box_v,
                                  fill=False, edgecolor='cyan',
                                  linewidth=0.8, alpha=0.7)
            ax1.add_patch(rect)

        # Draw brightest star
        if 0 <= local_bright_y < input_height and 0 <= local_bright_x < input_width:
            ax1.plot(local_bright_x, local_bright_y, marker='*', markersize=4,
                    color='red', markeredgecolor='yellow', markeredgewidth=0.3)

plt.colorbar(im1, ax=ax1, shrink=0.8)

# Right: Output image (100x100)
ax2 = axes[1]
im2 = ax2.imshow(output_values, cmap='gray', interpolation='nearest')
ax2.set_title(f'Output: 100x100 pixels\n'
              f'Each pixel = max of 3x3 around brightest\n'
              f'Range: [{output_values.min()}, {output_values.max()}]', fontsize=14)
plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
output_path = output_dir / 'center_100x100_full.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()

# Create zoomed sections (4 quadrants)
fig2, axes2 = plt.subplots(2, 2, figsize=(24, 24))

quadrants = [
    (0, 0, "Top-Left"),
    (0, 50, "Top-Right"),
    (50, 0, "Bottom-Left"),
    (50, 50, "Bottom-Right")
]

for idx, (start_row, start_col, name) in enumerate(quadrants):
    ax = axes2[idx // 2, idx % 2]

    # Calculate input region for this quadrant
    q_input_y = int((START_OUT_Y + start_row) * PITCH_V) - input_start_y
    q_input_x = int((START_OUT_X + start_col) * PITCH_H) - input_start_x
    q_input_h = int(50 * PITCH_V) + 3
    q_input_w = int(50 * PITCH_H) + 3

    q_region = input_region[q_input_y:q_input_y+q_input_h,
                            q_input_x:q_input_x+q_input_w].astype(np.float32)

    im = ax.imshow(q_region, cmap='gray', interpolation='nearest')
    ax.set_title(f'{name} Quadrant (50x50 output pixels)\nRed star=brightest, Cyan=3x3 box', fontsize=12)

    # Draw boxes and stars for this quadrant
    for out_y in range(start_row, min(start_row + 50, VIEW_SIZE)):
        for out_x in range(start_col, min(start_col + 50, VIEW_SIZE)):
            data = box_data[out_y][out_x]
            if data is None:
                continue

            bright_pos = data['brightest_pos']
            box_region_data = data['box_3x3_region']

            offset_y = int((START_OUT_Y + start_row) * PITCH_V)
            offset_x = int((START_OUT_X + start_col) * PITCH_H)

            local_bright_y = bright_pos[0] - offset_y
            local_bright_x = bright_pos[1] - offset_x

            local_box_y = box_region_data[0] - offset_y
            local_box_x = box_region_data[2] - offset_x
            box_h = box_region_data[3] - box_region_data[2]
            box_v = box_region_data[1] - box_region_data[0]

            if 0 <= local_box_y < q_input_h and 0 <= local_box_x < q_input_w:
                rect = plt.Rectangle((local_box_x - 0.5, local_box_y - 0.5),
                                      box_h, box_v,
                                      fill=False, edgecolor='cyan',
                                      linewidth=1.0, alpha=0.8)
                ax.add_patch(rect)

            if 0 <= local_bright_y < q_input_h and 0 <= local_bright_x < q_input_w:
                ax.plot(local_bright_x, local_bright_y, marker='*', markersize=6,
                        color='red', markeredgecolor='yellow', markeredgewidth=0.5)

    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('100x100 Center Region - 4 Quadrants (50x50 each)', fontsize=16, fontweight='bold')
plt.tight_layout()

quadrant_path = output_dir / 'center_100x100_quadrants.png'
plt.savefig(quadrant_path, dpi=200, bbox_inches='tight')
print(f"Saved: {quadrant_path}")
plt.close()

print("\n" + "=" * 70)
print("Visualization Complete")
print("=" * 70)
print(f"\nOutput value statistics:")
print(f"  Min: {output_values.min()}")
print(f"  Max: {output_values.max()}")
print(f"  Mean: {output_values.mean():.1f}")
print(f"  Std: {output_values.std():.1f}")
print(f"\nSaved files:")
print(f"  - {output_path}")
print(f"  - {quadrant_path}")

plt.close('all')
