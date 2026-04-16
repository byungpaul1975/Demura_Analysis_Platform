# -*- coding: utf-8 -*-
"""
Pitch Range Robustness Test

Test the brightest pixel detection + 3x3 box algorithm
with variable pitch ranges:
- Horizontal: 7.4 ~ 8.0 pixels
- Vertical: 3.7 ~ 4.0 pixels
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

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

# Pitch ranges to test
PITCH_H_VALUES = [7.4, 7.6, 7.7, 7.8, 8.0]  # Horizontal range
PITCH_V_VALUES = [3.7, 3.8, 3.85, 3.9, 4.0]  # Vertical range

print("=" * 70)
print("Pitch Range Robustness Test")
print("=" * 70)
print(f"Horizontal pitch range: {PITCH_H_VALUES}")
print(f"Vertical pitch range: {PITCH_V_VALUES}")
print(f"Target output: {TARGET_WIDTH} x {TARGET_HEIGHT}")

# Load image
print("\n" + "=" * 70)
print("Loading Image and Detecting ROI")
print("=" * 70)

img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# ROI Detection
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
print(f"ROI detected: {roi_result.width:.1f} x {roi_result.height:.1f}")


def find_brightest_and_group_3x3_fast(img, pitch_h, pitch_v, target_w, target_h):
    """
    Find brightest pixel in each pitch region, then group 3x3 around it.
    Optimized version without storing all block data.
    """
    h, w = img.shape

    output = np.zeros((target_h, target_w), dtype=img.dtype)
    brightest_positions = np.zeros((target_h, target_w, 2), dtype=np.int32)

    for out_y in range(target_h):
        for out_x in range(target_w):
            y_start = int(out_y * pitch_v)
            y_end = int((out_y + 1) * pitch_v)
            x_start = int(out_x * pitch_h)
            x_end = int((out_x + 1) * pitch_h)

            y_start = max(0, y_start)
            y_end = min(h, y_end)
            x_start = max(0, x_start)
            x_end = min(w, x_end)

            region = img[y_start:y_end, x_start:x_end]

            if region.size > 0:
                local_max_idx = np.unravel_index(np.argmax(region), region.shape)
                bright_y = y_start + local_max_idx[0]
                bright_x = x_start + local_max_idx[1]

                brightest_positions[out_y, out_x] = [bright_y, bright_x]

                box_y_start = max(0, bright_y - 1)
                box_y_end = min(h, bright_y + 2)
                box_x_start = max(0, bright_x - 1)
                box_x_end = min(w, bright_x + 2)

                box_3x3 = img[box_y_start:box_y_end, box_x_start:box_x_end]
                output[out_y, out_x] = np.max(box_3x3)

    return output, brightest_positions


def test_single_pitch(img, corners, pitch_h, pitch_v, target_w, target_h):
    """Test a single pitch configuration."""
    warped_width = int(target_w * pitch_h)
    warped_height = int(target_h * pitch_v)

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
    warped = cv2.warpPerspective(img, M, (warped_width, warped_height),
                                  flags=cv2.INTER_LINEAR)

    output, positions = find_brightest_and_group_3x3_fast(
        warped, pitch_h, pitch_v, target_w, target_h
    )

    return {
        'warped_shape': warped.shape,
        'output_shape': output.shape,
        'output_min': output.min(),
        'output_max': output.max(),
        'output_mean': output.mean(),
        'output_std': output.std(),
        'output': output
    }


# Test all combinations
print("\n" + "=" * 70)
print("Testing Pitch Combinations")
print("=" * 70)

results = {}
reference_output = None

# Test selected combinations (corners and center of range)
test_combinations = [
    (7.4, 3.7),   # Min-Min
    (7.4, 4.0),   # Min-Max
    (8.0, 3.7),   # Max-Min
    (8.0, 4.0),   # Max-Max
    (7.7, 3.85),  # Center (reference)
]

print(f"\nTesting {len(test_combinations)} pitch combinations...")
print("-" * 70)
print(f"{'Pitch (H, V)':<15} {'Warped Size':<18} {'Output Size':<15} {'Mean':<10} {'Std':<10} {'Time':<8}")
print("-" * 70)

for pitch_h, pitch_v in test_combinations:
    start_time = time.time()
    result = test_single_pitch(img, corners, pitch_h, pitch_v,
                                TARGET_WIDTH, TARGET_HEIGHT)
    elapsed = time.time() - start_time

    key = f"H{pitch_h}_V{pitch_v}"
    results[key] = result

    if pitch_h == 7.7 and pitch_v == 3.85:
        reference_output = result['output']

    print(f"({pitch_h}, {pitch_v}){'':<6} "
          f"{result['warped_shape'][1]}x{result['warped_shape'][0]:<7} "
          f"{result['output_shape'][1]}x{result['output_shape'][0]:<7} "
          f"{result['output_mean']:<10.1f} "
          f"{result['output_std']:<10.1f} "
          f"{elapsed:.1f}s")

print("-" * 70)

# Compare outputs
print("\n" + "=" * 70)
print("Comparison with Reference (H=7.7, V=3.85)")
print("=" * 70)

if reference_output is not None:
    print(f"\n{'Pitch (H, V)':<15} {'RMSE':<12} {'Max Diff':<12} {'Corr Coef':<12}")
    print("-" * 50)

    for key, result in results.items():
        output = result['output'].astype(np.float64)
        ref = reference_output.astype(np.float64)

        diff = output - ref
        rmse = np.sqrt(np.mean(diff ** 2))
        max_diff = np.abs(diff).max()

        # Correlation coefficient
        corr = np.corrcoef(output.flatten(), ref.flatten())[0, 1]

        print(f"{key:<15} {rmse:<12.2f} {max_diff:<12.0f} {corr:<12.6f}")

# Visualize comparison
print("\n" + "=" * 70)
print("Creating Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

keys = list(results.keys())
for idx, key in enumerate(keys[:6]):
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(results[key]['output'], cmap='gray')
    ax.set_title(f'{key}\nMean: {results[key]["output_mean"]:.1f}, '
                 f'Range: [{results[key]["output_min"]}, {results[key]["output_max"]}]')
    plt.colorbar(im, ax=ax)

plt.suptitle('Pitch Range Robustness Test\nOutput Images for Different Pitch Values',
             fontsize=14, fontweight='bold')
plt.tight_layout()

comparison_path = output_dir / 'pitch_robustness_comparison.png'
plt.savefig(comparison_path, dpi=150)
print(f"Comparison saved: {comparison_path}")
plt.close()

# Difference visualization
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

diff_keys = [k for k in keys if k != 'H7.7_V3.85'][:4]
for idx, key in enumerate(diff_keys):
    ax = axes2[idx // 2, idx % 2]
    diff = results[key]['output'].astype(np.float64) - reference_output.astype(np.float64)

    vmax = np.abs(diff).max()
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'{key} - Reference\nRMSE: {np.sqrt(np.mean(diff**2)):.2f}')
    plt.colorbar(im, ax=ax)

plt.suptitle('Difference from Reference (H=7.7, V=3.85)', fontsize=14, fontweight='bold')
plt.tight_layout()

diff_path = output_dir / 'pitch_robustness_diff.png'
plt.savefig(diff_path, dpi=150)
print(f"Difference map saved: {diff_path}")
plt.close()

# Summary statistics
print("\n" + "=" * 70)
print("Robustness Summary")
print("=" * 70)

rmse_values = []
corr_values = []

for key, result in results.items():
    if key == 'H7.7_V3.85':
        continue
    output = result['output'].astype(np.float64)
    ref = reference_output.astype(np.float64)
    rmse = np.sqrt(np.mean((output - ref) ** 2))
    corr = np.corrcoef(output.flatten(), ref.flatten())[0, 1]
    rmse_values.append(rmse)
    corr_values.append(corr)

print(f"\nAcross all tested pitch variations:")
print(f"  - RMSE range: {min(rmse_values):.2f} ~ {max(rmse_values):.2f}")
print(f"  - Correlation range: {min(corr_values):.6f} ~ {max(corr_values):.6f}")
print(f"  - Mean RMSE: {np.mean(rmse_values):.2f}")
print(f"  - Mean Correlation: {np.mean(corr_values):.6f}")

if max(corr_values) > 0.99:
    print("\n[PASS] Algorithm shows HIGH robustness across pitch range")
elif max(corr_values) > 0.95:
    print("\n[PASS] Algorithm shows GOOD robustness across pitch range")
else:
    print("\n[WARNING] Algorithm may be sensitive to pitch variations")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
print(f"\nSaved files:")
print(f"  - {comparison_path}")
print(f"  - {diff_path}")

plt.close('all')
