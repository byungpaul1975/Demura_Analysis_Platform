# -*- coding: utf-8 -*-
"""
ROI 알고리즘 전체 파이프라인 실행 스크립트
Display panel 이미지 처리 - 처음부터 끝까지 실행
"""
import numpy as np
import cv2
from PIL import Image
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
ROIDetector = roi_detector_module.ROIDetector

print("=" * 60)
print("Display Panel ROI Processing Pipeline")
print("=" * 60)

# =============================================================================
# Step 1: 이미지 로드 및 기본 정보 확인
# =============================================================================
print("\n" + "=" * 60)
print("Step 1: 이미지 로드 및 기본 정보 확인")
print("=" * 60)

img_path = data_dir / 'G32_cal.tif'
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

print(f"이미지 경로: {img_path}")
print(f"이미지 shape: {img.shape}")
print(f"이미지 dtype: {img.dtype}")
print(f"이미지 min: {img.min()}, max: {img.max()}")
print(f"이미지 mean: {img.mean():.2f}")

# 이미지 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Original Image ({img.shape[1]}x{img.shape[0]})')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.hist(img.ravel(), bins=256, range=(0, img.max()), color='gray', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.yscale('log')

plt.tight_layout()
plt.savefig(output_dir / '01_original_image.png', dpi=150)
print(f"저장: {output_dir / '01_original_image.png'}")

# =============================================================================
# Step 2: ROI (디스플레이 영역) 검출
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: ROI (디스플레이 영역) 검출")
print("=" * 60)

# ROI Detector 초기화 (threshold와 morph_kernel_size 설정)
detector = ROIDetector(threshold=460, morph_kernel_size=51)

# ROI 검출
detected_roi = detector.detect(img)

# 8비트 변환 (시각화용)
img_8bit = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

if detected_roi:
    print(f"검출 성공!")
    print(f"  Area: {detected_roi.area:.0f} pixels")
    print(f"  Width: {detected_roi.width:.1f} pixels")
    print(f"  Height: {detected_roi.height:.1f} pixels")
    print(f"  Tilt Angle: {detected_roi.angle:.3f} degrees")

    # ROI 영역 표시
    img_color = cv2.cvtColor(img_8bit.copy(), cv2.COLOR_GRAY2BGR)
    pts = detected_roi.corners.astype(np.int32)
    cv2.polylines(img_color, [pts], True, (0, 255, 0), 5)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title(f'Initial ROI Detection (Area={detected_roi.area:.0f}, Angle={detected_roi.angle:.3f}°)')
    plt.savefig(output_dir / '02_initial_roi.png', dpi=150)
    print(f"저장: {output_dir / '02_initial_roi.png'}")
else:
    print("ROI 검출 실패")

# =============================================================================
# Step 3: 디스플레이 경계 검출 및 코너 추출
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: 디스플레이 경계 검출 및 코너 추출")
print("=" * 60)

# 코너 좌표 추출 (TL, TR, BR, BL 순서)
corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
corners = {}
for i, name in enumerate(corner_names):
    corners[name] = detected_roi.corners[i]

print("검출된 코너 좌표:")
for name, point in corners.items():
    print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")

# 코너 시각화
plt.figure(figsize=(15, 10))

img_corners = cv2.cvtColor(img_8bit.copy(), cv2.COLOR_GRAY2BGR)

corner_colors = {
    'top_left': (255, 0, 0),
    'top_right': (0, 255, 0),
    'bottom_left': (0, 0, 255),
    'bottom_right': (255, 255, 0)
}

for name, point in corners.items():
    pt = (int(point[0]), int(point[1]))
    cv2.circle(img_corners, pt, 20, corner_colors[name], -1)
    cv2.putText(img_corners, name, (pt[0] + 30, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, corner_colors[name], 3)

# 코너를 연결하는 사각형
pts = np.array([
    corners['top_left'],
    corners['top_right'],
    corners['bottom_right'],
    corners['bottom_left']
], dtype=np.int32)
cv2.polylines(img_corners, [pts], True, (0, 255, 255), 3)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
plt.title('Detected Corners')

plt.subplot(1, 2, 2)
min_x = int(min(pt[0] for pt in corners.values())) - 50
max_x = int(max(pt[0] for pt in corners.values())) + 50
min_y = int(min(pt[1] for pt in corners.values())) - 50
max_y = int(max(pt[1] for pt in corners.values())) + 50

plt.imshow(cv2.cvtColor(img_corners[max(0,min_y):max_y, max(0,min_x):max_x], cv2.COLOR_BGR2RGB))
plt.title('Zoomed Corner Region')

plt.tight_layout()
plt.savefig(output_dir / '03_corners.png', dpi=150)
print(f"저장: {output_dir / '03_corners.png'}")

# =============================================================================
# Step 4: Warp를 통한 Tilt 보정
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Warp를 통한 Tilt 보정")
print("=" * 60)

display_width = 2160
display_height = 2160

src_pts = np.array([
    corners['top_left'],
    corners['top_right'],
    corners['bottom_right'],
    corners['bottom_left']
], dtype=np.float32)

dst_pts = np.array([
    [0, 0],
    [display_width - 1, 0],
    [display_width - 1, display_height - 1],
    [0, display_height - 1]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

print("Perspective Transform Matrix:")
print(M)

warped = cv2.warpPerspective(img, M, (display_width, display_height),
                              flags=cv2.INTER_LINEAR)

print(f"\n원본 이미지 shape: {img.shape}")
print(f"Warped 이미지 shape: {warped.shape}")
print(f"Warped 이미지 dtype: {warped.dtype}")
print(f"Warped min: {warped.min()}, max: {warped.max()}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Original ({img.shape[1]}x{img.shape[0]})')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(warped, cmap='gray')
plt.title(f'Warped ({warped.shape[1]}x{warped.shape[0]})')
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / '04_warped.png', dpi=150)
print(f"저장: {output_dir / '04_warped.png'}")

# =============================================================================
# Step 5: 디스플레이 해상도로 리사이즈
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: 디스플레이 해상도로 리사이즈")
print("=" * 60)

final_width = display_width
final_height = display_height

if warped.shape[1] == final_width and warped.shape[0] == final_height:
    resized = warped.copy()
    print(f"이미 목표 해상도입니다: {final_width}x{final_height}")
else:
    resized = cv2.resize(warped, (final_width, final_height),
                          interpolation=cv2.INTER_LINEAR)
    print(f"리사이즈 완료: {warped.shape[1]}x{warped.shape[0]} -> {final_width}x{final_height}")

print(f"최종 이미지 shape: {resized.shape}")
print(f"최종 이미지 dtype: {resized.dtype}")

# =============================================================================
# Step 6: 16bit 정규화 및 최종 결과 저장
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: 16bit 정규화 및 최종 결과 저장")
print("=" * 60)

resized_min = resized.min()
resized_max = resized.max()

if resized_max > resized_min:
    normalized = ((resized - resized_min) / (resized_max - resized_min) * 65535).astype(np.uint16)
else:
    normalized = np.zeros_like(resized, dtype=np.uint16)

print(f"정규화 전 범위: [{resized_min}, {resized_max}]")
print(f"정규화 후 범위: [{normalized.min()}, {normalized.max()}]")
print(f"정규화된 이미지 dtype: {normalized.dtype}")

output_filename = 'G32_cal_processed.tif'
output_path = output_dir / output_filename

Image.fromarray(normalized).save(str(output_path))
print(f"결과 저장 완료: {output_path}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Original Image\n{img.shape[1]}x{img.shape[0]}, range: [{img.min()}, {img.max()}]')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(normalized, cmap='gray')
plt.title(f'Processed Result\n{normalized.shape[1]}x{normalized.shape[0]}, range: [0, 65535]')
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / '05_final_result.png', dpi=150)
print(f"저장: {output_dir / '05_final_result.png'}")

# =============================================================================
# 처리 완료 요약
# =============================================================================
print("\n" + "=" * 60)
print("처리 완료 요약")
print("=" * 60)
print(f"입력: {img_path}")
print(f"출력: {output_path}")
print(f"원본 크기: {img.shape[1]}x{img.shape[0]}")
print(f"출력 크기: {normalized.shape[1]}x{normalized.shape[0]}")
print(f"출력 포맷: 16-bit TIFF")
print("=" * 60)

plt.show()
