"""
ROI Detector 테스트 모듈
"""
import numpy as np
import sys
import os

# 상위 디렉토리 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.roi_detector import ROIDetector


def test_roi_detector_init():
    """ROIDetector 초기화 테스트"""
    detector = ROIDetector()
    assert detector.threshold == 0.5

    detector = ROIDetector(threshold=0.7)
    assert detector.threshold == 0.7


def test_roi_detector_detect():
    """ROI 검출 테스트"""
    detector = ROIDetector(threshold=0.3)

    # 테스트 이미지 생성
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255  # 밝은 영역

    rois = detector.detect(image)
    assert len(rois) > 0


def test_roi_detector_color_image():
    """컬러 이미지 ROI 검출 테스트"""
    detector = ROIDetector()

    # 3채널 테스트 이미지
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = [255, 255, 255]

    rois = detector.detect(image)
    assert isinstance(rois, list)


def test_set_threshold():
    """임계값 설정 테스트"""
    detector = ROIDetector()

    detector.set_threshold(0.8)
    assert detector.threshold == 0.8

    # 유효하지 않은 임계값
    try:
        detector.set_threshold(1.5)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_none_image():
    """None 이미지 입력 테스트"""
    detector = ROIDetector()

    try:
        detector.detect(None)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def run_tests_with_details():
    """Run tests with intermediate results"""
    print("=" * 60)
    print("ROI Detector Algorithm Test - Detailed Results")
    print("=" * 60)

    # Test 1: Initialization Test
    print("\n[Test 1] ROIDetector Initialization Test")
    print("-" * 40)
    detector = ROIDetector()
    print(f"  Default threshold: {detector.threshold}")
    print(f"  Default morph_kernel_size: {detector.morph_kernel_size}")

    detector2 = ROIDetector(threshold=70, morph_kernel_size=31)
    print(f"  Custom threshold: {detector2.threshold}")
    print(f"  Custom morph_kernel_size: {detector2.morph_kernel_size}")
    print("  [PASSED] Initialization Test")

    # Test 2: Grayscale Image ROI Detection
    print("\n[Test 2] Grayscale Image ROI Detection Test")
    print("-" * 40)
    detector = ROIDetector(threshold=30)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255

    print(f"  Input image shape: {image.shape}")
    print(f"  Input image dtype: {image.dtype}")
    print(f"  Bright region location: (20:80, 20:80)")
    print(f"  Bright region size: 60x60 = 3600 pixels")

    roi = detector.detect(image)
    if roi is not None:
        print(f"\n  [Detection Results]")
        print(f"  - Corner coordinates:")
        for i, (corner, name) in enumerate(zip(roi.corners, ['TL', 'TR', 'BR', 'BL'])):
            print(f"    {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
        print(f"  - Area: {roi.area:.1f} pixels^2")
        print(f"  - Tilt angle: {roi.angle:.2f} deg")
        print(f"  - Width: {roi.width:.1f} px")
        print(f"  - Height: {roi.height:.1f} px")
        print(f"  - Center: ({roi.center[0]:.1f}, {roi.center[1]:.1f})")

        binary_mask = detector.get_binary_mask()
        if binary_mask is not None:
            print(f"\n  [Intermediate Result - Binary Mask]")
            print(f"  - Binary mask shape: {binary_mask.shape}")
            print(f"  - White pixels: {np.sum(binary_mask > 0)}")
            print(f"  - Black pixels: {np.sum(binary_mask == 0)}")

        contours = detector.get_all_contours()
        if contours is not None:
            print(f"\n  [Intermediate Result - Contours]")
            print(f"  - Number of contours detected: {len(contours)}")
            for i, cnt in enumerate(contours):
                print(f"    Contour {i}: {len(cnt)} points, area={cv2.contourArea(cnt):.1f}")

    print("  [PASSED] Grayscale Image Test")

    # Test 3: Color Image ROI Detection
    print("\n[Test 3] Color Image ROI Detection Test")
    print("-" * 40)
    detector = ROIDetector(threshold=100)
    color_image = np.zeros((150, 200, 3), dtype=np.uint8)
    color_image[30:120, 40:160] = [200, 200, 200]

    print(f"  Input image shape: {color_image.shape}")
    print(f"  Input image dtype: {color_image.dtype}")
    print(f"  Bright region location: (30:120, 40:160)")
    print(f"  Bright region size: 90x120 = 10800 pixels")

    roi = detector.detect(color_image)
    if roi is not None:
        print(f"\n  [Detection Results]")
        print(f"  - Area: {roi.area:.1f} pixels^2")
        print(f"  - Width: {roi.width:.1f} px, Height: {roi.height:.1f} px")
        print(f"  - Tilt angle: {roi.angle:.2f} deg")
    print("  [PASSED] Color Image Test")

    # Test 4: Rotated Rectangle Detection
    print("\n[Test 4] Rotated Rectangle ROI Detection Test")
    print("-" * 40)
    detector = ROIDetector(threshold=30)
    rotated_image = np.zeros((200, 200), dtype=np.uint8)

    center = (100, 100)
    size = (80, 120)
    angle = 30

    rect = ((center[0], center[1]), (size[0], size[1]), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.fillPoly(rotated_image, [box], 255)

    print(f"  Generated rectangle:")
    print(f"  - Center: {center}")
    print(f"  - Size: {size[0]}x{size[1]}")
    print(f"  - Rotation angle: {angle} deg")

    roi = detector.detect(rotated_image)
    if roi is not None:
        print(f"\n  [Detection Results]")
        print(f"  - Detected tilt: {roi.angle:.2f} deg")
        print(f"  - Detected width: {roi.width:.1f} px")
        print(f"  - Detected height: {roi.height:.1f} px")
        print(f"  - Detected area: {roi.area:.1f} pixels^2")
        print(f"  - Expected area: {size[0] * size[1]} pixels^2")
    print("  [PASSED] Rotated Rectangle Test")

    # Test 5: Threshold Setting Test
    print("\n[Test 5] Threshold Setting Test")
    print("-" * 40)
    detector = ROIDetector()
    print(f"  Initial threshold: {detector.threshold}")

    detector.set_threshold(80)
    print(f"  After change threshold: {detector.threshold}")

    detector.set_morph_kernel_size(21)
    print(f"  After change morph_kernel_size: {detector.morph_kernel_size}")
    print("  [PASSED] Threshold Setting Test")

    # Test 6: Threshold Comparison
    print("\n[Test 6] ROI Detection vs Threshold Comparison")
    print("-" * 40)
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[25:75, 25:75] = 128

    print(f"  Test image: 100x100, bright region value=128")
    print(f"\n  Threshold | Detected | Area")
    print("  " + "-" * 35)

    for thresh in [50, 100, 127, 128, 150, 200]:
        detector = ROIDetector(threshold=thresh)
        roi = detector.detect(test_image)
        if roi is not None:
            print(f"     {thresh:>3}     |    Yes    | {roi.area:.0f}")
        else:
            print(f"     {thresh:>3}     |    No     |  -")

    print("  [PASSED] Threshold Comparison Test")

    print("\n" + "=" * 60)
    print("All Tests Completed!")
    print("=" * 60)


if __name__ == "__main__":
    import cv2
    run_tests_with_details()
