"""
유틸리티 함수 모듈
"""
import numpy as np
from typing import Tuple, Optional


def load_image(filepath: str) -> np.ndarray:
    """이미지 파일 로드"""
    try:
        import cv2
        image = cv2.imread(filepath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except ImportError:
        from PIL import Image
        return np.array(Image.open(filepath))


def save_image(image: np.ndarray, filepath: str) -> bool:
    """이미지 파일 저장"""
    try:
        import cv2
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(filepath, image)
    except ImportError:
        from PIL import Image
        Image.fromarray(image).save(filepath)
        return True


def normalize_image(image: np.ndarray) -> np.ndarray:
    """이미지 정규화 (0-1 범위)"""
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image - img_min) / (img_max - img_min)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """이미지 리사이즈"""
    try:
        import cv2
        return cv2.resize(image, size)
    except ImportError:
        from PIL import Image
        return np.array(Image.fromarray(image).resize(size))


def draw_roi(image: np.ndarray, roi: Tuple[int, int, int, int],
             color: Tuple[int, int, int] = (255, 0, 0),
             thickness: int = 2) -> np.ndarray:
    """이미지에 ROI 사각형 그리기"""
    result = image.copy()
    x, y, w, h = roi

    try:
        import cv2
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    except ImportError:
        # numpy로 직접 그리기
        result[y:y+thickness, x:x+w] = color
        result[y+h-thickness:y+h, x:x+w] = color
        result[y:y+h, x:x+thickness] = color
        result[y:y+h, x+w-thickness:x+w] = color

    return result
