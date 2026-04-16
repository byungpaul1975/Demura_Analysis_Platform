# -*- coding: utf-8 -*-
"""
Image Cropper Module
Handles cropping of images with specified coordinates.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CropRegion:
    """Crop region definition"""
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def size(self) -> Tuple[int, int]:
        """Returns (width, height)"""
        return (self.width, self.height)


class ImageCropper:
    """Image cropping class with configurable region"""

    def __init__(self, crop_region: Optional[CropRegion] = None):
        """
        Initialize ImageCropper.

        Args:
            crop_region: CropRegion object defining the crop area
        """
        self.crop_region = crop_region

    def set_region(self, x_start: int, y_start: int, x_end: int, y_end: int) -> None:
        """
        Set crop region using coordinates.

        Args:
            x_start: Starting x coordinate
            y_start: Starting y coordinate
            x_end: Ending x coordinate
            y_end: Ending y coordinate
        """
        self.crop_region = CropRegion(x_start, y_start, x_end, y_end)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop the image using the defined region.

        Args:
            image: Input image array (H, W) or (H, W, C)

        Returns:
            Cropped image array
        """
        if self.crop_region is None:
            return image

        r = self.crop_region

        # Validate bounds
        h, w = image.shape[:2]
        x_start = max(0, min(r.x_start, w))
        x_end = max(0, min(r.x_end, w))
        y_start = max(0, min(r.y_start, h))
        y_end = max(0, min(r.y_end, h))

        # In numpy/cv2, array indexing is [row, col] = [y, x]
        return image[y_start:y_end, x_start:x_end].copy()

    def get_info(self) -> dict:
        """Get crop region information"""
        if self.crop_region is None:
            return {"status": "No crop region defined"}

        r = self.crop_region
        return {
            "x_start": r.x_start,
            "y_start": r.y_start,
            "x_end": r.x_end,
            "y_end": r.y_end,
            "width": r.width,
            "height": r.height
        }
