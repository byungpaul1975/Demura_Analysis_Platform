# -*- coding: utf-8 -*-
"""
Image Normalizer Module
Handles image normalization to various bit depths.
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NormalizationResult:
    """Result of normalization operation"""
    image: np.ndarray
    original_min: float
    original_max: float
    normalized_min: int
    normalized_max: int
    bit_depth: int


class ImageNormalizer:
    """
    Image normalization class for converting to various bit depths.
    Supports 8-bit, 16-bit, and custom range normalization.
    """

    def __init__(self, bit_depth: int = 16):
        """
        Initialize ImageNormalizer.

        Args:
            bit_depth: Target bit depth (8 or 16)
        """
        self.bit_depth = bit_depth
        self._max_value = (2 ** bit_depth) - 1

    def normalize(self, image: np.ndarray,
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None) -> NormalizationResult:
        """
        Normalize image to specified bit depth.

        Args:
            image: Input image array
            min_val: Optional minimum value for normalization (default: image min)
            max_val: Optional maximum value for normalization (default: image max)

        Returns:
            NormalizationResult containing normalized image and metadata
        """
        # Determine min/max values
        original_min = float(image.min()) if min_val is None else min_val
        original_max = float(image.max()) if max_val is None else max_val

        # Avoid division by zero
        if original_max - original_min == 0:
            normalized = np.zeros_like(image)
        else:
            # Normalize to [0, max_value] range
            normalized = (image - original_min) / (original_max - original_min) * self._max_value

        # Convert to appropriate dtype
        if self.bit_depth == 8:
            normalized = normalized.astype(np.uint8)
        elif self.bit_depth == 16:
            normalized = normalized.astype(np.uint16)
        else:
            normalized = normalized.astype(np.uint32)

        return NormalizationResult(
            image=normalized,
            original_min=original_min,
            original_max=original_max,
            normalized_min=int(normalized.min()),
            normalized_max=int(normalized.max()),
            bit_depth=self.bit_depth
        )

    def normalize_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """
        Quick normalization to 8-bit.

        Args:
            image: Input image array

        Returns:
            8-bit normalized image
        """
        original_bit_depth = self.bit_depth
        self.bit_depth = 8
        self._max_value = 255

        result = self.normalize(image)

        self.bit_depth = original_bit_depth
        self._max_value = (2 ** original_bit_depth) - 1

        return result.image

    def normalize_to_16bit(self, image: np.ndarray) -> np.ndarray:
        """
        Quick normalization to 16-bit.

        Args:
            image: Input image array

        Returns:
            16-bit normalized image
        """
        original_bit_depth = self.bit_depth
        self.bit_depth = 16
        self._max_value = 65535

        result = self.normalize(image)

        self.bit_depth = original_bit_depth
        self._max_value = (2 ** original_bit_depth) - 1

        return result.image

    def set_bit_depth(self, bit_depth: int) -> None:
        """Set target bit depth"""
        self.bit_depth = bit_depth
        self._max_value = (2 ** bit_depth) - 1

    @staticmethod
    def convert_16bit_to_8bit(image: np.ndarray) -> np.ndarray:
        """
        Convert 16-bit image to 8-bit.

        Args:
            image: 16-bit input image

        Returns:
            8-bit image
        """
        return (image / 256).astype(np.uint8)

    @staticmethod
    def convert_8bit_to_16bit(image: np.ndarray) -> np.ndarray:
        """
        Convert 8-bit image to 16-bit.

        Args:
            image: 8-bit input image

        Returns:
            16-bit image
        """
        return (image.astype(np.uint16) * 256)
