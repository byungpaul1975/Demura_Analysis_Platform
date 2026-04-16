# -*- coding: utf-8 -*-
"""
Perspective Warper Module
Handles perspective transformation to correct tilted images.
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class WarpResult:
    """Result of perspective warp operation"""
    image: np.ndarray
    transform_matrix: np.ndarray
    src_corners: np.ndarray
    dst_corners: np.ndarray
    width: int
    height: int


class PerspectiveWarper:
    """
    Perspective transformation class for correcting tilted display images.
    Uses INTER_NEAREST interpolation to preserve original pixel values.
    """

    def __init__(self, interpolation: int = cv2.INTER_NEAREST):
        """
        Initialize PerspectiveWarper.

        Args:
            interpolation: OpenCV interpolation flag (default: INTER_NEAREST to preserve pixel values)
        """
        self.interpolation = interpolation
        self._last_matrix = None

    def warp(self, image: np.ndarray, src_corners: np.ndarray,
             dst_size: Optional[Tuple[int, int]] = None) -> WarpResult:
        """
        Apply perspective warp to correct image tilt.

        Args:
            image: Input image array
            src_corners: 4 source corners ordered as TL, TR, BR, BL (shape: 4x2)
            dst_size: Optional (width, height) tuple. If None, calculated from src_corners.

        Returns:
            WarpResult containing warped image and transformation details
        """
        # Calculate destination size from corners if not provided
        if dst_size is None:
            width_top = np.linalg.norm(src_corners[1] - src_corners[0])
            width_bottom = np.linalg.norm(src_corners[2] - src_corners[3])
            height_left = np.linalg.norm(src_corners[3] - src_corners[0])
            height_right = np.linalg.norm(src_corners[2] - src_corners[1])

            dst_width = int(max(width_top, width_bottom))
            dst_height = int(max(height_left, height_right))
        else:
            dst_width, dst_height = dst_size

        # Define destination corners
        dst_corners = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype=np.float32)

        # Calculate transformation matrix
        M = cv2.getPerspectiveTransform(src_corners.astype(np.float32), dst_corners)
        self._last_matrix = M

        # Apply warp with specified interpolation (INTER_NEAREST preserves pixel values)
        warped = cv2.warpPerspective(image, M, (dst_width, dst_height),
                                     flags=self.interpolation)

        return WarpResult(
            image=warped,
            transform_matrix=M,
            src_corners=src_corners,
            dst_corners=dst_corners,
            width=dst_width,
            height=dst_height
        )

    def inverse_warp(self, image: np.ndarray, dst_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply inverse perspective transformation.

        Args:
            image: Warped image to inverse transform
            dst_size: Size of the output image (width, height)

        Returns:
            Inverse warped image
        """
        if self._last_matrix is None:
            raise ValueError("No transformation matrix available. Call warp() first.")

        M_inv = np.linalg.inv(self._last_matrix)
        return cv2.warpPerspective(image, M_inv, dst_size, flags=self.interpolation)

    def set_interpolation(self, interpolation: int) -> None:
        """
        Set interpolation method.

        Args:
            interpolation: OpenCV interpolation flag
                - cv2.INTER_NEAREST: Nearest neighbor (preserves pixel values)
                - cv2.INTER_LINEAR: Bilinear interpolation
                - cv2.INTER_CUBIC: Bicubic interpolation
        """
        self.interpolation = interpolation

    def get_transform_matrix(self) -> Optional[np.ndarray]:
        """Get the last computed transformation matrix"""
        return self._last_matrix
