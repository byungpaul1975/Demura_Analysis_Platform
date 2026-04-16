# -*- coding: utf-8 -*-
"""
ROI (Region of Interest) Detector Module
Detects display panel boundaries and extracts corner points.
Supports both global and per-corner adaptive thresholding.
"""
import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class CornerMetrics:
    """Metrics for a single corner"""
    position: Tuple[float, float]
    optimal_threshold: int
    gradient_max: float
    edge_sharpness: float
    edge_contrast: float
    confidence: float


@dataclass
class DetectedROI:
    """Detected ROI information"""
    corners: np.ndarray  # 4 corners: TL, TR, BR, BL
    contour: np.ndarray
    area: float
    angle: float  # Tilt angle in degrees
    width: float  # Width in pixels
    height: float  # Height in pixels
    corner_metrics: Optional[Dict[str, CornerMetrics]] = field(default=None)

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of ROI"""
        return (self.corners[:, 0].mean(), self.corners[:, 1].mean())


class ROIDetector:
    """
    Region of Interest detector for display panels.
    Detects the display area in camera images and extracts corner points.
    """

    def __init__(self, threshold: int = 460, morph_kernel_size: int = 51):
        """
        Initialize ROI Detector.

        Args:
            threshold: Pixel value threshold to separate display from background
            morph_kernel_size: Size of morphological kernel for noise removal
        """
        self.threshold = threshold
        self.morph_kernel_size = morph_kernel_size
        self._last_binary = None
        self._last_contours = None

    def detect(self, image: np.ndarray) -> Optional[DetectedROI]:
        """
        Detect ROI (display region) in the image.

        Args:
            image: Input image (H, W) grayscale or (H, W, C) color

        Returns:
            DetectedROI object containing corners and metadata, or None if not found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Binarize
        binary = (gray > self.threshold).astype(np.uint8) * 255

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.morph_kernel_size, self.morph_kernel_size))
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel)

        # Fill holes
        binary_filled = ndimage.binary_fill_holes(binary_cleaned).astype(np.uint8) * 255
        self._last_binary = binary_filled

        # Find contours
        contours, _ = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._last_contours = contours

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        corners = cv2.boxPoints(rect)

        # Order corners: TL, TR, BR, BL
        ordered_corners = self._order_corners(corners.astype(np.float32))

        # Calculate tilt angle
        top_edge = ordered_corners[1] - ordered_corners[0]
        angle = np.degrees(np.arctan2(top_edge[1], top_edge[0]))

        # Calculate dimensions
        width = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
        height = np.linalg.norm(ordered_corners[3] - ordered_corners[0])

        return DetectedROI(
            corners=ordered_corners,
            contour=largest_contour,
            area=area,
            angle=angle,
            width=width,
            height=height
        )

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.

        Args:
            pts: Array of 4 corner points

        Returns:
            Ordered corners array
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left

        return rect

    def set_threshold(self, threshold: int) -> None:
        """Set detection threshold"""
        self.threshold = threshold

    def set_morph_kernel_size(self, size: int) -> None:
        """Set morphological kernel size"""
        self.morph_kernel_size = size

    def get_binary_mask(self) -> Optional[np.ndarray]:
        """Get the last computed binary mask"""
        return self._last_binary

    def get_all_contours(self) -> Optional[List[np.ndarray]]:
        """Get all detected contours"""
        return self._last_contours


class AdaptiveROIDetector(ROIDetector):
    """
    Adaptive ROI detector with per-corner threshold optimization.
    Analyzes local gradient at each corner to find optimal thresholds.
    Supports display pixel grid alignment for accurate edge detection.
    """

    CORNER_NAMES = ['TL', 'TR', 'BR', 'BL']
    EDGE_NAMES = ['Top', 'Right', 'Bottom', 'Left']

    def __init__(
        self,
        initial_threshold: int = 200,
        morph_kernel_size: int = 51,
        corner_region_size: int = 60,
        threshold_range: Tuple[int, int] = (50, 600),
        threshold_step: int = 20,
        display_pixel_pitch: Tuple[float, float] = (3.5, 3.5),
        align_to_display_grid: bool = True
    ):
        """
        Initialize Adaptive ROI Detector.

        Args:
            initial_threshold: Initial threshold for ROI detection
            morph_kernel_size: Size of morphological kernel
            corner_region_size: Size of region around each corner for analysis
            threshold_range: (min, max) threshold values to test
            threshold_step: Step size for threshold search
            display_pixel_pitch: (horizontal, vertical) camera pixels per display pixel
                                 Range: 3.0 to 4.0 for typical display panels
            align_to_display_grid: If True, align ROI edges to display pixel boundaries
        """
        super().__init__(threshold=initial_threshold, morph_kernel_size=morph_kernel_size)
        self.corner_region_size = corner_region_size
        self.threshold_range = threshold_range
        self.threshold_step = threshold_step
        self.display_pixel_pitch = display_pixel_pitch
        self.align_to_display_grid = align_to_display_grid
        self._corner_thresholds: Dict[str, int] = {}
        self._corner_metrics: Dict[str, CornerMetrics] = {}
        self._detected_pitch: Optional[Tuple[float, float]] = None

    def detect_adaptive(self, image: np.ndarray) -> Optional[DetectedROI]:
        """
        Detect ROI using per-corner adaptive thresholding.

        Args:
            image: Input image (grayscale or color)

        Returns:
            DetectedROI with optimized corners and metrics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: Initial detection with base threshold
        initial_roi = self.detect(image)
        if initial_roi is None:
            return None

        # Step 2: Optimize threshold for each corner
        optimized_corners = initial_roi.corners.copy()
        self._corner_metrics = {}

        for idx, corner_name in enumerate(self.CORNER_NAMES):
            corner = initial_roi.corners[idx]
            metrics = self._optimize_corner_threshold(gray, corner, corner_name)
            self._corner_metrics[corner_name] = metrics
            self._corner_thresholds[corner_name] = metrics.optimal_threshold

            # Step 3: Refine corner position using optimal threshold
            refined_corner = self._refine_corner_position(
                gray, corner, metrics.optimal_threshold
            )
            optimized_corners[idx] = refined_corner

        # Recalculate ROI properties with refined corners
        top_edge = optimized_corners[1] - optimized_corners[0]
        angle = np.degrees(np.arctan2(top_edge[1], top_edge[0]))
        width = np.linalg.norm(optimized_corners[1] - optimized_corners[0])
        height = np.linalg.norm(optimized_corners[3] - optimized_corners[0])

        # Calculate area using Shoelace formula
        n = len(optimized_corners)
        area = 0.5 * abs(sum(
            optimized_corners[i][0] * optimized_corners[(i + 1) % n][1] -
            optimized_corners[(i + 1) % n][0] * optimized_corners[i][1]
            for i in range(n)
        ))

        return DetectedROI(
            corners=optimized_corners,
            contour=initial_roi.contour,
            area=area,
            angle=angle,
            width=width,
            height=height,
            corner_metrics=self._corner_metrics
        )

    def _optimize_corner_threshold(
        self,
        gray: np.ndarray,
        corner: np.ndarray,
        corner_name: str
    ) -> CornerMetrics:
        """
        Find optimal threshold for a specific corner.

        Args:
            gray: Grayscale image
            corner: Corner position (x, y)
            corner_name: Name of corner (TL, TR, BR, BL)

        Returns:
            CornerMetrics with optimal threshold and quality metrics
        """
        cx, cy = int(corner[0]), int(corner[1])
        half = self.corner_region_size // 2

        # Extract region around corner
        y1, y2 = max(0, cy - half), min(gray.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(gray.shape[1], cx + half)
        region = gray[y1:y2, x1:x2].astype(np.float32)

        if region.size == 0:
            return CornerMetrics(
                position=(cx, cy),
                optimal_threshold=self.threshold,
                gradient_max=0, edge_sharpness=0, edge_contrast=0, confidence=0
            )

        # Calculate gradient once
        sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        best_score = -1
        best_metrics = None
        best_threshold = self.threshold

        # Test different thresholds
        for thresh in range(self.threshold_range[0], self.threshold_range[1], self.threshold_step):
            binary = (region > thresh).astype(np.uint8) * 255
            binary_edge = cv2.Canny(binary, 50, 150)
            edge_mask = binary_edge > 0

            if np.sum(edge_mask) == 0:
                continue

            # Edge sharpness: gradient at edge locations
            edge_gradient = gradient_mag[edge_mask]
            edge_sharpness = np.percentile(edge_gradient, 90)

            # Edge contrast: difference between ROI and background
            roi_mask = binary > 0
            bg_mask = binary == 0

            if np.sum(roi_mask) > 0 and np.sum(bg_mask) > 0:
                roi_mean = np.mean(region[roi_mask])
                bg_mean = np.mean(region[bg_mask])
                edge_contrast = roi_mean - bg_mean
            else:
                edge_contrast = 0

            gradient_max = gradient_mag.max()

            # Normalize and combine scores
            def safe_normalize(val, max_val):
                return val / max_val if max_val > 0 else 0

            score = (
                0.30 * safe_normalize(gradient_max, 3000) +
                0.40 * safe_normalize(edge_sharpness, 2000) +
                0.30 * safe_normalize(edge_contrast, 1000)
            )

            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = {
                    'gradient_max': gradient_max,
                    'edge_sharpness': edge_sharpness,
                    'edge_contrast': edge_contrast
                }

        if best_metrics is None:
            best_metrics = {'gradient_max': 0, 'edge_sharpness': 0, 'edge_contrast': 0}

        return CornerMetrics(
            position=(cx, cy),
            optimal_threshold=best_threshold,
            gradient_max=best_metrics['gradient_max'],
            edge_sharpness=best_metrics['edge_sharpness'],
            edge_contrast=best_metrics['edge_contrast'],
            confidence=best_score
        )

    def _refine_corner_position(
        self,
        gray: np.ndarray,
        corner: np.ndarray,
        optimal_threshold: int
    ) -> np.ndarray:
        """
        Refine corner position using optimal threshold.

        Args:
            gray: Grayscale image
            corner: Initial corner position
            optimal_threshold: Optimized threshold for this corner

        Returns:
            Refined corner position
        """
        cx, cy = int(corner[0]), int(corner[1])
        half = self.corner_region_size // 2

        # Extract region
        y1, y2 = max(0, cy - half), min(gray.shape[0], cy + half)
        x1, x2 = max(0, cx - half), min(gray.shape[1], cx + half)
        region = gray[y1:y2, x1:x2]

        if region.size == 0:
            return corner

        # Binary threshold with optimal value
        binary = (region > optimal_threshold).astype(np.uint8) * 255

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours in region
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return corner

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 100:
            return corner

        # Get corner of the contour closest to the region center
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)

        # Find the point closest to the original corner (relative to region)
        rel_corner = np.array([cx - x1, cy - y1])
        distances = np.linalg.norm(box - rel_corner, axis=1)
        closest_idx = np.argmin(distances)
        refined_local = box[closest_idx]

        # Convert back to image coordinates
        refined = np.array([refined_local[0] + x1, refined_local[1] + y1])

        return refined

    def get_corner_thresholds(self) -> Dict[str, int]:
        """Get optimized thresholds for each corner"""
        return self._corner_thresholds.copy()

    def get_corner_metrics(self) -> Dict[str, CornerMetrics]:
        """Get detailed metrics for each corner"""
        return self._corner_metrics.copy()

    def get_average_threshold(self) -> float:
        """Get average threshold across all corners"""
        if not self._corner_thresholds:
            return float(self.threshold)
        return np.mean(list(self._corner_thresholds.values()))

    def set_display_pixel_pitch(self, horizontal: float, vertical: float) -> None:
        """
        Set display pixel pitch (camera pixels per display pixel).

        Args:
            horizontal: Horizontal pitch (typically 3.0 to 4.0)
            vertical: Vertical pitch (typically 3.0 to 4.0)
        """
        self.display_pixel_pitch = (horizontal, vertical)

    def align_to_display_pixel_grid(
        self,
        roi: DetectedROI,
        reference_corner: str = 'TL'
    ) -> DetectedROI:
        """
        Align ROI edges to display pixel grid boundaries.

        The display pixel grid is defined by the display_pixel_pitch parameter.
        This ensures that ROI edges fall exactly on display pixel boundaries,
        which is critical for accurate display pixel extraction.

        Args:
            roi: Detected ROI to align
            reference_corner: Corner to use as grid reference ('TL', 'TR', 'BR', 'BL')

        Returns:
            New DetectedROI with grid-aligned corners
        """
        pitch_x, pitch_y = self.display_pixel_pitch
        corners = roi.corners.copy()

        # Use reference corner as grid origin
        ref_idx = self.CORNER_NAMES.index(reference_corner)
        ref_corner = corners[ref_idx].copy()

        # Align reference corner to nearest grid point
        aligned_ref = self._snap_to_grid(ref_corner, pitch_x, pitch_y)

        # Calculate offset from original reference corner
        offset = aligned_ref - ref_corner

        # Calculate grid-aligned dimensions
        # Width and height should be integer multiples of pitch
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])

        # Round to nearest display pixel
        display_width = round(width / pitch_x)
        display_height = round(height / pitch_y)

        # Calculate new aligned width/height in camera pixels
        aligned_width = display_width * pitch_x
        aligned_height = display_height * pitch_y

        # Reconstruct corners based on reference corner and aligned dimensions
        aligned_corners = self._reconstruct_corners(
            aligned_ref, aligned_width, aligned_height, roi.angle, reference_corner
        )

        # Calculate new area
        n = len(aligned_corners)
        area = 0.5 * abs(sum(
            aligned_corners[i][0] * aligned_corners[(i + 1) % n][1] -
            aligned_corners[(i + 1) % n][0] * aligned_corners[i][1]
            for i in range(n)
        ))

        return DetectedROI(
            corners=aligned_corners,
            contour=roi.contour,
            area=area,
            angle=roi.angle,
            width=aligned_width,
            height=aligned_height,
            corner_metrics=roi.corner_metrics
        )

    def _snap_to_grid(self, point: np.ndarray, pitch_x: float, pitch_y: float) -> np.ndarray:
        """
        Snap a point to the nearest display pixel grid intersection.

        Args:
            point: (x, y) coordinates
            pitch_x: Horizontal pitch
            pitch_y: Vertical pitch

        Returns:
            Grid-aligned point
        """
        aligned_x = round(point[0] / pitch_x) * pitch_x
        aligned_y = round(point[1] / pitch_y) * pitch_y
        return np.array([aligned_x, aligned_y], dtype=np.float32)

    def _reconstruct_corners(
        self,
        ref_corner: np.ndarray,
        width: float,
        height: float,
        angle: float,
        reference: str
    ) -> np.ndarray:
        """
        Reconstruct all 4 corners from reference corner, dimensions, and angle.

        Args:
            ref_corner: Reference corner position
            width: ROI width in camera pixels
            height: ROI height in camera pixels
            angle: Rotation angle in degrees
            reference: Which corner is the reference ('TL', 'TR', 'BR', 'BL')

        Returns:
            Array of 4 corners (TL, TR, BR, BL)
        """
        # Convert angle to radians
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        # Direction vectors
        right = np.array([cos_a, sin_a]) * width
        down = np.array([-sin_a, cos_a]) * height

        corners = np.zeros((4, 2), dtype=np.float32)

        if reference == 'TL':
            corners[0] = ref_corner  # TL
            corners[1] = ref_corner + right  # TR
            corners[2] = ref_corner + right + down  # BR
            corners[3] = ref_corner + down  # BL
        elif reference == 'TR':
            corners[1] = ref_corner  # TR
            corners[0] = ref_corner - right  # TL
            corners[2] = ref_corner + down  # BR
            corners[3] = ref_corner - right + down  # BL
        elif reference == 'BR':
            corners[2] = ref_corner  # BR
            corners[1] = ref_corner - down  # TR
            corners[0] = ref_corner - right - down  # TL
            corners[3] = ref_corner - right  # BL
        elif reference == 'BL':
            corners[3] = ref_corner  # BL
            corners[0] = ref_corner - down  # TL
            corners[1] = ref_corner + right - down  # TR
            corners[2] = ref_corner + right  # BR

        return corners

    def detect_with_grid_alignment(self, image: np.ndarray) -> Optional[DetectedROI]:
        """
        Detect ROI with adaptive thresholding and display pixel grid alignment.

        This is the main method for production use. It combines:
        1. Per-corner adaptive threshold optimization
        2. Corner position refinement
        3. Display pixel grid alignment

        Args:
            image: Input image (grayscale or color)

        Returns:
            Grid-aligned DetectedROI
        """
        # Step 1: Adaptive detection
        roi = self.detect_adaptive(image)
        if roi is None:
            return None

        # Step 2: Align to display pixel grid if enabled
        if self.align_to_display_grid:
            roi = self.align_to_display_pixel_grid(roi, reference_corner='TL')

        return roi

    def get_display_pixel_info(self, roi: DetectedROI) -> Dict:
        """
        Get display pixel information from ROI.

        Args:
            roi: Detected ROI

        Returns:
            Dictionary with display pixel information
        """
        pitch_x, pitch_y = self.display_pixel_pitch

        return {
            'display_width_pixels': round(roi.width / pitch_x),
            'display_height_pixels': round(roi.height / pitch_y),
            'camera_width_pixels': roi.width,
            'camera_height_pixels': roi.height,
            'horizontal_pitch': pitch_x,
            'vertical_pitch': pitch_y,
            'total_display_pixels': round(roi.width / pitch_x) * round(roi.height / pitch_y)
        }

    def detect_by_crossing(
        self,
        image: np.ndarray,
        edge_offset: int = 2,
        search_radius: int = 30
    ) -> Optional[DetectedROI]:
        """
        Detect ROI using X/Y crossing point method.

        Algorithm for each corner:
        1. Find outermost X edge of display (leftmost or rightmost bright pixel)
        2. Find outermost Y edge of display (topmost or bottommost bright pixel)
        3. Cross point = where these X and Y edges intersect
        4. ROI edge = cross point + edge_offset pixels diagonally outward

        Args:
            image: Input image (grayscale or color)
            edge_offset: Pixels to move outward from cross point (default: 2)
            search_radius: Search radius around approximate corners (default: 30)

        Returns:
            DetectedROI with corners based on X/Y cross points
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: Get approximate corners using global detection
        approx_corners = self._get_approximate_corners(gray)
        if approx_corners is None:
            return None

        # Step 2: Find precise edge points using X/Y crossing for each corner
        roi_corners = np.zeros((4, 2), dtype=np.float32)
        cross_points = {}
        edge_infos = {}

        for idx, corner_name in enumerate(self.CORNER_NAMES):
            approx_corner = approx_corners[idx]
            roi_edge, cross_point, edge_info = self._find_edge_crossing_point(
                gray, approx_corner, corner_name, edge_offset, search_radius
            )
            roi_corners[idx] = roi_edge
            cross_points[corner_name] = cross_point
            edge_infos[corner_name] = edge_info

        # Calculate ROI properties
        top_edge = roi_corners[1] - roi_corners[0]
        angle = np.degrees(np.arctan2(top_edge[1], top_edge[0]))
        width = np.linalg.norm(roi_corners[1] - roi_corners[0])
        height = np.linalg.norm(roi_corners[3] - roi_corners[0])

        # Calculate area using Shoelace formula
        n = len(roi_corners)
        area = 0.5 * abs(sum(
            roi_corners[i][0] * roi_corners[(i + 1) % n][1] -
            roi_corners[(i + 1) % n][0] * roi_corners[i][1]
            for i in range(n)
        ))

        # Store cross points and edge info for later access
        self._cross_points = cross_points
        self._edge_infos = edge_infos

        return DetectedROI(
            corners=roi_corners,
            contour=np.array([]),
            area=area,
            angle=angle,
            width=width,
            height=height,
            corner_metrics=None
        )

    def _get_approximate_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Get approximate corner positions using Otsu thresholding.

        Args:
            gray: Grayscale image

        Returns:
            Ordered corners (TL, TR, BR, BL) or None if not found
        """
        # Normalize to 8-bit for Otsu
        gray_norm = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
        _, binary = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)

        return self._order_corners(box)

    def _find_edge_crossing_point(
        self,
        image: np.ndarray,
        approx_corner: np.ndarray,
        corner_name: str,
        offset: int = 2,
        search_radius: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find ROI edge point using X/Y crossing method.

        Algorithm:
        1. Extract region around approximate corner
        2. Find all bright pixels (display area)
        3. Find outermost X and Y based on corner type
        4. Cross point = (edge_x, edge_y)
        5. ROI edge = cross point + offset diagonally outward

        Args:
            image: Grayscale image
            approx_corner: Approximate corner position (x, y)
            corner_name: Corner name ('TL', 'TR', 'BR', 'BL')
            offset: Pixels to move outward from cross point
            search_radius: Search radius around approximate corner

        Returns:
            Tuple of (roi_edge, cross_point, edge_info)
        """
        cx, cy = int(approx_corner[0]), int(approx_corner[1])

        # Define search region
        y1 = max(0, cy - search_radius)
        y2 = min(image.shape[0], cy + search_radius)
        x1 = max(0, cx - search_radius)
        x2 = min(image.shape[1], cx + search_radius)

        region = image[y1:y2, x1:x2].astype(np.float32)

        # Normalize and apply Otsu threshold
        region_norm = ((region - region.min()) / (region.max() - region.min()) * 255).astype(np.uint8)
        _, binary = cv2.threshold(region_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find all bright pixels
        bright_pixels = np.where(binary > 0)

        if len(bright_pixels[0]) == 0:
            # No bright pixels found, return approximate corner
            return (
                np.array([cx, cy], dtype=np.float32),
                np.array([cx, cy], dtype=np.float32),
                {'edge_x': 0, 'edge_y': 0, 'x1': x1, 'y1': y1}
            )

        bright_y = bright_pixels[0]  # row indices
        bright_x = bright_pixels[1]  # column indices

        # Find outermost X and Y based on corner type
        if corner_name == 'TL':
            edge_x = np.min(bright_x)  # leftmost
            edge_y = np.min(bright_y)  # topmost
            cross_local = np.array([edge_x, edge_y])
            roi_edge_local = cross_local - np.array([offset, offset])

        elif corner_name == 'TR':
            edge_x = np.max(bright_x)  # rightmost
            edge_y = np.min(bright_y)  # topmost
            cross_local = np.array([edge_x, edge_y])
            roi_edge_local = cross_local + np.array([offset, -offset])

        elif corner_name == 'BR':
            edge_x = np.max(bright_x)  # rightmost
            edge_y = np.max(bright_y)  # bottommost
            cross_local = np.array([edge_x, edge_y])
            roi_edge_local = cross_local + np.array([offset, offset])

        elif corner_name == 'BL':
            edge_x = np.min(bright_x)  # leftmost
            edge_y = np.max(bright_y)  # bottommost
            cross_local = np.array([edge_x, edge_y])
            roi_edge_local = cross_local + np.array([-offset, offset])

        else:
            raise ValueError(f"Unknown corner name: {corner_name}")

        # Convert to global coordinates
        cross_global = np.array([cross_local[0] + x1, cross_local[1] + y1], dtype=np.float32)
        roi_edge_global = np.array([roi_edge_local[0] + x1, roi_edge_local[1] + y1], dtype=np.float32)

        edge_info = {
            'edge_x': int(edge_x),
            'edge_y': int(edge_y),
            'x1': x1,
            'y1': y1,
            'region_shape': region.shape
        }

        return roi_edge_global, cross_global, edge_info

    def get_cross_points(self) -> Dict[str, np.ndarray]:
        """
        Get the cross points (star markers) from the last detect_by_crossing() call.

        Returns:
            Dictionary mapping corner names to cross point coordinates
        """
        return getattr(self, '_cross_points', {})

    def get_edge_infos(self) -> Dict[str, Dict]:
        """
        Get edge information from the last detect_by_crossing() call.

        Returns:
            Dictionary mapping corner names to edge info dictionaries
        """
        return getattr(self, '_edge_infos', {})
