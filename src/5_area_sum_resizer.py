# -*- coding: utf-8 -*-
"""
Area Sum Resizer Module
Resizes images by summing all camera pixels corresponding to each display pixel.
Handles fractional boundaries with proportional weighting.
"""
import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Display pixel mode will be slower.")


@dataclass
class ResizeResult:
    """Result of resize operation"""
    image: np.ndarray
    scale_x: float
    scale_y: float
    pixels_per_output: float  # Average number of input pixels summed per output pixel


# Numba-optimized core function for strict 3x3 grid tracking (0 or 1 pixel gap)
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _find_brightest_in_range(img_float, y_start, y_end, x_start, x_end, h, w):
        """Find brightest pixel within given range."""
        best_val = -1.0
        best_y = (y_start + y_end) // 2
        best_x = (x_start + x_end) // 2

        for sy in range(max(0, y_start), min(h, y_end + 1)):
            for sx in range(max(0, x_start), min(w, x_end + 1)):
                val = img_float[sy, sx]
                if val > best_val:
                    best_val = val
                    best_y = sy
                    best_x = sx

        return best_x, best_y, best_val

    @jit(nopython=True, cache=True)
    def _find_brightest_in_small_region(img_float, cy, cx, h, w):
        """Find brightest pixel in 3x3 region centered at (cx, cy)."""
        best_val = -1.0
        best_y = cy
        best_x = cx

        for sy in range(max(0, cy - 1), min(h, cy + 2)):
            for sx in range(max(0, cx - 1), min(w, cx + 2)):
                val = img_float[sy, sx]
                if val > best_val:
                    best_val = val
                    best_y = sy
                    best_x = sx

        return best_x, best_y, best_val

    @jit(nopython=True, cache=True)
    def _resize_2d_grid_tracking_core(
        img_float: np.ndarray,
        target_width: int,
        target_height: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D Grid tracking: Track bright pixels with 0 or 1 pixel gap in BOTH directions.

        Algorithm:
        1. First pixel at (0,0): search top-left corner for brightest
        2. First row: track with pitch 3 or 4 horizontally
        3. For each subsequent row:
           - Use ABOVE pixel's position to find this row's position
           - Vertical pitch also 3 or 4
        4. Horizontal tracking also uses 3 or 4 pitch

        This ensures ZIGZAG pattern is captured because each pixel's position
        depends on both its left neighbor AND the pixel above.
        """
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)
        all_centers_x = np.zeros((target_height, target_width), dtype=np.int32)
        all_centers_y = np.zeros((target_height, target_width), dtype=np.int32)

        # First pixel: search in top-left 5x5 region
        cx0, cy0, _ = _find_brightest_in_range(img_float, 0, 5, 0, 5, h, w)
        all_centers_x[0, 0] = cx0
        all_centers_y[0, 0] = cy0

        # First row: track horizontally with pitch 3 or 4
        for i in range(1, target_width):
            prev_x = all_centers_x[0, i - 1]
            prev_y = all_centers_y[0, i - 1]

            # Option 1: pitch = 3
            x1 = prev_x + 3
            cx1, cy1, val1 = _find_brightest_in_small_region(img_float, prev_y, x1, h, w)

            # Option 2: pitch = 4
            x2 = prev_x + 4
            cx2, cy2, val2 = _find_brightest_in_small_region(img_float, prev_y, x2, h, w)

            if val1 >= val2:
                all_centers_x[0, i] = cx1
                all_centers_y[0, i] = cy1
            else:
                all_centers_x[0, i] = cx2
                all_centers_y[0, i] = cy2

        # Subsequent rows: use pixel ABOVE as reference for vertical position
        for j in range(1, target_height):
            for i in range(target_width):
                # Reference: pixel directly above
                above_x = all_centers_x[j - 1, i]
                above_y = all_centers_y[j - 1, i]

                # Vertical options: y + 3 or y + 4
                # Search in small region around expected position

                # Option A: vertical pitch = 3
                yA = above_y + 3
                cxA, cyA, valA = _find_brightest_in_small_region(img_float, yA, above_x, h, w)

                # Option B: vertical pitch = 4
                yB = above_y + 4
                cxB, cyB, valB = _find_brightest_in_small_region(img_float, yB, above_x, h, w)

                # Pick brighter option
                if valA >= valB:
                    base_x = cxA
                    base_y = cyA
                else:
                    base_x = cxB
                    base_y = cyB

                # If not first column, also consider horizontal constraint
                if i > 0:
                    left_x = all_centers_x[j, i - 1]
                    left_y = all_centers_y[j, i - 1]

                    # Horizontal pitch should also be 3 or 4
                    # Option H1: horizontal pitch = 3
                    hx1 = left_x + 3
                    hcx1, hcy1, hval1 = _find_brightest_in_small_region(img_float, left_y, hx1, h, w)

                    # Option H2: horizontal pitch = 4
                    hx2 = left_x + 4
                    hcx2, hcy2, hval2 = _find_brightest_in_small_region(img_float, left_y, hx2, h, w)

                    # Combine: prefer horizontal tracking but stay close to vertical reference
                    if hval1 >= hval2:
                        # Check if h1 is close to vertical reference
                        if abs(hcy1 - base_y) <= 2:
                            base_x = hcx1
                            base_y = hcy1
                    else:
                        if abs(hcy2 - base_y) <= 2:
                            base_x = hcx2
                            base_y = hcy2

                all_centers_x[j, i] = base_x
                all_centers_y[j, i] = base_y

        # Calculate 3x3 MEAN for each center
        for j in range(target_height):
            for i in range(target_width):
                cx = all_centers_x[j, i]
                cy = all_centers_y[j, i]

                y1 = max(0, cy - 1)
                y2 = min(h, cy + 2)
                x1 = max(0, cx - 1)
                x2 = min(w, cx + 2)

                total = 0.0
                count = 0
                for py in range(y1, y2):
                    for px in range(x1, x2):
                        total += img_float[py, px]
                        count += 1

                if count > 0:
                    result[j, i] = total / count

        return result, all_centers_x, all_centers_y

    @jit(nopython=True, cache=True)
    def _strict_3x3_row_tracking(
        img_float: np.ndarray,
        target_width: int,
        row_y_center: int,
        h: int,
        w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track 3x3 boxes with STRICT 0 or 1 pixel gap constraint.

        3x3 box center spacing must be exactly 3 or 4 pixels:
        - 3 pixels = 0 pixel gap (adjacent boxes touch)
        - 4 pixels = 1 pixel gap

        For each position, search at +3 and +4 from previous center,
        pick the one with brighter center pixel.
        """
        centers_x = np.zeros(target_width, dtype=np.int32)
        centers_y = np.zeros(target_width, dtype=np.int32)

        # Vertical search range: ±2 pixels from row center
        y_search = 2
        y_start = max(0, row_y_center - y_search)
        y_end = min(h - 1, row_y_center + y_search)

        # Find first bright pixel at start of row (search 0 to 5)
        first_x, first_y, _ = _find_brightest_in_range(
            img_float, y_start, y_end, 0, 5, h, w
        )
        centers_x[0] = first_x
        centers_y[0] = first_y

        # Track subsequent pixels with strict pitch 3 or 4
        for i in range(1, target_width):
            prev_x = centers_x[i - 1]
            prev_y = centers_y[i - 1]

            # Option 1: pitch = 3 (0 pixel gap)
            x1 = prev_x + 3
            y1_start = max(0, prev_y - 1)
            y1_end = min(h - 1, prev_y + 1)
            cx1, cy1, val1 = _find_brightest_in_range(
                img_float, y1_start, y1_end, x1 - 1, x1 + 1, h, w
            )

            # Option 2: pitch = 4 (1 pixel gap)
            x2 = prev_x + 4
            y2_start = max(0, prev_y - 1)
            y2_end = min(h - 1, prev_y + 1)
            cx2, cy2, val2 = _find_brightest_in_range(
                img_float, y2_start, y2_end, x2 - 1, x2 + 1, h, w
            )

            # Choose the brighter option
            if val1 >= val2:
                centers_x[i] = cx1
                centers_y[i] = cy1
            else:
                centers_x[i] = cx2
                centers_y[i] = cy2

        return centers_x, centers_y

    @jit(nopython=True, cache=True)
    def _resize_strict_3x3_core(
        img_float: np.ndarray,
        target_width: int,
        target_height: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Strict 3x3 extraction with 0 or 1 pixel gap constraint.

        Algorithm:
        1. For each row, track 3x3 boxes with spacing exactly 3 or 4 pixels
        2. Pick brighter center between +3 and +4 options
        3. Row spacing is also 3 or 4 pixels (pick brighter row position)
        4. Each 3x3 → MEAN for output

        This ensures NO pixels are skipped because gap is at most 1 pixel.
        """
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)
        all_centers_x = np.zeros((target_height, target_width), dtype=np.int32)
        all_centers_y = np.zeros((target_height, target_width), dtype=np.int32)

        # First row: find starting y position
        first_y = 2  # Start near top with margin

        for j in range(target_height):
            if j == 0:
                row_y = first_y
            else:
                # Previous row's average y
                prev_row_y = int(np.mean(all_centers_y[j - 1, :]))

                # Option 1: y + 3 (0 gap)
                # Option 2: y + 4 (1 gap)
                # For rows, use simple +3 or +4 based on which has brighter pixels
                opt_y1 = prev_row_y + 3
                opt_y2 = prev_row_y + 4

                # Sample a few x positions to decide
                sample_sum1 = 0.0
                sample_sum2 = 0.0
                sample_count = min(10, target_width)

                for si in range(sample_count):
                    sx = int(si * w / sample_count)
                    if 0 <= opt_y1 < h and 0 <= sx < w:
                        sample_sum1 += img_float[opt_y1, sx]
                    if 0 <= opt_y2 < h and 0 <= sx < w:
                        sample_sum2 += img_float[opt_y2, sx]

                if sample_sum1 >= sample_sum2:
                    row_y = opt_y1
                else:
                    row_y = opt_y2

            # Clamp row_y to valid range
            row_y = max(1, min(h - 2, row_y))

            # Track this row with strict 3x3 spacing
            centers_x, centers_y = _strict_3x3_row_tracking(
                img_float, target_width, row_y, h, w
            )

            all_centers_x[j, :] = centers_x
            all_centers_y[j, :] = centers_y

            # Calculate 3x3 MEAN for each center
            for i in range(target_width):
                cx = centers_x[i]
                cy = centers_y[i]

                # Extract 3x3 and calculate MEAN
                y1 = max(0, cy - 1)
                y2 = min(h, cy + 2)
                x1 = max(0, cx - 1)
                x2 = min(w, cx + 2)

                total = 0.0
                count = 0
                for py in range(y1, y2):
                    for px in range(x1, x2):
                        total += img_float[py, px]
                        count += 1

                if count > 0:
                    result[j, i] = total / count

        return result, all_centers_x, all_centers_y

else:
    _resize_strict_3x3_core = None


# Numba-optimized core function for 3x3 centered on brightest pixel
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _resize_brightest_center_3x3_core(
        img_float: np.ndarray,
        target_width: int,
        target_height: int,
        pitch_x: float,
        pitch_y: float,
        search_half: int
    ) -> np.ndarray:
        """
        Numba JIT optimized: Find brightest SINGLE pixel, then 3x3 centered on it.

        Algorithm:
        1. For each output pixel (i, j), calculate input position using pitch
           - x_center = i * pitch_x
           - y_center = j * pitch_y
        2. Search within ±search_half to find the BRIGHTEST SINGLE PIXEL
        3. Extract 3x3 centered on that brightest pixel
        4. Output = MEAN of 3x3 (9 pixels)

        Note: Input pixels can overlap between different output pixels.
        """
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)

        for j in prange(target_height):
            y_center = j * pitch_y

            for i in range(target_width):
                x_center = i * pitch_x

                # Define search region around pitch-based center
                y_start = max(1, int(y_center) - search_half)
                y_end = min(h - 2, int(y_center) + search_half)
                x_start = max(1, int(x_center) - search_half)
                x_end = min(w - 2, int(x_center) + search_half)

                # Find BRIGHTEST SINGLE PIXEL
                max_val = -1.0
                best_y = int(y_center)
                best_x = int(x_center)

                for sy in range(y_start, y_end + 1):
                    for sx in range(x_start, x_end + 1):
                        val = img_float[sy, sx]  # Single pixel value
                        if val > max_val:
                            max_val = val
                            best_y = sy
                            best_x = sx

                # Extract 3x3 region CENTERED on brightest pixel and calculate MEAN
                sum_y_start = max(0, best_y - 1)
                sum_y_end = min(h, best_y + 2)
                sum_x_start = max(0, best_x - 1)
                sum_x_end = min(w, best_x + 2)

                total = 0.0
                count = 0
                for py in range(sum_y_start, sum_y_end):
                    for px in range(sum_x_start, sum_x_end):
                        total += img_float[py, px]
                        count += 1

                if count > 0:
                    result[j, i] = total / count
                else:
                    result[j, i] = 0.0

        return result

else:
    _resize_brightest_center_3x3_core = None


# Numba-optimized core function for 3x3 brightest region (outside class for JIT compilation)
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _resize_display_pixel_3x3_core(
        img_float: np.ndarray,
        sum_3x3_map: np.ndarray,
        target_width: int,
        target_height: int,
        pitch_x: float,
        pitch_y: float,
        search_half: int
    ) -> np.ndarray:
        """
        Numba JIT optimized core function for display pixel resize.

        Algorithm (Pitch-based 3x3 brightest region):
        1. For each output pixel (i, j), calculate input position using pitch
           - x_center = i * pitch_x
           - y_center = j * pitch_y
        2. Search within ±search_half to find brightest 3x3 region
        3. Output = MEAN of that 3x3 region

        Note: Input pixels can overlap between different output pixels.
        """
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)

        for j in prange(target_height):
            y_center = j * pitch_y

            for i in range(target_width):
                x_center = i * pitch_x

                # Define search region around pitch-based center
                y_start = max(1, int(y_center) - search_half)
                y_end = min(h - 2, int(y_center) + search_half)
                x_start = max(1, int(x_center) - search_half)
                x_end = min(w - 2, int(x_center) + search_half)

                # Find brightest 3x3 region (using pre-computed 3x3 sum map)
                max_val = -1.0
                best_y = int(y_center)
                best_x = int(x_center)

                for sy in range(y_start, y_end + 1):
                    for sx in range(x_start, x_end + 1):
                        val = sum_3x3_map[sy, sx]
                        if val > max_val:
                            max_val = val
                            best_y = sy
                            best_x = sx

                # Extract 3x3 region centered on brightest position and calculate MEAN
                sum_y_start = max(0, best_y - 1)
                sum_y_end = min(h, best_y + 2)
                sum_x_start = max(0, best_x - 1)
                sum_x_end = min(w, best_x + 2)

                total = 0.0
                count = 0
                for py in range(sum_y_start, sum_y_end):
                    for px in range(sum_x_start, sum_x_end):
                        total += img_float[py, px]
                        count += 1

                if count > 0:
                    result[j, i] = total / count
                else:
                    result[j, i] = 0.0

        return result

else:
    _resize_display_pixel_3x3_core = None


class AreaSumResizer:
    """
    Resizes images using area-based summation method.

    Each output pixel is the weighted sum of all input pixels that overlap
    with the output pixel's region. Fractional pixel boundaries are handled
    by applying proportional weights based on overlap area.

    This method preserves total light energy and is ideal for
    downsampling camera sensor data to display resolution.
    """

    def __init__(self, show_progress: bool = True, progress_callback: Optional[Callable] = None):
        """
        Initialize AreaSumResizer.

        Args:
            show_progress: Whether to print progress updates
            progress_callback: Optional callback function(progress: float) for custom progress reporting
        """
        self.show_progress = show_progress
        self.progress_callback = progress_callback

    def resize_display_pixel_3x3(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        pitch_range: Tuple[float, float] = (3.7, 4.0)
    ) -> ResizeResult:
        """
        Resize image using display pixel structure detection with 3x3 brightest region.

        Algorithm (Pitch-based 3x3 brightest region MEAN):
        1. Calculate pitch from input/output size ratio
        2. For each output pixel (i, j), calculate input position using pitch:
           - x_center = i * pitch_x
           - y_center = j * pitch_y
        3. Within pitch_range, find the brightest 3x3 region
        4. Output = MEAN of that 3x3 region (9 pixels)

        Note: Input pixels can overlap between different output pixels.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height) - e.g., (2412, 2288)
            pitch_range: Expected pitch range for search (min, max)

        Returns:
            ResizeResult containing resized image with MEAN values
        """
        from scipy.ndimage import uniform_filter
        import time

        target_width, target_height = target_size
        h, w = image.shape[:2]

        # Calculate pitch from image dimensions (to get exact target size)
        pitch_x = w / target_width
        pitch_y = h / target_height

        if self.show_progress:
            print(f"Display Pixel Resize Mode (3x3 Brightest MEAN):")
            print(f"  Input size: {w} x {h}")
            print(f"  Target size: {target_width} x {target_height}")
            print(f"  Calculated pitch: {pitch_x:.4f} x {pitch_y:.4f}")
            print(f"  Search range: {pitch_range[0]:.1f} ~ {pitch_range[1]:.1f}")
            print(f"  Region: 3x3 (9 pixels)")
            print(f"  Output calculation: MEAN")
            print(f"  Pixel overlap: ALLOWED")

        # Use float64 for precision
        img_float = image.astype(np.float64)

        # Pre-compute 3x3 local sum using uniform_filter
        if self.show_progress:
            print("  Pre-computing 3x3 sum map...")
        sum_3x3_map = uniform_filter(img_float, size=3, mode='constant') * 9

        search_half = int(np.ceil(max(pitch_range) / 2)) + 1

        if NUMBA_AVAILABLE and _resize_display_pixel_3x3_core is not None:
            if self.show_progress:
                print(f"  Using Numba JIT optimization (parallel mode)...")
                print(f"  First run may take longer due to JIT compilation...")

            start_time = time.time()
            result = _resize_display_pixel_3x3_core(
                img_float, sum_3x3_map,
                target_width, target_height,
                pitch_x, pitch_y,
                search_half
            )
            elapsed = time.time() - start_time

            if self.show_progress:
                print(f"  Completed in {elapsed:.2f} seconds")
        else:
            # Fallback to pure Python (slow)
            if self.show_progress:
                print("  Warning: Numba not available, using slow Python loop...")
            result = self._resize_display_pixel_3x3_python(
                img_float, sum_3x3_map,
                target_width, target_height,
                pitch_x, pitch_y,
                search_half
            )

        return ResizeResult(
            image=result,
            scale_x=pitch_x,
            scale_y=pitch_y,
            pixels_per_output=9  # 3x3 = 9 pixels
        )

    def _resize_display_pixel_3x3_python(
        self,
        img_float: np.ndarray,
        sum_3x3_map: np.ndarray,
        target_width: int,
        target_height: int,
        pitch_x: float,
        pitch_y: float,
        search_half: int
    ) -> np.ndarray:
        """Pure Python fallback for resize_display_pixel_3x3 (slow)."""
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)

        for j in range(target_height):
            y_center = j * pitch_y

            for i in range(target_width):
                x_center = i * pitch_x

                y_start = max(1, int(y_center) - search_half)
                y_end = min(h - 2, int(y_center) + search_half)
                x_start = max(1, int(x_center) - search_half)
                x_end = min(w - 2, int(x_center) + search_half)

                search_region = sum_3x3_map[y_start:y_end+1, x_start:x_end+1]

                if search_region.size == 0:
                    continue

                max_idx = np.argmax(search_region)
                local_y, local_x = np.unravel_index(max_idx, search_region.shape)

                best_y = y_start + local_y
                best_x = x_start + local_x

                # Extract 3x3 region centered on best position
                sum_y_start = max(0, best_y - 1)
                sum_y_end = min(h, best_y + 2)
                sum_x_start = max(0, best_x - 1)
                sum_x_end = min(w, best_x + 2)

                region = img_float[sum_y_start:sum_y_end, sum_x_start:sum_x_end]
                if region.size > 0:
                    result[j, i] = np.mean(region)

            if self.show_progress and j % (target_height // 10) == 0:
                print(f"  Progress: {j}/{target_height} rows ({100*j/target_height:.0f}%)")

        if self.show_progress:
            print(f"  Progress: {target_height}/{target_height} rows (100%)")

        return result

    def resize_brightest_center_3x3(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        pitch_range: Tuple[float, float] = (3.7, 4.0)
    ) -> ResizeResult:
        """
        Resize image by finding brightest SINGLE pixel, then 3x3 centered on it.

        Algorithm:
        1. Calculate pitch from input/output size ratio
        2. For each output pixel (i, j), calculate input position using pitch:
           - x_center = i * pitch_x
           - y_center = j * pitch_y
        3. Within pitch_range, find the BRIGHTEST SINGLE PIXEL
        4. Extract 3x3 CENTERED on that brightest pixel
        5. Output = MEAN of 3x3 (9 pixels)

        Note: Input pixels can overlap between different output pixels.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height) - e.g., (2412, 2288)
            pitch_range: Expected pitch range for search (min, max)

        Returns:
            ResizeResult containing resized image with MEAN values
        """
        import time

        target_width, target_height = target_size
        h, w = image.shape[:2]

        # Calculate pitch from image dimensions (to get exact target size)
        pitch_x = w / target_width
        pitch_y = h / target_height

        if self.show_progress:
            print(f"Display Pixel Resize Mode (Brightest Pixel Center → 3x3 MEAN):")
            print(f"  Input size: {w} x {h}")
            print(f"  Target size: {target_width} x {target_height}")
            print(f"  Calculated pitch: {pitch_x:.4f} x {pitch_y:.4f}")
            print(f"  Search range: {pitch_range[0]:.1f} ~ {pitch_range[1]:.1f}")
            print(f"  Method: Find brightest SINGLE pixel → 3x3 centered on it")
            print(f"  Output calculation: MEAN of 3x3 (9 pixels)")
            print(f"  Pixel overlap: ALLOWED")

        # Use float64 for precision
        img_float = image.astype(np.float64)

        search_half = int(np.ceil(max(pitch_range) / 2)) + 1

        if NUMBA_AVAILABLE and _resize_brightest_center_3x3_core is not None:
            if self.show_progress:
                print(f"  Using Numba JIT optimization (parallel mode)...")

            start_time = time.time()
            result = _resize_brightest_center_3x3_core(
                img_float,
                target_width, target_height,
                pitch_x, pitch_y,
                search_half
            )
            elapsed = time.time() - start_time

            if self.show_progress:
                print(f"  Completed in {elapsed:.2f} seconds")
        else:
            # Fallback to pure Python (slow)
            if self.show_progress:
                print("  Warning: Numba not available, using slow Python loop...")
            result = self._resize_brightest_center_3x3_python(
                img_float,
                target_width, target_height,
                pitch_x, pitch_y,
                search_half
            )

        return ResizeResult(
            image=result,
            scale_x=pitch_x,
            scale_y=pitch_y,
            pixels_per_output=9  # 3x3 = 9 pixels
        )

    def _resize_brightest_center_3x3_python(
        self,
        img_float: np.ndarray,
        target_width: int,
        target_height: int,
        pitch_x: float,
        pitch_y: float,
        search_half: int
    ) -> np.ndarray:
        """Pure Python fallback for resize_brightest_center_3x3 (slow)."""
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)

        for j in range(target_height):
            y_center = j * pitch_y

            for i in range(target_width):
                x_center = i * pitch_x

                y_start = max(1, int(y_center) - search_half)
                y_end = min(h - 2, int(y_center) + search_half)
                x_start = max(1, int(x_center) - search_half)
                x_end = min(w - 2, int(x_center) + search_half)

                # Find brightest SINGLE pixel
                search_region = img_float[y_start:y_end+1, x_start:x_end+1]

                if search_region.size == 0:
                    continue

                max_idx = np.argmax(search_region)
                local_y, local_x = np.unravel_index(max_idx, search_region.shape)

                best_y = y_start + local_y
                best_x = x_start + local_x

                # Extract 3x3 region CENTERED on brightest pixel
                sum_y_start = max(0, best_y - 1)
                sum_y_end = min(h, best_y + 2)
                sum_x_start = max(0, best_x - 1)
                sum_x_end = min(w, best_x + 2)

                region = img_float[sum_y_start:sum_y_end, sum_x_start:sum_x_end]
                if region.size > 0:
                    result[j, i] = np.mean(region)

            if self.show_progress and j % (target_height // 10) == 0:
                print(f"  Progress: {j}/{target_height} rows ({100*j/target_height:.0f}%)")

        if self.show_progress:
            print(f"  Progress: {target_height}/{target_height} rows (100%)")

        return result

    def resize_2d_grid_tracking(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resize with 2D grid tracking for ZIGZAG pattern.

        Each pixel position is determined by:
        1. Pixel directly above (vertical constraint: pitch 3 or 4)
        2. Pixel to the left (horizontal constraint: pitch 3 or 4)

        This captures zigzag patterns where rows alternate positions.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height) - e.g., (2412, 2288)

        Returns:
            Tuple of (result_image, centers_x, centers_y)
        """
        import time

        target_width, target_height = target_size
        h, w = image.shape[:2]

        if self.show_progress:
            print(f"2D Grid Tracking Resize (Zigzag pattern):")
            print(f"  Input size: {w} x {h}")
            print(f"  Target size: {target_width} x {target_height}")
            print(f"  Constraint: pitch 3 or 4 in BOTH directions")
            print(f"  Uses: above pixel + left pixel references")

        img_float = image.astype(np.float64)

        if NUMBA_AVAILABLE and _resize_2d_grid_tracking_core is not None:
            if self.show_progress:
                print(f"  Using Numba JIT optimization...")

            start_time = time.time()
            result, centers_x, centers_y = _resize_2d_grid_tracking_core(
                img_float, target_width, target_height
            )
            elapsed = time.time() - start_time

            if self.show_progress:
                print(f"  Completed in {elapsed:.2f} seconds")
        else:
            if self.show_progress:
                print("  Warning: Numba not available")
            raise RuntimeError("Numba required for 2D grid tracking")

        return result, centers_x, centers_y

    def resize_strict_3x3(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resize with STRICT 3x3 grid constraint: 0 or 1 pixel gap only.

        This ensures NO pixels are skipped because:
        - 3x3 box spacing is exactly 3 or 4 pixels
        - Gap = 0 → center spacing = 3 pixels
        - Gap = 1 → center spacing = 4 pixels

        For each position, both options (+3 and +4) are checked,
        and the brighter center is selected.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height) - e.g., (2412, 2288)

        Returns:
            Tuple of (result_image, centers_x, centers_y)
        """
        import time

        target_width, target_height = target_size
        h, w = image.shape[:2]

        if self.show_progress:
            print(f"Strict 3x3 Grid Resize (0 or 1 pixel gap only):")
            print(f"  Input size: {w} x {h}")
            print(f"  Target size: {target_width} x {target_height}")
            print(f"  Constraint: 3x3 box centers must be 3 or 4 pixels apart")
            print(f"  Gap: 0 pixel (pitch=3) or 1 pixel (pitch=4)")

        img_float = image.astype(np.float64)

        if NUMBA_AVAILABLE and _resize_strict_3x3_core is not None:
            if self.show_progress:
                print(f"  Using Numba JIT optimization...")

            start_time = time.time()
            result, centers_x, centers_y = _resize_strict_3x3_core(
                img_float, target_width, target_height
            )
            elapsed = time.time() - start_time

            if self.show_progress:
                print(f"  Completed in {elapsed:.2f} seconds")
        else:
            if self.show_progress:
                print("  Warning: Numba not available, using Python fallback...")
            result, centers_x, centers_y = self._resize_strict_3x3_python(
                img_float, target_width, target_height
            )

        return result, centers_x, centers_y

    def _resize_strict_3x3_python(
        self,
        img_float: np.ndarray,
        target_width: int,
        target_height: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure Python fallback for strict 3x3 grid."""
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)
        all_centers_x = np.zeros((target_height, target_width), dtype=np.int32)
        all_centers_y = np.zeros((target_height, target_width), dtype=np.int32)

        row_y = 2  # Start near top

        for j in range(target_height):
            if j > 0:
                prev_y = int(np.mean(all_centers_y[j - 1, :]))
                # Choose between +3 and +4 based on brightness
                if prev_y + 3 < h:
                    sum3 = np.sum(img_float[prev_y + 3, :min(100, w)])
                else:
                    sum3 = 0
                if prev_y + 4 < h:
                    sum4 = np.sum(img_float[prev_y + 4, :min(100, w)])
                else:
                    sum4 = 0
                row_y = prev_y + 3 if sum3 >= sum4 else prev_y + 4

            row_y = max(1, min(h - 2, row_y))
            y_search = 2

            # First pixel
            x_range = img_float[max(0, row_y-1):min(h, row_y+2), 0:6]
            if x_range.size > 0:
                max_idx = np.argmax(x_range)
                local_y, local_x = np.unravel_index(max_idx, x_range.shape)
                all_centers_x[j, 0] = local_x
                all_centers_y[j, 0] = max(0, row_y - 1) + local_y

            # Track with pitch 3 or 4
            for i in range(1, target_width):
                prev_x = all_centers_x[j, i - 1]
                prev_cy = all_centers_y[j, i - 1]

                # Option 1: +3
                x1 = prev_x + 3
                region1 = img_float[max(0, prev_cy-1):min(h, prev_cy+2),
                                   max(0, x1-1):min(w, x1+2)]
                val1 = region1.max() if region1.size > 0 else 0

                # Option 2: +4
                x2 = prev_x + 4
                region2 = img_float[max(0, prev_cy-1):min(h, prev_cy+2),
                                   max(0, x2-1):min(w, x2+2)]
                val2 = region2.max() if region2.size > 0 else 0

                if val1 >= val2 and region1.size > 0:
                    max_idx = np.argmax(region1)
                    local_y, local_x = np.unravel_index(max_idx, region1.shape)
                    all_centers_x[j, i] = max(0, x1 - 1) + local_x
                    all_centers_y[j, i] = max(0, prev_cy - 1) + local_y
                elif region2.size > 0:
                    max_idx = np.argmax(region2)
                    local_y, local_x = np.unravel_index(max_idx, region2.shape)
                    all_centers_x[j, i] = max(0, x2 - 1) + local_x
                    all_centers_y[j, i] = max(0, prev_cy - 1) + local_y
                else:
                    all_centers_x[j, i] = prev_x + 3
                    all_centers_y[j, i] = prev_cy

            # Calculate 3x3 MEAN
            for i in range(target_width):
                cx, cy = all_centers_x[j, i], all_centers_y[j, i]
                region = img_float[max(0, cy-1):min(h, cy+2), max(0, cx-1):min(w, cx+2)]
                if region.size > 0:
                    result[j, i] = np.mean(region)

            if self.show_progress and j % (target_height // 10) == 0:
                print(f"  Progress: {j}/{target_height} ({100*j/target_height:.0f}%)")

        return result, all_centers_x, all_centers_y

    def resize_sequential_tracking(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        pitch_range: Tuple[float, float] = (3.7, 4.0)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resize using sequential bright pixel tracking - NO SKIPPING.

        This method tracks bright pixels sequentially along each row,
        ensuring no bright pixels are skipped.

        Algorithm:
        1. For each row, start from x=0 and find first bright pixel
        2. Track horizontally: each next pixel is found at pitch_min~pitch_max
           distance from previous pixel
        3. Vertically: use row_index * avg_pitch as the y center
        4. Each bright pixel center → 3x3 MEAN for output

        This is different from fixed-pitch methods because it FOLLOWS
        the actual bright pixels rather than jumping to fixed positions.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height) - e.g., (2412, 2288)
            pitch_range: Expected pitch range for search (min, max)

        Returns:
            Tuple of (result_image, centers_x, centers_y)
            - result_image: 2412 x 2288 output image (MEAN of 3x3)
            - centers_x: 2288 x 2412 array of bright pixel x positions
            - centers_y: 2288 x 2412 array of bright pixel y positions
        """
        import time

        target_width, target_height = target_size
        h, w = image.shape[:2]

        pitch_min, pitch_max = pitch_range

        if self.show_progress:
            print(f"Sequential Tracking Resize Mode (No Skipping):")
            print(f"  Input size: {w} x {h}")
            print(f"  Target size: {target_width} x {target_height}")
            print(f"  Pitch range: {pitch_min:.2f} ~ {pitch_max:.2f}")
            print(f"  Method: Track bright pixels sequentially along each row")
            print(f"  Output: 3x3 MEAN centered on each bright pixel")
            print(f"  Key feature: NO SKIPPING - follows actual bright pixels")

        img_float = image.astype(np.float64)

        if NUMBA_AVAILABLE and _resize_sequential_tracking_core is not None:
            if self.show_progress:
                print(f"  Using Numba JIT optimization...")

            start_time = time.time()
            result, centers_x, centers_y = _resize_sequential_tracking_core(
                img_float,
                target_width, target_height,
                pitch_min, pitch_max
            )
            elapsed = time.time() - start_time

            if self.show_progress:
                print(f"  Completed in {elapsed:.2f} seconds")
        else:
            if self.show_progress:
                print("  Warning: Numba not available, using Python fallback...")
            result, centers_x, centers_y = self._resize_sequential_tracking_python(
                img_float, target_width, target_height, pitch_min, pitch_max
            )

        return result, centers_x, centers_y

    def _resize_sequential_tracking_python(
        self,
        img_float: np.ndarray,
        target_width: int,
        target_height: int,
        pitch_min: float,
        pitch_max: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure Python fallback for sequential tracking."""
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)
        all_centers_x = np.zeros((target_height, target_width), dtype=np.int32)
        all_centers_y = np.zeros((target_height, target_width), dtype=np.int32)

        avg_pitch = (pitch_min + pitch_max) / 2
        search_y = int(np.ceil(pitch_max / 2))

        for j in range(target_height):
            row_y_center = int(j * avg_pitch)
            y_start = max(1, row_y_center - search_y)
            y_end = min(h - 1, row_y_center + search_y)

            # Find first bright pixel at start of row
            x_end = int(pitch_max) + 1
            search_region = img_float[y_start:y_end+1, 0:x_end+1]
            if search_region.size > 0:
                max_idx = np.argmax(search_region)
                local_y, local_x = np.unravel_index(max_idx, search_region.shape)
                first_x = local_x
                first_y = y_start + local_y
            else:
                first_x = 0
                first_y = row_y_center

            all_centers_x[j, 0] = first_x
            all_centers_y[j, 0] = first_y

            # Track subsequent pixels
            for i in range(1, target_width):
                prev_x = all_centers_x[j, i-1]
                x_start = prev_x + int(pitch_min)
                x_end = prev_x + int(pitch_max) + 1

                if x_start >= w:
                    all_centers_x[j, i] = w - 1
                    all_centers_y[j, i] = row_y_center
                else:
                    search_region = img_float[y_start:y_end+1, max(0,x_start):min(w,x_end+1)]
                    if search_region.size > 0:
                        max_idx = np.argmax(search_region)
                        local_y, local_x = np.unravel_index(max_idx, search_region.shape)
                        all_centers_x[j, i] = max(0, x_start) + local_x
                        all_centers_y[j, i] = y_start + local_y
                    else:
                        all_centers_x[j, i] = prev_x + int(avg_pitch)
                        all_centers_y[j, i] = row_y_center

            # Calculate 3x3 MEAN for each center
            for i in range(target_width):
                cx, cy = all_centers_x[j, i], all_centers_y[j, i]
                y1 = max(0, cy - 1)
                y2 = min(h, cy + 2)
                x1 = max(0, cx - 1)
                x2 = min(w, cx + 2)

                region = img_float[y1:y2, x1:x2]
                if region.size > 0:
                    result[j, i] = np.mean(region)

            if self.show_progress and j % (target_height // 10) == 0:
                print(f"  Progress: {j}/{target_height} ({100*j/target_height:.0f}%)")

        return result, all_centers_x, all_centers_y

    def resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> ResizeResult:
        """
        Resize image by summing all camera pixels corresponding to each display pixel.
        Handles fractional boundaries with proportional weighting.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height)

        Returns:
            ResizeResult containing resized image and scale factors
        """
        target_width, target_height = target_size
        h, w = image.shape[:2]

        scale_x = w / target_width
        scale_y = h / target_height

        if self.show_progress:
            print(f"Area Sum Resize Mode:")
            print(f"  Input size: {w} x {h}")
            print(f"  Target size: {target_width} x {target_height}")
            print(f"  Scale factors: {scale_x:.4f} x {scale_y:.4f}")

        img_float = image.astype(np.float64)
        result = np.zeros((target_height, target_width), dtype=np.float64)

        for j in range(target_height):
            y_start = j * scale_y
            y_end = (j + 1) * scale_y
            y_start_int = int(np.floor(y_start))
            y_end_int = int(np.ceil(y_end))

            for i in range(target_width):
                x_start = i * scale_x
                x_end = (i + 1) * scale_x
                x_start_int = int(np.floor(x_start))
                x_end_int = int(np.ceil(x_end))

                pixel_sum = 0.0

                for yi in range(y_start_int, min(y_end_int, h)):
                    y_weight = 1.0
                    if yi == y_start_int:
                        y_weight = 1.0 - (y_start - y_start_int)
                    if yi == y_end_int - 1:
                        y_weight = min(y_weight, y_end - yi)

                    for xi in range(x_start_int, min(x_end_int, w)):
                        x_weight = 1.0
                        if xi == x_start_int:
                            x_weight = 1.0 - (x_start - x_start_int)
                        if xi == x_end_int - 1:
                            x_weight = min(x_weight, x_end - xi)

                        pixel_sum += img_float[yi, xi] * x_weight * y_weight

                result[j, i] = pixel_sum

            if self.show_progress and j % (target_height // 10 + 1) == 0:
                print(f"  Progress: {j}/{target_height} ({100*j/target_height:.0f}%)")

        if self.show_progress:
            print(f"  Progress: {target_height}/{target_height} (100%)")

        pixels_per_output = scale_x * scale_y

        return ResizeResult(
            image=result,
            scale_x=scale_x,
            scale_y=scale_y,
            pixels_per_output=pixels_per_output
        )

    def resize_mean(self, image: np.ndarray, target_size: Tuple[int, int]) -> ResizeResult:
        """
        Resize image by averaging pixels (normalized sum).

        Same as resize() but divides by the total weight to get mean instead of sum.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height)

        Returns:
            ResizeResult containing resized image with mean values
        """
        result = self.resize(image, target_size)

        # Normalize by the area (divide by total pixels summed)
        result.image = result.image / result.pixels_per_output

        return result

    def resize_display_pixel(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        pitch: float = 3.88,
        pitch_range: Tuple[float, float] = (3.7, 4.0),
        bright_kernel_size: int = 2,
        sum_kernel_size: int = 4
    ) -> ResizeResult:
        """
        Resize image using display pixel structure detection (Numba optimized).

        Algorithm (Pitch-based with overlap, MEAN calculation):
        1. For each output pixel (i, j), calculate input position using pitch:
           - x_center = i * pitch
           - y_center = j * pitch
        2. Within ±search_half, find the brightest 2x2 pixel center
        3. Extract 4x4 pixels centered on that bright center
        4. Calculate MEAN of 4x4 pixels (not sum)

        Note: Input pixels can overlap between different output pixels.
              If edge pixels are missing, mean is calculated from available pixels.

        Args:
            image: Input image array (H, W) grayscale
            target_size: Target size as (width, height)
            pitch: Display pixel pitch in camera pixels (default 3.88)
            pitch_range: Expected pitch range for search (min, max)
            bright_kernel_size: Size of bright center kernel (default 2x2)
            sum_kernel_size: Size of sum kernel (default 4x4)

        Returns:
            ResizeResult containing resized image with MEAN values
        """
        from scipy.ndimage import uniform_filter
        import time

        target_width, target_height = target_size
        h, w = image.shape[:2]

        # Use pitch directly (not scale factors)
        pitch_x = pitch
        pitch_y = pitch

        if self.show_progress:
            print(f"Display Pixel Resize Mode (Pitch-based MEAN):")
            print(f"  Input size: {w} x {h}")
            print(f"  Target size: {target_width} x {target_height}")
            print(f"  Pitch: {pitch_x:.4f} x {pitch_y:.4f} camera pixels per display pixel")
            print(f"  Search range: {pitch_range[0]:.1f} ~ {pitch_range[1]:.1f}")
            print(f"  Bright kernel: {bright_kernel_size}x{bright_kernel_size}")
            print(f"  Sum kernel: {sum_kernel_size}x{sum_kernel_size}")
            print(f"  Output calculation: MEAN (not sum)")
            print(f"  Pixel overlap: ALLOWED")

        # Use float64 for precision
        img_float = image.astype(np.float64)

        # Pre-compute 2x2 local sum using uniform_filter
        if self.show_progress:
            print("  Pre-computing brightness sum map...")
        bright_sum_map = uniform_filter(img_float, size=bright_kernel_size, mode='constant') * (bright_kernel_size ** 2)

        half_sum = sum_kernel_size // 2
        search_half = int(np.ceil(max(pitch_range) / 2)) + 1

        if NUMBA_AVAILABLE:
            if self.show_progress:
                print(f"  Using Numba JIT optimization (parallel mode)...")
                print(f"  First run may take longer due to JIT compilation...")

            start_time = time.time()
            result = _resize_display_pixel_core(
                img_float, bright_sum_map,
                target_width, target_height,
                pitch_x, pitch_y,
                search_half, half_sum
            )
            elapsed = time.time() - start_time

            if self.show_progress:
                print(f"  Completed in {elapsed:.2f} seconds")
        else:
            # Fallback to pure Python (slow)
            if self.show_progress:
                print("  Warning: Numba not available, using slow Python loop...")
            result = self._resize_display_pixel_python(
                img_float, bright_sum_map,
                target_width, target_height,
                pitch_x, pitch_y,
                search_half, half_sum
            )

        return ResizeResult(
            image=result,
            scale_x=pitch_x,
            scale_y=pitch_y,
            pixels_per_output=sum_kernel_size * sum_kernel_size
        )

    def _resize_display_pixel_python(
        self,
        img_float: np.ndarray,
        bright_sum_map: np.ndarray,
        target_width: int,
        target_height: int,
        pitch_x: float,
        pitch_y: float,
        search_half: int,
        half_sum: int
    ) -> np.ndarray:
        """Pure Python fallback for resize_display_pixel (slow)."""
        h, w = img_float.shape
        result = np.zeros((target_height, target_width), dtype=np.float64)

        for j in range(target_height):
            # Pitch-based position
            y_center = j * pitch_y

            for i in range(target_width):
                x_center = i * pitch_x

                y_start = max(0, int(y_center) - search_half)
                y_end = min(h - 1, int(y_center) + search_half)
                x_start = max(0, int(x_center) - search_half)
                x_end = min(w - 1, int(x_center) + search_half)

                search_region = bright_sum_map[y_start:y_end+1, x_start:x_end+1]

                if search_region.size == 0:
                    continue

                max_idx = np.argmax(search_region)
                local_y, local_x = np.unravel_index(max_idx, search_region.shape)

                bright_center_y = y_start + local_y
                bright_center_x = x_start + local_x

                sum_y_start = max(0, bright_center_y - half_sum)
                sum_y_end = min(h, bright_center_y + half_sum)
                sum_x_start = max(0, bright_center_x - half_sum)
                sum_x_end = min(w, bright_center_x + half_sum)

                region = img_float[sum_y_start:sum_y_end, sum_x_start:sum_x_end]
                if region.size > 0:
                    result[j, i] = np.mean(region)

            if self.show_progress and j % (target_height // 10) == 0:
                print(f"  Progress: {j}/{target_height} rows ({100*j/target_height:.0f}%)")

        if self.show_progress:
            print(f"  Progress: {target_height}/{target_height} rows (100%)")

        return result
