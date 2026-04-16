# -*- coding: utf-8 -*-
"""
Display Panel Processor Module
Main pipeline class that orchestrates the entire ROI processing workflow.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from importlib import import_module as _import_module

# Import modules with numbered prefixes using importlib
# Use relative imports within the same package
_image_cropper = _import_module('.3_image_cropper', package='src')
_roi_detector = _import_module('.2_roi_detector', package='src')
_perspective_warper = _import_module('.4_perspective_warper', package='src')
_area_sum_resizer = _import_module('.5_area_sum_resizer', package='src')
_image_normalizer = _import_module('.6_image_normalizer', package='src')

ImageCropper = _image_cropper.ImageCropper
CropRegion = _image_cropper.CropRegion
ROIDetector = _roi_detector.ROIDetector
DetectedROI = _roi_detector.DetectedROI
PerspectiveWarper = _perspective_warper.PerspectiveWarper
WarpResult = _perspective_warper.WarpResult
AreaSumResizer = _area_sum_resizer.AreaSumResizer
ResizeResult = _area_sum_resizer.ResizeResult
ImageNormalizer = _image_normalizer.ImageNormalizer
NormalizationResult = _image_normalizer.NormalizationResult


@dataclass
class ProcessingConfig:
    """Configuration for display panel processing"""
    # Crop settings
    crop_x_start: int = 1700
    crop_y_start: int = 300
    crop_x_end: int = 11700
    crop_y_end: int = 9900
    use_crop: bool = True

    # ROI detection settings
    roi_threshold: int = 50
    morph_kernel_size: int = 51

    # Display settings
    display_width: int = 2412
    display_height: int = 2288
    camera_display_ratio: float = 3.8

    # Output settings
    output_bit_depth: int = 16
    save_intermediates: bool = True


@dataclass
class ProcessingResult:
    """Result of the entire processing pipeline"""
    original_image: np.ndarray
    cropped_image: Optional[np.ndarray]
    roi: Optional[DetectedROI]
    warped_image: Optional[np.ndarray]
    resized_image: Optional[np.ndarray]
    normalized_image: Optional[np.ndarray]
    config: ProcessingConfig
    stats: Dict[str, Any] = field(default_factory=dict)


class DisplayPanelProcessor:
    """
    Main processor class for display panel ROI extraction and processing.

    This class orchestrates the entire pipeline:
    1. Image cropping (optional)
    2. ROI detection
    3. Perspective warp correction
    4. Area-sum resizing to display resolution
    5. 16-bit normalization

    Example:
        processor = DisplayPanelProcessor()
        result = processor.process("path/to/image.tif")
        processor.save_results(result, "output/")
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize DisplayPanelProcessor.

        Args:
            config: ProcessingConfig object with all settings
        """
        self.config = config or ProcessingConfig()

        # Initialize components
        self._cropper = ImageCropper()
        if self.config.use_crop:
            self._cropper.set_region(
                self.config.crop_x_start,
                self.config.crop_y_start,
                self.config.crop_x_end,
                self.config.crop_y_end
            )

        self._roi_detector = ROIDetector(
            threshold=self.config.roi_threshold,
            morph_kernel_size=self.config.morph_kernel_size
        )

        self._warper = PerspectiveWarper(interpolation=cv2.INTER_NEAREST)

        self._resizer = AreaSumResizer(show_progress=True)

        self._normalizer = ImageNormalizer(bit_depth=self.config.output_bit_depth)

    def process(self, image_path: str) -> ProcessingResult:
        """
        Process an image through the entire pipeline.

        Args:
            image_path: Path to the input image file

        Returns:
            ProcessingResult containing all processing stages and metadata
        """
        print("=" * 60)
        print("Display Panel Processing Pipeline")
        print("=" * 60)

        # Step 1: Load image
        print("\n[Step 1] Loading image...")
        original = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if original is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        print(f"  Original image: {original.shape[1]} x {original.shape[0]} pixels")
        print(f"  Dtype: {original.dtype}, Range: [{original.min()}, {original.max()}]")

        stats = {
            "original_shape": original.shape,
            "original_dtype": str(original.dtype),
            "original_range": (int(original.min()), int(original.max()))
        }

        # Step 2: Crop (optional)
        print("\n[Step 2] Cropping image...")
        if self.config.use_crop:
            cropped = self._cropper.crop(original)
            print(f"  Crop region: ({self.config.crop_x_start}, {self.config.crop_y_start}) to "
                  f"({self.config.crop_x_end}, {self.config.crop_y_end})")
            print(f"  Cropped image: {cropped.shape[1]} x {cropped.shape[0]} pixels")
            stats["cropped_shape"] = cropped.shape
        else:
            cropped = original
            print("  Cropping disabled")

        # Step 3: Detect ROI
        print("\n[Step 3] Detecting ROI...")
        roi = self._roi_detector.detect(cropped)
        if roi is None:
            print("  WARNING: No ROI detected!")
            return ProcessingResult(
                original_image=original,
                cropped_image=cropped,
                roi=None,
                warped_image=None,
                resized_image=None,
                normalized_image=None,
                config=self.config,
                stats=stats
            )

        print(f"  Detected ROI: {roi.width:.0f} x {roi.height:.0f} pixels")
        print(f"  Tilt angle: {roi.angle:.2f} degrees")
        print(f"  Contour area: {roi.area:.0f} pixels")
        stats["roi_size"] = (roi.width, roi.height)
        stats["tilt_angle"] = roi.angle

        # Step 4: Perspective warp
        print("\n[Step 4] Applying perspective warp...")
        warp_result = self._warper.warp(cropped, roi.corners)
        warped = warp_result.image
        print(f"  Warped image: {warped.shape[1]} x {warped.shape[0]} pixels")
        print(f"  Using INTER_NEAREST to preserve pixel values")
        stats["warped_shape"] = warped.shape

        # Step 5: Resize to display resolution
        print("\n[Step 5] Resizing to display resolution...")
        target_size = (self.config.display_width, self.config.display_height)
        print(f"  Target size: {target_size[0]} x {target_size[1]} pixels")

        resize_result = self._resizer.resize(warped, target_size)
        resized = resize_result.image
        print(f"  Resized image: {resized.shape[1]} x {resized.shape[0]} pixels")
        print(f"  Scale factors: {resize_result.scale_x:.4f} x {resize_result.scale_y:.4f}")
        stats["resized_shape"] = resized.shape
        stats["scale_factors"] = (resize_result.scale_x, resize_result.scale_y)

        # Step 6: Normalize to 16-bit
        print("\n[Step 6] Normalizing to 16-bit...")
        norm_result = self._normalizer.normalize(resized)
        normalized = norm_result.image
        print(f"  Original range: [{norm_result.original_min:.2f}, {norm_result.original_max:.2f}]")
        print(f"  Normalized range: [{norm_result.normalized_min}, {norm_result.normalized_max}]")
        stats["normalization"] = {
            "original_range": (norm_result.original_min, norm_result.original_max),
            "normalized_range": (norm_result.normalized_min, norm_result.normalized_max)
        }

        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)

        return ProcessingResult(
            original_image=original,
            cropped_image=cropped,
            roi=roi,
            warped_image=warped,
            resized_image=resized,
            normalized_image=normalized,
            config=self.config,
            stats=stats
        )

    def save_results(self, result: ProcessingResult, output_dir: str) -> Dict[str, str]:
        """
        Save processing results to files.

        Args:
            result: ProcessingResult from process()
            output_dir: Output directory path

        Returns:
            Dictionary mapping result names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save cropped image
        if result.cropped_image is not None and self.config.save_intermediates:
            path = output_path / "00_cropped_image.tif"
            cv2.imwrite(str(path), result.cropped_image)
            saved_files["cropped"] = str(path)
            print(f"Saved: {path}")

        # Save warped image
        if result.warped_image is not None and self.config.save_intermediates:
            path = output_path / "04_warped_image.tif"
            cv2.imwrite(str(path), result.warped_image.astype(np.uint16))
            saved_files["warped"] = str(path)
            print(f"Saved: {path}")

        # Save resized image
        if result.resized_image is not None and self.config.save_intermediates:
            path = output_path / "05_resized_image.tif"
            cv2.imwrite(str(path), result.resized_image.astype(np.uint16))
            saved_files["resized"] = str(path)
            print(f"Saved: {path}")

        # Save normalized image (main output)
        if result.normalized_image is not None:
            path = output_path / "06_final_normalized.tif"
            cv2.imwrite(str(path), result.normalized_image)
            saved_files["normalized"] = str(path)
            print(f"Saved: {path}")

            # Also save 8-bit preview
            preview_8bit = ImageNormalizer.convert_16bit_to_8bit(result.normalized_image)
            path = output_path / "06_final_normalized_preview.png"
            cv2.imwrite(str(path), preview_8bit)
            saved_files["preview"] = str(path)
            print(f"Saved: {path}")

        return saved_files

    def save_visualization(self, result: ProcessingResult, output_dir: str) -> str:
        """
        Save visualization of the processing pipeline.

        Args:
            result: ProcessingResult from process()
            output_dir: Output directory path

        Returns:
            Path to saved visualization
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original/Cropped
        axes[0].imshow(result.cropped_image, cmap='gray')
        axes[0].set_title(f'Cropped ({result.cropped_image.shape[1]}x{result.cropped_image.shape[0]})')
        if result.roi is not None:
            corners = result.roi.corners
            for i in range(4):
                pt1 = corners[i]
                pt2 = corners[(i+1) % 4]
                axes[0].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=2)

        # Warped
        if result.warped_image is not None:
            axes[1].imshow(result.warped_image, cmap='gray')
            axes[1].set_title(f'Warped ({result.warped_image.shape[1]}x{result.warped_image.shape[0]})')

        # Final
        if result.normalized_image is not None:
            axes[2].imshow(result.normalized_image, cmap='gray')
            axes[2].set_title(f'Final ({result.normalized_image.shape[1]}x{result.normalized_image.shape[0]})')

        plt.tight_layout()

        path = output_path / "06_final_comparison.png"
        plt.savefig(str(path), dpi=150)
        plt.close()

        print(f"Saved: {path}")
        return str(path)

    # Property accessors for component customization
    @property
    def cropper(self) -> ImageCropper:
        return self._cropper

    @property
    def roi_detector(self) -> ROIDetector:
        return self._roi_detector

    @property
    def warper(self) -> PerspectiveWarper:
        return self._warper

    @property
    def resizer(self) -> AreaSumResizer:
        return self._resizer

    @property
    def normalizer(self) -> ImageNormalizer:
        return self._normalizer
