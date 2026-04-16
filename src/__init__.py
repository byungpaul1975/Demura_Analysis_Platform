# -*- coding: utf-8 -*-
"""
ROI Algorithm Package

This package provides tools for processing display panel images
captured by mono cameras. It includes ROI detection, perspective
correction, area-sum resizing, and 16-bit normalization.

Main Classes:
    - DisplayPanelProcessor: Main pipeline orchestrator
    - ImageCropper: Image cropping utility
    - ROIDetector: Region of interest detection
    - PerspectiveWarper: Tilt correction via perspective transform
    - AreaSumResizer: Area-based pixel summation resizing
    - ImageNormalizer: Bit-depth normalization

Example:
    from src import DisplayPanelProcessor, ProcessingConfig

    config = ProcessingConfig(
        display_width=2412,
        display_height=2288
    )
    processor = DisplayPanelProcessor(config)
    result = processor.process("data/image.tif")
    processor.save_results(result, "output/")
"""

from importlib import import_module as _import_module

# Import modules with numbered prefixes using importlib
_image_cropper = _import_module('.3_image_cropper', package=__name__)
_roi_detector = _import_module('.2_roi_detector', package=__name__)
_perspective_warper = _import_module('.4_perspective_warper', package=__name__)
_area_sum_resizer = _import_module('.5_area_sum_resizer', package=__name__)
_image_normalizer = _import_module('.6_image_normalizer', package=__name__)
_display_panel_processor = _import_module('.7_display_panel_processor', package=__name__)

# Re-export classes
ImageCropper = _image_cropper.ImageCropper
CropRegion = _image_cropper.CropRegion
ROIDetector = _roi_detector.ROIDetector
AdaptiveROIDetector = _roi_detector.AdaptiveROIDetector
DetectedROI = _roi_detector.DetectedROI
PerspectiveWarper = _perspective_warper.PerspectiveWarper
WarpResult = _perspective_warper.WarpResult
AreaSumResizer = _area_sum_resizer.AreaSumResizer
ResizeResult = _area_sum_resizer.ResizeResult
ImageNormalizer = _image_normalizer.ImageNormalizer
NormalizationResult = _image_normalizer.NormalizationResult
DisplayPanelProcessor = _display_panel_processor.DisplayPanelProcessor
ProcessingConfig = _display_panel_processor.ProcessingConfig
ProcessingResult = _display_panel_processor.ProcessingResult

__version__ = "1.0.0"

__all__ = [
    # Main processor
    "DisplayPanelProcessor",
    "ProcessingConfig",
    "ProcessingResult",

    # Components
    "ImageCropper",
    "CropRegion",
    "ROIDetector",
    "AdaptiveROIDetector",
    "DetectedROI",
    "PerspectiveWarper",
    "WarpResult",
    "AreaSumResizer",
    "ResizeResult",
    "ImageNormalizer",
    "NormalizationResult",
]
