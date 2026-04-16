# -*- coding: utf-8 -*-
"""
ROI Processing Pipeline Runner
Runs all src modules in order from 00 to 06
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src import DisplayPanelProcessor, ProcessingConfig

def main():
    # Configure processing
    config = ProcessingConfig(
        crop_x_start=1700,
        crop_y_start=300,
        crop_x_end=11700,
        crop_y_end=9900,
        use_crop=True,
        roi_threshold=50,
        morph_kernel_size=51,
        display_width=2412,
        display_height=2288,
        output_bit_depth=16,
        save_intermediates=True
    )

    # Initialize processor (uses all modules 00-05)
    processor = DisplayPanelProcessor(config)

    # Input image path
    image_path = os.path.join(project_root, "data", "G32_cal.tif")

    # Output directory
    output_dir = os.path.join(project_root, "output")

    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print()

    # Run the full pipeline
    result = processor.process(image_path)

    # Save results
    if result.normalized_image is not None:
        saved_files = processor.save_results(result, output_dir)
        processor.save_visualization(result, output_dir)
        print("\n" + "=" * 60)
        print("Saved files:")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")
    else:
        print("\nWARNING: Processing failed - no output generated")

if __name__ == "__main__":
    main()
