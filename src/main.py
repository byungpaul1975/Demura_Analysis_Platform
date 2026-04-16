# -*- coding: utf-8 -*-
"""
Main script for Display Panel ROI Processing

This script demonstrates how to use the OOP-based processing pipeline.

Usage:
    python src/main.py
test
Or with custom configuration:
    python src/main.py --input data/image.tif --output output/ --width 2412 --height 2288
"""

import argparse
from pathlib import Path
import sys

# Add project root to path (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import DisplayPanelProcessor, ProcessingConfig


def main():
    """Main entry point for the processing pipeline."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Display Panel ROI Processing')
    parser.add_argument('--input', '-i', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'G32_cal.tif'),
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str,
                        default=str(PROJECT_ROOT / 'output'),
                        help='Output directory')
    parser.add_argument('--width', '-W', type=int,
                        default=2412,
                        help='Target display width')
    parser.add_argument('--height', '-H', type=int,
                        default=2288,
                        help='Target display height')
    parser.add_argument('--no-crop', action='store_true',
                        help='Disable image cropping')
    parser.add_argument('--crop-x-start', type=int, default=1700,
                        help='Crop region X start (default: 1700)')
    parser.add_argument('--crop-y-start', type=int, default=300,
                        help='Crop region Y start (default: 300)')
    parser.add_argument('--crop-x-end', type=int, default=11700,
                        help='Crop region X end (default: 11700)')
    parser.add_argument('--crop-y-end', type=int, default=9900,
                        help='Crop region Y end (default: 9900)')
    parser.add_argument('--threshold', '-t', type=int,
                        default=50,
                        help='ROI detection threshold')
    parser.add_argument('--no-intermediates', action='store_true',
                        help='Do not save intermediate images')

    args = parser.parse_args()

    # Create configuration
    config = ProcessingConfig(
        crop_x_start=args.crop_x_start,
        crop_y_start=args.crop_y_start,
        crop_x_end=args.crop_x_end,
        crop_y_end=args.crop_y_end,
        use_crop=not args.no_crop,
        roi_threshold=args.threshold,
        display_width=args.width,
        display_height=args.height,
        save_intermediates=not args.no_intermediates
    )

    # Create processor
    processor = DisplayPanelProcessor(config)

    # Process image
    print(f"\nProcessing: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Target resolution: {args.width} x {args.height}")
    print()

    try:
        result = processor.process(args.input)

        # Save results
        print("\nSaving results...")
        saved_files = processor.save_results(result, args.output)

        # Save visualization
        processor.save_visualization(result, args.output)

        print("\n" + "=" * 60)
        print("All files saved successfully!")
        print("=" * 60)

        # Print statistics
        print("\nProcessing Statistics:")
        for key, value in result.stats.items():
            print(f"  {key}: {value}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
