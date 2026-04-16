# -*- coding: utf-8 -*-
"""
Final ROI Processing for Display Panel Images
Uses X/Y Crossing Method to find precise display boundaries
Outputs display at target resolution: 2412 x 2288 pixels
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from src.roi_detector import AdaptiveROIDetector, DetectedROI

# ============================================================================
# Configuration
# ============================================================================
DISPLAY_PITCH = 3.8  # Camera pixels per display pixel
TARGET_DISPLAY_WIDTH = 2412  # Target display width in pixels
TARGET_DISPLAY_HEIGHT = 2288  # Target display height in pixels
EDGE_OFFSET = 0  # No offset - use cross points directly for exact display boundary


def process_display_image(
    image_path: str,
    output_dir: str = 'output',
    display_pitch: float = DISPLAY_PITCH,
    target_width: int = TARGET_DISPLAY_WIDTH,
    target_height: int = TARGET_DISPLAY_HEIGHT,
    edge_offset: int = EDGE_OFFSET,
    visualize: bool = True
) -> dict:
    """
    Process display panel image and extract display region.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        display_pitch: Camera pixels per display pixel
        target_width: Target display width in pixels
        target_height: Target display height in pixels
        edge_offset: Offset from cross point (0 = exact boundary)
        visualize: Whether to save visualization images
    
    Returns:
        Dictionary with processing results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # ========================================================================
    # Step 1: Load Image
    # ========================================================================
    print('='*70)
    print('DISPLAY PANEL ROI PROCESSING')
    print('='*70)
    
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f'\n[Step 1] Image Loaded')
    print(f'  Path: {image_path}')
    print(f'  Shape: {img.shape}')
    print(f'  Dtype: {img.dtype}')
    print(f'  Range: {img.min()} ~ {img.max()}')
    
    # ========================================================================
    # Step 2: Detect ROI using X/Y Crossing Method
    # ========================================================================
    print(f'\n[Step 2] ROI Detection (X/Y Crossing Method)')
    
    detector = AdaptiveROIDetector(display_pixel_pitch=(display_pitch, display_pitch))
    
    # Use edge_offset=0 to get exact display boundary (cross points)
    roi = detector.detect_by_crossing(img, edge_offset=edge_offset, search_radius=30)
    
    if roi is None:
        raise ValueError("ROI detection failed")
    
    cross_points = detector.get_cross_points()
    corner_names = ['TL', 'TR', 'BR', 'BL']
    
    print(f'\n  Detected Corners (Cross Points = Display Boundary):')
    print(f'  {"Corner":<8} {"Position (x, y)":<25}')
    print('  ' + '-'*40)
    for idx, corner_name in enumerate(corner_names):
        cp = cross_points.get(corner_name, roi.corners[idx])
        print(f'  {corner_name:<8} ({cp[0]:>8.1f}, {cp[1]:>8.1f})')
    
    # Calculate detected display size
    detected_width = roi.width / display_pitch
    detected_height = roi.height / display_pitch
    
    print(f'\n  Detected ROI:')
    print(f'    Camera pixels: {roi.width:.1f} x {roi.height:.1f}')
    print(f'    Display pixels: {detected_width:.1f} x {detected_height:.1f}')
    print(f'    Angle: {roi.angle:.3f} degrees')
    
    # ========================================================================
    # Step 3: Apply Perspective Transform (Warp)
    # ========================================================================
    print(f'\n[Step 3] Perspective Transform')
    
    # Source points (detected cross points = display boundary)
    src_points = np.array([
        cross_points['TL'],
        cross_points['TR'],
        cross_points['BR'],
        cross_points['BL']
    ], dtype=np.float32)
    
    # Destination points (rectified rectangle)
    # Use detected size for intermediate warp
    warp_width = int(roi.width)
    warp_height = int(roi.height)
    
    dst_points = np.array([
        [0, 0],
        [warp_width - 1, 0],
        [warp_width - 1, warp_height - 1],
        [0, warp_height - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply warp
    warped = cv2.warpPerspective(
        img, M, (warp_width, warp_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f'  Warped image size: {warped.shape[1]} x {warped.shape[0]} camera pixels')
    
    # ========================================================================
    # Step 4: Resize to Target Display Resolution
    # ========================================================================
    print(f'\n[Step 4] Resize to Display Resolution')
    
    # Resize to target display resolution
    final_display = cv2.resize(
        warped,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA  # Best for downscaling
    )
    
    print(f'  Target size: {target_width} x {target_height} display pixels')
    print(f'  Final output size: {final_display.shape[1]} x {final_display.shape[0]}')
    
    # ========================================================================
    # Step 5: Normalize and Save Results
    # ========================================================================
    print(f'\n[Step 5] Save Results')
    
    # Save as 16-bit TIFF
    output_tiff = output_path / 'display_extracted.tif'
    cv2.imwrite(str(output_tiff), final_display)
    print(f'  Saved: {output_tiff}')
    
    # Save normalized 8-bit PNG for preview
    if final_display.dtype == np.uint16:
        display_8bit = (final_display / 256).astype(np.uint8)
    else:
        display_8bit = final_display.copy()
    
    output_png = output_path / 'display_extracted_preview.png'
    cv2.imwrite(str(output_png), display_8bit)
    print(f'  Saved: {output_png}')
    
    # ========================================================================
    # Step 6: Visualization
    # ========================================================================
    if visualize:
        print(f'\n[Step 6] Visualization')
        
        # Visualization 1: ROI Detection Result
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original with ROI overlay
        ax1 = axes[0]
        img_display = (img / img.max() * 255).astype(np.uint8)
        ax1.imshow(img_display, cmap='gray')
        
        # Draw ROI polygon using cross points
        roi_polygon = np.array([
            cross_points['TL'],
            cross_points['TR'],
            cross_points['BR'],
            cross_points['BL'],
            cross_points['TL']
        ])
        ax1.plot(roi_polygon[:, 0], roi_polygon[:, 1], 'r-', linewidth=2, label='ROI')
        
        # Mark corners
        colors = ['cyan', 'magenta', 'yellow', 'lime']
        for idx, corner_name in enumerate(corner_names):
            cp = cross_points[corner_name]
            ax1.plot(cp[0], cp[1], '*', color=colors[idx], markersize=15,
                    markeredgecolor='white', markeredgewidth=1)
            ax1.text(cp[0]+30, cp[1], corner_name, fontsize=12, color='white',
                    fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
        
        ax1.set_title(f'Original Image with ROI\n({img.shape[1]} x {img.shape[0]} camera pixels)')
        ax1.legend()
        
        # Final display output
        ax2 = axes[1]
        ax2.imshow(display_8bit, cmap='gray')
        ax2.set_title(f'Extracted Display\n({target_width} x {target_height} display pixels)')
        
        plt.tight_layout()
        vis_path = output_path / 'roi_processing_result.png'
        plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
        print(f'  Saved: {vis_path}')
        
        # Visualization 2: Corner Details
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 16))
        axes2 = axes2.flatten()
        
        half = 20
        for idx, corner_name in enumerate(corner_names):
            cp = cross_points[corner_name]
            cx, cy = int(cp[0]), int(cp[1])
            
            y1 = max(0, cy - half)
            y2 = min(img.shape[0], cy + half)
            x1 = max(0, cx - half)
            x2 = min(img.shape[1], cx + half)
            
            region = img[y1:y2, x1:x2].copy()
            
            ax = axes2[idx]
            ax.imshow(region, cmap='gray', interpolation='nearest')
            ax.set_title(f'{corner_name} Corner\n({cx}, {cy})', fontsize=14)
            
            # Draw display pixel grid
            for gx in np.arange(0, 40, display_pitch):
                ax.axvline(x=gx, color='yellow', linestyle='-', linewidth=0.5, alpha=0.5)
            for gy in np.arange(0, 40, display_pitch):
                ax.axhline(y=gy, color='yellow', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Mark cross point
            rel_x = cp[0] - x1
            rel_y = cp[1] - y1
            ax.plot(rel_x, rel_y, 'w*', markersize=20, markeredgecolor='red', markeredgewidth=1.5)
            ax.axhline(y=rel_y, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(x=rel_x, color='lime', linestyle='--', linewidth=1, alpha=0.7)
            
            ax.set_xlim(0, 40)
            ax.set_ylim(40, 0)
        
        plt.suptitle('Display Boundary Corners (X/Y Crossing Points)\n'
                    f'Display Pitch: {display_pitch} camera pixels/display pixel', fontsize=14)
        plt.tight_layout()
        corners_path = output_path / 'corner_details.png'
        plt.savefig(str(corners_path), dpi=200, bbox_inches='tight')
        print(f'  Saved: {corners_path}')
        
        plt.close('all')
    
    # ========================================================================
    # Summary
    # ========================================================================
    print('\n' + '='*70)
    print('PROCESSING COMPLETE')
    print('='*70)
    print(f'\n  Input:')
    print(f'    Image: {image_path}')
    print(f'    Size: {img.shape[1]} x {img.shape[0]} camera pixels')
    print(f'\n  Detection:')
    print(f'    Display pitch: {display_pitch} camera pixels/display pixel')
    print(f'    ROI angle: {roi.angle:.3f} degrees')
    print(f'\n  Output:')
    print(f'    Display size: {target_width} x {target_height} pixels')
    print(f'    Files saved to: {output_path}')
    
    return {
        'roi': roi,
        'cross_points': cross_points,
        'warped': warped,
        'final_display': final_display,
        'input_shape': img.shape,
        'output_shape': final_display.shape,
        'display_width': target_width,
        'display_height': target_height
    }


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    # Process image
    result = process_display_image(
        image_path='data/G32_cal.tif',
        output_dir='output',
        display_pitch=DISPLAY_PITCH,
        target_width=TARGET_DISPLAY_WIDTH,
        target_height=TARGET_DISPLAY_HEIGHT,
        edge_offset=0,  # Use cross points directly (exact display boundary)
        visualize=True
    )
    
    print(f'\nFinal Display Output: {result["display_width"]} x {result["display_height"]} pixels')
