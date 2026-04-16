# -*- coding: utf-8 -*-
"""
Step-by-step visualization image generator for the ROI pipeline.
Runs the pipeline and saves intermediate images to docs/images/.
"""
import sys, os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    ImageCropper, CropRegion, ROIDetector,
    PerspectiveWarper, AreaSumResizer, ImageNormalizer,
    DisplayPanelProcessor, ProcessingConfig
)

IMG_DIR = Path(__file__).parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    path = IMG_DIR / name
    fig.savefig(str(path), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return str(path)


def _to8(img):
    mx = max(float(img.max()), 1)
    return np.clip(img / mx * 255, 0, 255).astype(np.uint8)


def vis_step0(image):
    """원본 이미지 + 히스토그램"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(_to8(image), cmap='gray')
    ax.set_title(f"Original Image: {image.shape[1]}x{image.shape[0]}\n"
                 f"dtype={image.dtype}, range=[{image.min()}, {image.max()}]",
                 fontsize=12, fontweight='bold')
    p1 = _save(fig, "step0_original.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(image.flatten(), bins=200, color='steelblue', alpha=0.8)
    axes[0].set_title("Pixel Value Histogram", fontweight='bold')
    axes[0].set_xlabel("Pixel Value"); axes[0].set_ylabel("Frequency")
    axes[0].axvline(x=50, color='red', ls='--', label='Threshold=50'); axes[0].legend()
    axes[1].axis('off')
    info = (f"Resolution: {image.shape[1]} x {image.shape[0]}\n"
            f"Data Type: {image.dtype}\nMin: {image.min()}, Max: {image.max()}\n"
            f"Mean: {image.mean():.1f}, Std: {image.std():.1f}\nFile: G32_cal.tif")
    axes[1].text(0.1, 0.7, info, fontsize=12, transform=axes[1].transAxes, va='top',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    plt.tight_layout()
    p2 = _save(fig, "step0_histogram.png")
    return [p1, p2]


def vis_step1(original, cropped, config):
    """크롭 전후 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(_to8(original), cmap='gray')
    r = plt.Rectangle((config.crop_x_start, config.crop_y_start),
                       config.crop_x_end - config.crop_x_start,
                       config.crop_y_end - config.crop_y_start,
                       lw=2, ec='red', fc='none', ls='--')
    axes[0].add_patch(r)
    axes[0].set_title(f"Original ({original.shape[1]}x{original.shape[0]})\nRed = Crop region", fontweight='bold')
    axes[1].imshow(_to8(cropped), cmap='gray')
    axes[1].set_title(f"Cropped ({cropped.shape[1]}x{cropped.shape[0]})\n"
                      f"({config.crop_x_start},{config.crop_y_start})→({config.crop_x_end},{config.crop_y_end})",
                      fontweight='bold')
    plt.tight_layout()
    return [_save(fig, "step1_crop.png")]


def vis_step2(cropped, roi, detector):
    """ROI 검출: 이진화, 모폴로지, 컨투어, 코너"""
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
    binary = (gray > detector.threshold).astype(np.uint8) * 255
    mask = detector.get_binary_mask()

    # Binarization steps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(_to8(gray), cmap='gray')
    axes[0].set_title(f"Grayscale\nThreshold = {detector.threshold}", fontweight='bold')
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f"Binary (> {detector.threshold})", fontweight='bold')
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Morphology ({detector.morph_kernel_size}x{detector.morph_kernel_size})\n+ Hole fill",
                      fontweight='bold')
    plt.tight_layout()
    p1 = _save(fig, "step2_binary.png")

    # Contour + Corners
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    d1 = cv2.cvtColor(_to8(cropped), cv2.COLOR_GRAY2BGR) if len(_to8(cropped).shape) == 2 else _to8(cropped).copy()
    if roi:
        cv2.drawContours(d1, [roi.contour], -1, (0, 255, 0), 3)
    axes[0].imshow(cv2.cvtColor(d1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Largest Contour (green)\nArea = {roi.area:,.0f} px²", fontweight='bold')

    d2 = cv2.cvtColor(_to8(cropped), cv2.COLOR_GRAY2BGR) if len(_to8(cropped).shape) == 2 else _to8(cropped).copy()
    if roi:
        labels = ['TL','TR','BR','BL']
        colors = [(255,0,0),(0,200,0),(0,0,255),(255,165,0)]
        for i in range(4):
            pt = tuple(roi.corners[i].astype(int))
            cv2.circle(d2, pt, 12, colors[i], -1)
            cv2.putText(d2, labels[i], (pt[0]+15, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[i], 3)
            pt2 = tuple(roi.corners[(i+1)%4].astype(int))
            cv2.line(d2, pt, pt2, (0,255,255), 2)
    axes[1].imshow(cv2.cvtColor(d2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"4 Corners + Edges\nTilt={roi.angle:.2f}°, {roi.width:.0f}x{roi.height:.0f}", fontweight='bold')
    plt.tight_layout()
    p2 = _save(fig, "step2_corners.png")

    # Corner table
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    ax.set_title("Corner Coordinates & ROI Info", fontweight='bold', fontsize=14, pad=15)
    tdata = [["Corner","X","Y"],
             ["TL",f"{roi.corners[0][0]:.1f}",f"{roi.corners[0][1]:.1f}"],
             ["TR",f"{roi.corners[1][0]:.1f}",f"{roi.corners[1][1]:.1f}"],
             ["BR",f"{roi.corners[2][0]:.1f}",f"{roi.corners[2][1]:.1f}"],
             ["BL",f"{roi.corners[3][0]:.1f}",f"{roi.corners[3][1]:.1f}"]]
    t = ax.table(cellText=tdata, loc='center', cellLoc='center', colWidths=[.25,.25,.25])
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 2)
    for i in range(5):
        for j in range(3):
            c = t[i, j]
            if i == 0:
                c.set_facecolor('#003C71'); c.set_text_props(color='white', fontweight='bold')
            else:
                c.set_facecolor('#F0F2F5' if i%2==0 else 'white')
    ax.text(0.5, 0.05, f"Width={roi.width:.1f}  Height={roi.height:.1f}  Tilt={roi.angle:.4f}°  Area={roi.area:,.0f}px²",
            fontsize=10, transform=ax.transAxes, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    p3 = _save(fig, "step2_table.png")
    return [p1, p2, p3]


def vis_step3(cropped, roi, warped, warp_result):
    """Perspective warp 전후"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(_to8(cropped), cmap='gray')
    if roi:
        c = roi.corners
        for i in range(4):
            axes[0].plot([c[i][0], c[(i+1)%4][0]], [c[i][1], c[(i+1)%4][1]], 'r-', lw=2)
            axes[0].plot(c[i][0], c[i][1], 'ro', ms=8)
    axes[0].set_title(f"Before Warp (tilt={roi.angle:.2f}°)", fontweight='bold')
    axes[1].imshow(_to8(warped), cmap='gray')
    axes[1].set_title(f"After Perspective Warp\n{warped.shape[1]}x{warped.shape[0]} | INTER_NEAREST", fontweight='bold')
    plt.tight_layout()
    p1 = _save(fig, "step3_warp.png")

    # Matrix + mapping
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title("Perspective Transform Details", fontweight='bold', fontsize=14, pad=15)
    M = warp_result.transform_matrix
    txt = "Transform Matrix (3x3):\n"
    for r in range(3):
        txt += f"  [{M[r,0]:+10.4f}  {M[r,1]:+10.4f}  {M[r,2]:+10.4f}]\n"
    txt += f"\nCorner Mapping (Source → Dest):\n"
    for i, lbl in enumerate(['TL','TR','BR','BL']):
        s = warp_result.src_corners[i]; d = warp_result.dst_corners[i]
        txt += f"  {lbl}: ({s[0]:.1f},{s[1]:.1f}) → ({d[0]:.1f},{d[1]:.1f})\n"
    txt += f"\nInterpolation: INTER_NEAREST (preserves pixel values)"
    ax.text(0.5, 0.5, txt, fontsize=11, ha='center', va='center', family='monospace',
            transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.8', facecolor='#E3F2FD'))
    p2 = _save(fig, "step3_matrix.png")
    return [p1, p2]


def vis_step4(warped, resized, resize_result, config):
    """Area Sum Resize"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(_to8(warped), cmap='gray')
    axes[0].set_title(f"Warped (Camera Res.)\n{warped.shape[1]}x{warped.shape[0]}", fontweight='bold')
    axes[1].imshow(_to8(resized), cmap='gray')
    axes[1].set_title(f"Resized (Display Res.)\n{config.display_width}x{config.display_height}", fontweight='bold')
    plt.tight_layout()
    p1 = _save(fig, "step4_resize.png")

    # Concept diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Area-Sum Resize Concept", fontweight='bold', fontsize=14)
    for i in range(5):
        ax.plot([2+i*1.8, 2+i*1.8], [1, 8.2], 'k-', lw=0.5, alpha=0.5)
    for j in range(5):
        ax.plot([2, 9.2], [1+j*1.8, 1+j*1.8], 'k-', lw=0.5, alpha=0.5)
    ax.add_patch(plt.Rectangle((2, 2.8), 7.2, 3.6, lw=3, ec='red', fc='red', alpha=0.12))
    ax.text(5.6, 0.3, "Camera Pixel Grid", fontsize=11, ha='center', fontweight='bold')
    ax.text(5.6, 9.5, f"1 Display Pixel ≈ {resize_result.scale_x:.1f}x{resize_result.scale_y:.1f} cam px",
            fontsize=10, ha='center', color='red', fontweight='bold')
    ax.annotate("MEAN", xy=(10.5, 4.6), xytext=(12, 4.6), fontsize=12, fontweight='bold',
                color='blue', arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.add_patch(plt.Rectangle((12.5, 3.6), 2, 2, lw=2, ec='blue', fc='blue', alpha=0.15))
    ax.text(13.5, 4.6, "1px", fontsize=12, ha='center', va='center', fontweight='bold', color='blue')
    info = f"Scale: {resize_result.scale_x:.4f} x {resize_result.scale_y:.4f}"
    ax.text(13.5, 8.5, info, fontsize=9, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    p2 = _save(fig, "step4_concept.png")

    # Zoom compare
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cy, cx = warped.shape[0]//2, warped.shape[1]//2
    rw = warped[cy-50:cy+50, cx-50:cx+50]
    axes[0].imshow(_to8(rw), cmap='gray', interpolation='nearest')
    axes[0].set_title("Warped Center 100x100\n(Camera pixels)", fontweight='bold')
    cy2, cx2 = resized.shape[0]//2, resized.shape[1]//2
    rr = resized[cy2-15:cy2+15, cx2-15:cx2+15]
    axes[1].imshow(_to8(rr), cmap='gray', interpolation='nearest')
    axes[1].set_title("Resized Center 30x30\n(Display pixels)", fontweight='bold')
    plt.tight_layout()
    p3 = _save(fig, "step4_zoom.png")
    return [p1, p2, p3]


def vis_step5(resized, normalized, norm_result):
    """Normalization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].hist(resized.flatten(), bins=200, color='steelblue', alpha=0.8)
    axes[0,0].set_title("Before Normalization", fontweight='bold')
    axes[0,0].set_xlabel("Pixel Value")
    axes[0,1].hist(normalized.flatten(), bins=200, color='coral', alpha=0.8)
    axes[0,1].set_title("After 16-bit Normalization (0~65535)", fontweight='bold')
    axes[0,1].set_xlabel("Pixel Value (16-bit)")
    axes[1,0].imshow(_to8(normalized), cmap='gray')
    axes[1,0].set_title(f"Normalized {normalized.shape[1]}x{normalized.shape[0]} | {normalized.dtype}", fontweight='bold')
    ax = axes[1,1]; ax.axis('off')
    info = (f"Input:  [{norm_result.original_min:.4f}, {norm_result.original_max:.4f}]\n"
            f"Output: [{norm_result.normalized_min}, {norm_result.normalized_max}]\n"
            f"Bit Depth: {norm_result.bit_depth}-bit (max={2**norm_result.bit_depth-1})\n\n"
            f"Formula:\n  out = (px - min)/(max - min) × 65535")
    ax.text(0.1, 0.7, info, fontsize=12, transform=ax.transAxes, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0'))
    plt.tight_layout()
    return [_save(fig, "step5_norm.png")]


def vis_step6(original, cropped, warped, resized, normalized):
    """파이프라인 전체 요약"""
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    imgs = [(original,"Original"),(cropped,"Cropped"),(warped,"Warped"),
            (resized,"Resized"),(normalized,"Normalized")]
    for i,(img,title) in enumerate(imgs):
        axes[i].imshow(_to8(img), cmap='gray')
        axes[i].set_title(f"{title}\n{img.shape[1]}x{img.shape[0]}", fontsize=10, fontweight='bold')
        axes[i].set_xticks([]); axes[i].set_yticks([])
    plt.tight_layout()
    p1 = _save(fig, "step6_summary.png")

    # Architecture
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off'); ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.set_title("Pipeline Architecture (src/ modules)", fontsize=16, fontweight='bold', pad=20)
    mods = [
(2,6.5,"1_utils.py","Utility\nFunctions","#E3F2FD"),
        (2,4.5,"2_roi_detector.py","ROI Detection","#E8F5E9"),
        (5.5,4.5,"3_image_cropper.py","Image Crop","#FFF3E0"),
        (9,6.5,"4_perspective_warper.py","Perspective\nWarp","#FCE4EC"),
        (9,4.5,"5_area_sum_resizer.py","Area-Sum\nResize","#E0F7FA"),
        (9,2.5,"6_image_normalizer.py","16-bit\nNormalize","#F3E5F5"),
        (5.5,2.5,"7_display_panel_\nprocessor.py","Pipeline\nOrchestrator","#E8EAF6"),
    ]
    for x,y,name,desc,col in mods:
        ax.add_patch(plt.Rectangle((x-1.3,y-0.65),2.6,1.3,lw=1.5,ec='#333',fc=col,alpha=0.9,zorder=2))
        ax.text(x,y+0.2,name,fontsize=7.5,ha='center',va='center',fontweight='bold',zorder=3)
        ax.text(x,y-0.25,desc,fontsize=7,ha='center',va='center',color='#555',zorder=3)
    arr = dict(arrowstyle='->',color='#1877F2',lw=2)
    ax.annotate('',xy=(3.2,4.5),xytext=(2.7,5.85),arrowprops=arr)
    ax.annotate('',xy=(4.2,4.5),xytext=(3.2,4.5),arrowprops=arr)
    ax.annotate('',xy=(7.7,6.5),xytext=(5.8,4.8),arrowprops=arr)
    ax.annotate('',xy=(9,5.85),xytext=(9,5.2),arrowprops=arr)
    ax.annotate('',xy=(9,3.85),xytext=(9,3.15),arrowprops=arr)
    p2 = _save(fig, "step6_arch.png")
    return [p1, p2]


def run_pipeline_and_generate_images():
    """파이프라인 실행 + 모든 중간 이미지 생성"""
    config = ProcessingConfig(
        crop_x_start=1700, crop_y_start=300,
        crop_x_end=11700, crop_y_end=9900,
        use_crop=True, roi_threshold=50, morph_kernel_size=51,
        display_width=2412, display_height=2288,
        output_bit_depth=16, save_intermediates=True
    )
    image_path = str(PROJECT_ROOT / "data" / "G32_cal.tif")
    print(f"Loading: {image_path}")
    original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    print(f"  Shape={original.shape}, dtype={original.dtype}")

    all_images = {}

    # Step 0
    all_images['step0'] = vis_step0(original)

    # Step 1: Crop
    cropper = ImageCropper(CropRegion(config.crop_x_start, config.crop_y_start,
                                       config.crop_x_end, config.crop_y_end))
    cropped = cropper.crop(original)
    all_images['step1'] = vis_step1(original, cropped, config)

    # Step 2: ROI Detection
    detector = ROIDetector(threshold=config.roi_threshold, morph_kernel_size=config.morph_kernel_size)
    roi = detector.detect(cropped)
    all_images['step2'] = vis_step2(cropped, roi, detector)

    # Step 3: Perspective Warp
    warper = PerspectiveWarper(interpolation=cv2.INTER_NEAREST)
    warp_result = warper.warp(cropped, roi.corners)
    warped = warp_result.image
    all_images['step3'] = vis_step3(cropped, roi, warped, warp_result)

    # Step 4: Area Sum Resize
    resizer = AreaSumResizer(show_progress=True)
    resize_result = resizer.resize(warped, (config.display_width, config.display_height))
    resized = resize_result.image
    all_images['step4'] = vis_step4(warped, resized, resize_result, config)

    # Step 5: Normalize
    normalizer = ImageNormalizer(bit_depth=16)
    norm_result = normalizer.normalize(resized)
    normalized = norm_result.image
    all_images['step5'] = vis_step5(resized, normalized, norm_result)

    # Step 6: Summary
    all_images['step6'] = vis_step6(original, cropped, warped, resized, normalized)

    pipeline_data = {
        'config': config, 'roi': roi, 'warp_result': warp_result,
        'resize_result': resize_result, 'norm_result': norm_result,
        'original_shape': original.shape, 'cropped_shape': cropped.shape,
        'warped_shape': warped.shape, 'resized_shape': resized.shape,
        'normalized_shape': normalized.shape,
    }
    print(f"\nAll images saved to: {IMG_DIR}")
    return all_images, pipeline_data


if __name__ == "__main__":
    run_pipeline_and_generate_images()
