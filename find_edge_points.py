# -*- coding: utf-8 -*-
"""
Find Final Edge Points for ROI - X/Y Crossing Point Algorithm
1. Find outermost X position (left/right edge of display)
2. Find outermost Y position (top/bottom edge of display)
3. The corner is where these X and Y cross
4. Move 2 pixels diagonally outward from that crossing point
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load image
img = cv2.imread(str(Path('data/G32_cal.tif')), cv2.IMREAD_UNCHANGED)
print(f'Image: {img.shape}, dtype: {img.dtype}')

# Display pixel pitch
DISPLAY_PITCH = 3.8
EDGE_OFFSET = 2

corner_names = ['TL', 'TR', 'BR', 'BL']

# Get approximate corners
gray = img.copy()
_, binary_global = cv2.threshold(
    (gray / gray.max() * 255).astype(np.uint8),
    0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

contours, _ = cv2.findContours(binary_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(largest)
box = cv2.boxPoints(rect)

def order_corners(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

approx_corners = order_corners(box)


def find_edge_crossing_point(image, approx_corner, corner_name, offset=2, search_radius=30):
    """
    Find edge point by finding X and Y crossing:
    1. Find outermost X edge of display
    2. Find outermost Y edge of display
    3. The crossing point is the corner
    4. Move 2 pixels outward diagonally
    """
    cx, cy = int(approx_corner[0]), int(approx_corner[1])

    y1 = max(0, cy - search_radius)
    y2 = min(image.shape[0], cy + search_radius)
    x1 = max(0, cx - search_radius)
    x2 = min(image.shape[1], cx + search_radius)

    region = image[y1:y2, x1:x2].astype(np.float32)
    region_h, region_w = region.shape

    # Normalize and threshold
    region_norm = ((region - region.min()) / (region.max() - region.min()) * 255).astype(np.uint8)
    _, binary = cv2.threshold(region_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find all bright pixels
    bright_pixels = np.where(binary > 0)

    if len(bright_pixels[0]) == 0:
        return np.array([cx, cy], dtype=np.float32), None, None, region, binary

    bright_y = bright_pixels[0]  # row indices
    bright_x = bright_pixels[1]  # column indices

    # Find outermost X and Y based on corner type
    if corner_name == 'TL':
        # TL: minimum X (leftmost) and minimum Y (topmost)
        edge_x = np.min(bright_x)  # leftmost bright pixel X
        edge_y = np.min(bright_y)  # topmost bright pixel Y
        # Cross point
        cross_local = np.array([edge_x, edge_y])
        # Move 2 pixels outward (toward top-left)
        roi_edge_local = cross_local - np.array([offset, offset])

    elif corner_name == 'TR':
        # TR: maximum X (rightmost) and minimum Y (topmost)
        edge_x = np.max(bright_x)  # rightmost bright pixel X
        edge_y = np.min(bright_y)  # topmost bright pixel Y
        # Cross point
        cross_local = np.array([edge_x, edge_y])
        # Move 2 pixels outward (toward top-right)
        roi_edge_local = cross_local + np.array([offset, -offset])

    elif corner_name == 'BR':
        # BR: maximum X (rightmost) and maximum Y (bottommost)
        edge_x = np.max(bright_x)  # rightmost bright pixel X
        edge_y = np.max(bright_y)  # bottommost bright pixel Y
        # Cross point
        cross_local = np.array([edge_x, edge_y])
        # Move 2 pixels outward (toward bottom-right)
        roi_edge_local = cross_local + np.array([offset, offset])

    elif corner_name == 'BL':
        # BL: minimum X (leftmost) and maximum Y (bottommost)
        edge_x = np.min(bright_x)  # leftmost bright pixel X
        edge_y = np.max(bright_y)  # bottommost bright pixel Y
        # Cross point
        cross_local = np.array([edge_x, edge_y])
        # Move 2 pixels outward (toward bottom-left)
        roi_edge_local = cross_local + np.array([-offset, offset])

    # Convert to global coordinates
    cross_global = np.array([cross_local[0] + x1, cross_local[1] + y1], dtype=np.float32)
    roi_edge_global = np.array([roi_edge_local[0] + x1, roi_edge_local[1] + y1], dtype=np.float32)

    # Also return the X and Y edge lines for visualization
    edge_info = {
        'edge_x': edge_x,
        'edge_y': edge_y,
        'x1': x1, 'y1': y1
    }

    return roi_edge_global, cross_global, edge_info, region, binary


# Find edge points
print('\n' + '='*70)
print('FINDING EDGE POINTS (X/Y Crossing Method)')
print('='*70)

edge_points = {}
cross_points = {}
edge_infos = {}
regions = {}
binaries = {}

for corner_name, approx_corner in zip(corner_names, approx_corners):
    roi_edge, cross_point, edge_info, region, binary = find_edge_crossing_point(
        gray, approx_corner, corner_name, EDGE_OFFSET
    )
    edge_points[corner_name] = roi_edge
    cross_points[corner_name] = cross_point
    edge_infos[corner_name] = edge_info
    regions[corner_name] = region
    binaries[corner_name] = binary

    print(f'\n{corner_name}:')
    print(f'  Approx corner: ({approx_corner[0]:.1f}, {approx_corner[1]:.1f})')
    print(f'  Edge X (local): {edge_info["edge_x"]}')
    print(f'  Edge Y (local): {edge_info["edge_y"]}')
    print(f'  Cross point (X,Y meet): ({cross_point[0]:.1f}, {cross_point[1]:.1f})')
    print(f'  ROI Edge (2px outward): ({roi_edge[0]:.1f}, {roi_edge[1]:.1f})')


# Visualize 30x30 regions with X/Y edge lines and star marker
print('\n' + '='*70)
print('VISUALIZING 30x30 EDGE REGIONS')
print('='*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

half = 15

for idx, corner_name in enumerate(corner_names):
    roi_edge = edge_points[corner_name]
    cross_point = cross_points[corner_name]
    edge_info = edge_infos[corner_name]

    cx, cy = int(roi_edge[0]), int(roi_edge[1])

    y1 = max(0, cy - half)
    y2 = min(img.shape[0], cy + half)
    x1 = max(0, cx - half)
    x2 = min(img.shape[1], cx + half)

    region = img[y1:y2, x1:x2].copy()

    ax = axes[idx]
    ax.imshow(region, cmap='gray', interpolation='nearest')
    ax.set_title(f'{corner_name} Edge Point\nROI: ({cx}, {cy})', fontsize=14)

    # Draw display pixel grid
    for gx in np.arange(0, 30, DISPLAY_PITCH):
        ax.axvline(x=gx, color='yellow', linestyle='-', linewidth=0.5, alpha=0.5)
    for gy in np.arange(0, 30, DISPLAY_PITCH):
        ax.axhline(y=gy, color='yellow', linestyle='-', linewidth=0.5, alpha=0.5)

    # Calculate relative positions within 30x30 view
    cross_rel_x = cross_point[0] - roi_edge[0] + half
    cross_rel_y = cross_point[1] - roi_edge[1] + half

    # Draw X edge line (vertical line at edge_x)
    edge_x_rel = cross_rel_x
    ax.axvline(x=edge_x_rel, color='lime', linestyle='-', linewidth=2, alpha=0.8, label='X edge')

    # Draw Y edge line (horizontal line at edge_y)
    edge_y_rel = cross_rel_y
    ax.axhline(y=edge_y_rel, color='cyan', linestyle='-', linewidth=2, alpha=0.8, label='Y edge')

    # Mark the crossing point with a STAR
    ax.plot(cross_rel_x, cross_rel_y, 'w*', markersize=25, markeredgecolor='black',
            markeredgewidth=1.5, label='X/Y Cross')

    # Mark the ROI edge point (red cross at center)
    ax.axhline(y=half, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=half, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(half, half, 'r+', markersize=20, markeredgewidth=3)

    ax.text(0.02, 0.98,
            f'ROI Edge: ({cx}, {cy})\n'
            f'Cross point: ({cross_point[0]:.0f}, {cross_point[1]:.0f})\n'
            f'Green=X edge, Cyan=Y edge\n'
            f'White star=Cross point',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Camera X (pixels)')
    ax.set_ylabel('Camera Y (pixels)')
    ax.set_xlim(0, 30)
    ax.set_ylim(30, 0)

plt.suptitle('ROI Edge Points (X/Y Crossing Method)\n'
             'White star = Cross point of X & Y edges, Red cross = ROI edge (2px outward)',
             fontsize=14)
plt.tight_layout()
plt.savefig('output/roi_edge_points_xy_cross.png', dpi=200, bbox_inches='tight')
print('\nSaved: output/roi_edge_points_xy_cross.png')


# Summary
print('\n' + '='*70)
print('FINAL EDGE POINTS')
print('='*70)
print(f'\n{"Corner":<8} {"ROI Edge (x, y)":<22} {"Cross Point (x, y)":<22} {"Offset"}')
print('-'*70)
for corner_name in corner_names:
    ep = edge_points[corner_name]
    cp = cross_points[corner_name]
    offset_x = ep[0] - cp[0]
    offset_y = ep[1] - cp[1]
    print(f'{corner_name:<8} ({ep[0]:>7.1f}, {ep[1]:>7.1f})    ({cp[0]:>7.1f}, {cp[1]:>7.1f})    ({offset_x:+.0f}, {offset_y:+.0f})')

width = np.linalg.norm(edge_points['TR'] - edge_points['TL'])
height = np.linalg.norm(edge_points['BL'] - edge_points['TL'])
display_w = round(width / DISPLAY_PITCH)
display_h = round(height / DISPLAY_PITCH)

print(f'\nROI Dimensions:')
print(f'  Camera pixels: {width:.1f} x {height:.1f}')
print(f'  Display pixels: {display_w} x {display_h}')
print(f'  Pixel pitch: {DISPLAY_PITCH}')
