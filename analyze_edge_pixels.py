# -*- coding: utf-8 -*-
"""
Edge ROI Optimization based on Display Pixel Grid
Camera pixels: 3x3 to 4x4 camera pixels = 1 display pixel
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.roi_detector import AdaptiveROIDetector

# Load image
img = cv2.imread(str(Path('data/G32_cal.tif')), cv2.IMREAD_UNCHANGED)
print(f'Image: {img.shape}, dtype: {img.dtype}')

# Detect ROI
detector = AdaptiveROIDetector(initial_threshold=200, corner_region_size=60)
roi = detector.detect_adaptive(img)
corners = roi.corners  # TL, TR, BR, BL

print(f'ROI Corners: TL={corners[0]}, TR={corners[1]}, BR={corners[2]}, BL={corners[3]}')

# Calculate edge centers
edge_centers = {
    'Top': ((corners[0] + corners[1]) / 2).astype(int),
    'Right': ((corners[1] + corners[2]) / 2).astype(int),
    'Bottom': ((corners[2] + corners[3]) / 2).astype(int),
    'Left': ((corners[3] + corners[0]) / 2).astype(int)
}

print('\nEdge Centers:')
for name, center in edge_centers.items():
    print(f'  {name}: ({center[0]}, {center[1]})')

# Extract 30x30 regions at each edge center
half = 15
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

edge_data = {}

for idx, (name, center) in enumerate(edge_centers.items()):
    cx, cy = center[0], center[1]

    # Extract region
    y1, y2 = max(0, cy - half), min(img.shape[0], cy + half)
    x1, x2 = max(0, cx - half), min(img.shape[1], cx + half)
    region = img[y1:y2, x1:x2].copy()

    edge_data[name] = {
        'center': (cx, cy),
        'region': region,
        'bounds': (x1, x2, y1, y2)
    }

    # Top row: Original regions
    ax1 = axes[0, idx]
    im1 = ax1.imshow(region, cmap='gray')
    ax1.set_title(f'{name} Edge\nCenter: ({cx}, {cy})')
    ax1.axhline(y=half, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(x=half, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Analyze pixel pattern
    # Display pixel = 3x3 to 4x4 camera pixels
    # Let's detect the grid pattern

    # Calculate local gradient to find pixel boundaries
    sobel_x = cv2.Sobel(region.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(region.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    ax2 = axes[1, idx]
    im2 = ax2.imshow(gradient_mag, cmap='hot')
    ax2.set_title(f'{name} Gradient\nDisplay pixel grid detection')
    ax2.axhline(y=half, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=half, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Print statistics
    print(f'\n{name} Edge Analysis:')
    print(f'  Region: x=[{x1}:{x2}], y=[{y1}:{y2}]')
    print(f'  Pixel range: [{region.min()}, {region.max()}]')
    print(f'  Mean: {region.mean():.1f}')
    print(f'  Gradient max: {gradient_mag.max():.1f}')

plt.suptitle('Edge Centers - 30x30 Pixel Regions\n(Camera pixels, 3-4 camera px = 1 display px)', fontsize=14)
plt.tight_layout()
plt.savefig('output/edge_centers_30x30.png', dpi=150, bbox_inches='tight')
print('\nSaved: output/edge_centers_30x30.png')

# Detailed analysis: Find display pixel grid spacing
print('\n' + '='*70)
print('DISPLAY PIXEL GRID ANALYSIS')
print('='*70)
print('Expected: 3x3 to 4x4 camera pixels = 1 display pixel')
print()

fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))

for idx, (name, data) in enumerate(edge_data.items()):
    region = data['region'].astype(np.float32)
    cx, cy = data['center']

    # Horizontal profile (across edge)
    h_profile = region[half, :] if half < region.shape[0] else region[region.shape[0]//2, :]

    # Vertical profile (across edge)
    v_profile = region[:, half] if half < region.shape[1] else region[:, region.shape[1]//2]

    # Plot horizontal profile
    ax1 = axes2[0, idx]
    ax1.plot(h_profile, 'b-', linewidth=1.5, label='Horizontal')
    ax1.set_title(f'{name} - Horizontal Profile')
    ax1.set_xlabel('X (camera pixels)')
    ax1.set_ylabel('Intensity')
    ax1.grid(True, alpha=0.3)

    # Mark every 3-4 pixels (display pixel boundaries)
    for i in range(0, len(h_profile), 3):
        ax1.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
    for i in range(0, len(h_profile), 4):
        ax1.axvline(x=i, color='orange', linestyle=':', alpha=0.3)

    # Plot vertical profile
    ax2 = axes2[1, idx]
    ax2.plot(v_profile, 'r-', linewidth=1.5, label='Vertical')
    ax2.set_title(f'{name} - Vertical Profile')
    ax2.set_xlabel('Y (camera pixels)')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, alpha=0.3)

    # Mark every 3-4 pixels
    for i in range(0, len(v_profile), 3):
        ax2.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
    for i in range(0, len(v_profile), 4):
        ax2.axvline(x=i, color='orange', linestyle=':', alpha=0.3)

    # Detect periodic pattern using FFT
    if len(h_profile) > 10:
        fft = np.fft.fft(h_profile - h_profile.mean())
        freqs = np.fft.fftfreq(len(h_profile))

        # Find dominant frequency (excluding DC)
        fft_mag = np.abs(fft[1:len(fft)//2])
        if len(fft_mag) > 0:
            dominant_idx = np.argmax(fft_mag) + 1
            dominant_freq = freqs[dominant_idx]
            if dominant_freq > 0:
                period = 1 / dominant_freq
                print(f'{name} Edge - Detected period: {period:.2f} camera pixels/display pixel')

plt.suptitle('Edge Profiles with Display Pixel Grid\n(Gray: 3px grid, Orange: 4px grid)', fontsize=14)
plt.tight_layout()
plt.savefig('output/edge_profiles_grid.png', dpi=150, bbox_inches='tight')
print('\nSaved: output/edge_profiles_grid.png')

# Create zoomed view with display pixel grid overlay
print('\n' + '='*70)
print('DISPLAY PIXEL GRID OVERLAY')
print('='*70)

# Use average pixel pitch of 3.5 camera pixels per display pixel
DISPLAY_PIXEL_PITCH = 3.5

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 14))
axes3 = axes3.flatten()

for idx, (name, data) in enumerate(edge_data.items()):
    region = data['region']
    ax = axes3[idx]

    # Show region
    ax.imshow(region, cmap='gray', interpolation='nearest')
    ax.set_title(f'{name} Edge - Display Pixel Grid Overlay\n(Grid spacing: {DISPLAY_PIXEL_PITCH} camera px)')

    # Overlay display pixel grid (assuming 3.5 camera px per display px)
    for i in np.arange(0, region.shape[1], DISPLAY_PIXEL_PITCH):
        ax.axvline(x=i, color='yellow', linestyle='-', linewidth=0.5, alpha=0.6)
    for i in np.arange(0, region.shape[0], DISPLAY_PIXEL_PITCH):
        ax.axhline(y=i, color='yellow', linestyle='-', linewidth=0.5, alpha=0.6)

    # Mark center
    ax.axhline(y=half, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=half, color='red', linestyle='--', linewidth=1.5)
    ax.plot(half, half, 'r+', markersize=15, markeredgewidth=2)

    # Add pixel coordinate labels
    ax.set_xlabel('Camera X (pixels)')
    ax.set_ylabel('Camera Y (pixels)')

plt.tight_layout()
plt.savefig('output/edge_display_grid_overlay.png', dpi=200, bbox_inches='tight')
print('Saved: output/edge_display_grid_overlay.png')

# Calculate optimal edge offset to align with display pixel grid
print('\n' + '='*70)
print('OPTIMAL EDGE ALIGNMENT')
print('='*70)

for name, data in edge_data.items():
    cx, cy = data['center']

    # Calculate offset to nearest display pixel boundary
    x_offset = cx % DISPLAY_PIXEL_PITCH
    y_offset = cy % DISPLAY_PIXEL_PITCH

    # Optimal position (align to display pixel grid)
    if x_offset < DISPLAY_PIXEL_PITCH / 2:
        optimal_x = cx - x_offset
    else:
        optimal_x = cx + (DISPLAY_PIXEL_PITCH - x_offset)

    if y_offset < DISPLAY_PIXEL_PITCH / 2:
        optimal_y = cy - y_offset
    else:
        optimal_y = cy + (DISPLAY_PIXEL_PITCH - y_offset)

    print(f'{name} Edge:')
    print(f'  Current center: ({cx}, {cy})')
    print(f'  Grid offset: ({x_offset:.2f}, {y_offset:.2f}) camera px')
    print(f'  Optimal center: ({optimal_x:.1f}, {optimal_y:.1f})')
    print(f'  Adjustment: ({optimal_x - cx:.1f}, {optimal_y - cy:.1f}) camera px')
    print()
