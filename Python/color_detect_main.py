"""
Enhanced Color Detection System with Optimized Urban & Natural Object Detection
"""

import cv2
import numpy as np
import time

# Performance configuration
ENABLE_BLOB_DETECTION = False   # Turn off for better performance
ENABLE_EDGE_DETECTION = False   # Turn off for better performance
PROCESSING_SCALE = 0.75         # Scale down frames for faster processing
MIN_CONTOUR_AREA = 1000         # Minimum area for contour detection
DISPLAY_SCALE = 1.0             # Display frame scale (1.0 = original size)
OUTPUT_FPS_MULTIPLIER = 3       # Make output video play 3x faster than original

# Additional configuration parameters
COLOR_SENSITIVITY = 1.3         # Increased for better color detection
MIN_BLOB_AREA = 800             # Minimum blob area threshold
CLOSE_KERNEL_SIZE = 11          # Size for closing operation kernel
OPEN_KERNEL_SIZE = 5            # Size for opening operation kernel
ENABLE_HOLE_FILLING = True      # Enable filling holes in blobs
ENABLE_SHADOW_DETECTION = True  # Enable detection of darker versions of colors
PANEL_BG = (50, 50, 50)         # Dark gray background for panel
PANEL_ALPHA = 0.6               # Panel transparency

# HSV thresholds optimized for both cityscape and natural objects
color_defs = {
    # Base colors
    'Red':       {'hsv_lo': (0,   100, 80),  'hsv_hi': (10,  255, 255), 'bgr': (0, 0, 255)},
    'Red2':      {'hsv_lo': (170, 100, 80),  'hsv_hi': (180, 255, 255), 'bgr': (0, 0, 200)},
    'Orange':    {'hsv_lo': (5,   80,  80),  'hsv_hi': (25,  255, 255), 'bgr': (0, 165, 255)},
    'Yellow':    {'hsv_lo': (20,  80,  80),  'hsv_hi': (35,  255, 255), 'bgr': (0, 255, 255)},
    # 'Green':     {'hsv_lo': (36,  25,  25),  'hsv_hi': (85,  255, 255), 'bgr': (0, 255, 0)},
    'Blue':      {'hsv_lo': (94,  80,  2),   'hsv_hi': (126, 255, 255), 'bgr': (255, 0, 0)},
    'Purple':    {'hsv_lo': (130, 40,  40),  'hsv_hi': (155, 255, 255), 'bgr': (128, 0, 128)},
    'Pink':      {'hsv_lo': (145, 30, 150),  'hsv_hi': (170, 255, 255), 'bgr': (203, 192, 255)},
    'Brown':     {'hsv_lo': (5,   50,  20),  'hsv_hi': (30,  150, 150), 'bgr': (42, 42, 165)},
    'LightBlue': {'hsv_lo': (85,  40,  100), 'hsv_hi': (105, 180, 255), 'bgr': (255, 255, 0)},
    
    # Improved Gray/Black/White detection for urban landscapes
    'Gray':      {'hsv_lo': (0,   0,   40),  'hsv_hi': (180, 30,  200), 'bgr': (128, 128, 128)},
    'Black':     {'hsv_lo': (0,   0,   0),   'hsv_hi': (180, 255, 30),  'bgr': (0, 0, 0)},
    'Dark':  {'hsv_lo': (0,   0,   0), 'hsv_hi': (180,  60, 100), 'bgr': (50,50,50)},
    'White':     {'hsv_lo': (0,   0,   200), 'hsv_hi': (180, 30,  255), 'bgr': (255, 255, 255)},
    
    # Only true leafy greens (hue from 45°→85°, good saturation & brightness)
    'Green':     {'hsv_lo': (45,  80,  80),  'hsv_hi': (85,  255, 255), 'bgr': (0, 255, 0)},
    # If you still want a secondary “light” green for very pale leaves:
    'LightGreen':{'hsv_lo': (50, 100, 100), 'hsv_hi': (90, 180, 255), 'bgr': (144, 238, 144)},

    # Specialized colors
    # 'LightGreen':{'hsv_lo': (25,  40,  40),  'hsv_hi': (50,  255, 255), 'bgr': (144, 238, 144)},
    'DarkGreen': {'hsv_lo': (36,  40,  20),  'hsv_hi': (85,  180, 120), 'bgr': (0, 100, 0)},
    'DarkBrown': {'hsv_lo': (0,   15,  0),   'hsv_hi': (20,  150, 100), 'bgr': (25, 25, 100)},
}

def enhance_mask(mask, min_area=MIN_BLOB_AREA):
    """Enhanced morphological processing for better blob detection"""
    if np.sum(mask) == 0:  # Skip empty masks
        return mask
        
    # Remove small noise
    kernel_open = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
    kernel_close = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)
    
    # First remove small dots (noise)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Close small holes
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)
    
    # Fill any remaining holes if enabled
    mask_filled = mask_closed.copy()
    if ENABLE_HOLE_FILLING:
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_filled, contours, -1, 255, -1)  # Fill all contours
    
    # Final cleanup - remove small blobs
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled)
    mask_final = np.zeros_like(mask_filled)
    
    # Start from 1 to skip background
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_final[labels == i] = 255
            
    return mask_final

# Initialize blob detector only if enabled
if ENABLE_BLOB_DETECTION:
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.minArea = MIN_CONTOUR_AREA
    blob_params.filterByCircularity = False
    blob_params.filterByConvexity = False
    blob_params.filterByInertia = False
    blob_detector = cv2.SimpleBlobDetector_create(blob_params)

# Video input/output and logging setup
cap = cv2.VideoCapture('fruit-and-vegetable-detection.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Handle zero fps case (happens with some cameras or still images)
if fps == 0:
    fps = 30  # Default to 30 fps

# Create video writer with faster playback speed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps*OUTPUT_FPS_MULTIPLIER, (width, height))
log = open('log.csv', 'w')
log.write('Frame,' + ','.join(color_defs.keys()) + ',Total\n')

# Performance tracking
frame_idx = 0
start_time = time.time()

# Pre-compute color arrays for speed
color_arrays = {name: (np.array(params['hsv_lo']), np.array(params['hsv_hi']), params['bgr']) 
                for name, params in color_defs.items()}

# Optional: Brightness adjustment values
brightness_adjust = 0  # No brightness adjustment for cityscapes
contrast_adjust = 1.1  # Slight contrast boost

print("Starting video processing with optimized settings...")
print(f"- Min contour area: {MIN_CONTOUR_AREA}")
print(f"- Processing scale: {PROCESSING_SCALE}")
print(f"- Output speed: {OUTPUT_FPS_MULTIPLIER}x faster")

while True:
    # Measure per-frame time
    frame_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    
    # Optional: Improve image contrast and brightness for better detection
    frame = cv2.convertScaleAbs(frame, alpha=contrast_adjust, beta=brightness_adjust)
    
    # Scale down for faster processing
    if PROCESSING_SCALE != 1.0:
        proc_frame = cv2.resize(frame, (0, 0), fx=PROCESSING_SCALE, fy=PROCESSING_SCALE)
    else:
        proc_frame = frame
    
    # Convert to HSV for color masks (just once)
    hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
    
    # Create display frame
    display = frame.copy()
    
    # Create panel for displaying counts - styled like color_counting.py
    panel_height = 30 + len(color_defs) * 20
    panel_width = 200
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = PANEL_BG  # Dark gray background
    
    # Blend panel onto display frame
    display[0:panel_height, 0:panel_width] = cv2.addWeighted(
        display[0:panel_height, 0:panel_width], 1-PANEL_ALPHA, 
        panel, PANEL_ALPHA, 0)
    
    # Process edge detection if enabled
    if ENABLE_EDGE_DETECTION:
        edges = cv2.Canny(proc_frame, 100, 200)
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        display = cv2.addWeighted(display, 0.8, cv2.resize(edge_bgr, (width, height)), 0.2, 0)
    
    # Process each color
    counts = []
    all_contours = []  # Collect all contours to prevent overlapping detections
    
    for name, (lo, hi, bgr_color) in color_arrays.items():
        # Create color mask
        mask = cv2.inRange(hsv, lo, hi)
        
        # Apply shadow detection for specific colors if enabled
        if ENABLE_SHADOW_DETECTION and name in ['Green', 'DarkGreen', 'LightGreen', 'Brown', 'DarkBrown']:
            # Create a shadow mask with lower saturation and value
            shadow_lo = lo.copy()
            shadow_hi = hi.copy()
            
            # Adjust shadow thresholds (lower brightness/saturation but same hue)
            shadow_lo[1] = max(0, shadow_lo[1] - 40)       # Lower min saturation
            shadow_lo[2] = max(0, shadow_lo[2] - 50)       # Lower min value
            shadow_hi[1] = min(255, shadow_hi[1] - 20)     # Lower max saturation
            
            shadow_mask = cv2.inRange(hsv, shadow_lo, shadow_hi)
            mask = cv2.bitwise_or(mask, shadow_mask)
        
        # Special handling for gray detection to avoid confusion with concrete
        if name == 'Gray' and 'Green' in color_defs:
            # Create a mask for green areas to exclude from gray
            green_lo = color_arrays['Green'][0]
            green_hi = color_arrays['Green'][1]
            green_mask = cv2.inRange(hsv, green_lo, green_hi)
            # Remove green areas from gray mask
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))
            
        # Optimized morphological operations - only if we have some pixels
        if np.count_nonzero(mask) > 0:
            # Enhanced mask processing
            mask = enhance_mask(mask, min_area=MIN_BLOB_AREA)
            
            # Find contours on the enhanced mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check each contour
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue
                
                # Scale coordinates if we processed at lower resolution
                if PROCESSING_SCALE != 1.0:
                    cnt = cnt * (1/PROCESSING_SCALE)
                
                # Check if the contour overlaps with already detected objects
                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
                overlap = False
                for existing_cnt in all_contours:
                    ex, ey, ew, eh = cv2.boundingRect(existing_cnt.astype(np.int32))
                    # Calculate overlap area
                    x_overlap = max(0, min(x+w, ex+ew) - max(x, ex))
                    y_overlap = max(0, min(y+h, ey+eh) - max(y, ey))
                    overlap_area = x_overlap * y_overlap
                    
                    if overlap_area > 0.7 * area:  # If more than 70% overlap
                        overlap = True
                        break
                
                if not overlap:
                    valid_contours.append(cnt)
                    all_contours.append(cnt)
            
            # Process valid contours
            count = 0
            for cnt in valid_contours:
                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
                
                # Draw bounding box and label
                cv2.rectangle(display, (x, y), (x+w, y+h), bgr_color, 2)
                
                # Add text label above the rectangle
                cv2.putText(display, name, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, bgr_color, 2)
                
                count += 1
        else:
            count = 0
            
        counts.append(count)
    
    # Add blob detection if enabled
    blob_count = 0
    if ENABLE_BLOB_DETECTION:
        keypoints = blob_detector.detect(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY))
        display = cv2.drawKeypoints(display, keypoints, None, (0, 255, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        blob_count = len(keypoints)
    
    # Overlay counts like color_counting.py - more compact
    total = sum(counts)
    
    # Write Total count
    y0 = 20
    cv2.putText(display, f'Total: {total}', (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Write individual color counts - more compact style
    for i, (name, count) in enumerate(zip(color_defs.keys(), counts)):
        # Skip showing zero counts for cleaner display
        if count > 0 or name in ['Red', 'Green', 'Blue', 'Yellow', 'Gray', 'Black', 'White']:
            color_bgr = color_arrays[name][2]
            cv2.putText(display, f'{name}: {count}', (10, y0 + 20*(i+1)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
    
    # Calculate and show FPS
    frame_time = time.time() - frame_start
    fps_text = f'FPS: {1.0/frame_time:.1f}'
    cv2.putText(display, fps_text, (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    
    # Write output and show preview (scaled for display if needed)
    out.write(display)
    
    if DISPLAY_SCALE != 1.0:
        display_resize = cv2.resize(display, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow('Optimized Detection', display_resize)
    else:
        cv2.imshow('Optimized Detection', display)
    
    # Log to CSV
    log.write(f'{frame_idx},' + ','.join(map(str, counts)) + f',{total}\n')
    
    # Print progress every 30 frames
    if frame_idx % 30 == 0:
        elapsed = time.time() - start_time
        avg_fps = frame_idx / elapsed
        print(f"Frame {frame_idx}: {avg_fps:.1f} FPS average")
    
    # Exit on ESC or 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# Cleanup
total_time = time.time() - start_time
print(f"Processed {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.1f} FPS)")
cap.release()
out.release()
log.close()
cv2.destroyAllWindows()