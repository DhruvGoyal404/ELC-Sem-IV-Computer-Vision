# # enhanced_color_detect_cityscape.py
# """
# Real-Time Multi-Color Object & Blob Detection with Enhanced Speed and Accuracy
# """

# import cv2
# import numpy as np
# import time

# # Performance configuration
# ENABLE_BLOB_DETECTION = False   # Turn off for better performance
# ENABLE_EDGE_DETECTION = False   # Turn off for better performance
# PROCESSING_SCALE = 0.75         # Scale down frames for faster processing
# MIN_CONTOUR_AREA = 2500         # Minimum area to be considered an object
# DISPLAY_SCALE = 1.0             # Display frame scale (1.0 = original size)
# OUTPUT_FPS_MULTIPLIER = 3       # Make output video play 3x faster than original

# # Additional configuration parameters from Code B
# COLOR_SENSITIVITY = 1.0       # Adjust HSV sensitivity (1.0 = normal)
# MIN_BLOB_AREA = 1000          # Minimum area for initial blob filtering
# CLOSE_KERNEL_SIZE = 11        # Size for closing operation kernel
# OPEN_KERNEL_SIZE = 5          # Size for opening operation kernel
# ENABLE_HOLE_FILLING = True    # Enable filling holes in blobs

# # Step 1: Define HSV thresholds and BGR display colors for each target
# # IMPROVED: Wider thresholds especially for orange and better color separation
# color_defs = {
#     'Red':       {'hsv_lo': (0,   160, 120), 'hsv_hi': (7,   255, 255), 'bgr': (0, 0, 255)},
#     'Red2':      {'hsv_lo': (173, 160, 120), 'hsv_hi': (180, 255, 255), 'bgr': (0, 0, 200)},
#     'Orange':    {'hsv_lo': (8,  120, 130), 'hsv_hi': (22,  255, 255), 'bgr': (0, 165, 255)}, # Wider gap from red
#     'Yellow':    {'hsv_lo': (23,  130, 130), 'hsv_hi': (32,  255, 255), 'bgr': (0, 255, 255)},
#     'Green':     {'hsv_lo': (40,  70,  70),  'hsv_hi': (80,  255, 255), 'bgr': (0, 255, 0)},
#     'Blue':      {'hsv_lo': (100, 100, 70),  'hsv_hi': (130, 255, 255), 'bgr': (255, 0, 0)},
#     'DarkGreen': {'hsv_lo': (45,  70,  40),  'hsv_hi': (75,  255, 100), 'bgr': (0, 100, 0)},
#     'Purple':    {'hsv_lo': (130, 75,  50),  'hsv_hi': (155, 255, 255), 'bgr': (128, 0, 128)},
#     'Pink':      {'hsv_lo': (145, 80,  100), 'hsv_hi': (170, 255, 255), 'bgr': (203, 192, 255)},
#     'Brown':     {'hsv_lo': (5,   50,  20),  'hsv_hi': (15,  150, 150), 'bgr': (42, 42, 165)},
#     'LightBlue': {'hsv_lo': (85,  40,  150), 'hsv_hi': (95,  150, 255), 'bgr': (255, 255, 0)},
# }

# # Enhanced morphological processing function from Code B
# def enhance_mask(mask, min_area=1000):
#     # Remove small noise
#     kernel_open = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
#     kernel_close = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)
    
#     # First remove small dots (noise)
#     mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
#     # Close small holes
#     mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)
    
#     # Fill any remaining holes if enabled
#     mask_filled = mask_closed.copy()
#     if ENABLE_HOLE_FILLING:
#         contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(mask_filled, contours, -1, 255, -1)  # Fill all contours
    
#     # Final cleanup - remove small blobs
#     nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled)
#     mask_final = np.zeros_like(mask_filled)
    
#     # Start from 1 to skip background
#     for i in range(1, nlabels):
#         if stats[i, cv2.CC_STAT_AREA] >= min_area:
#             mask_final[labels == i] = 255
            
#     return mask_final

# # Initialize blob detector only if enabled
# if ENABLE_BLOB_DETECTION:
#     blob_params = cv2.SimpleBlobDetector_Params()
#     blob_params.filterByArea = True
#     blob_params.minArea = MIN_CONTOUR_AREA
#     blob_params.filterByCircularity = False
#     blob_params.filterByConvexity = False
#     blob_params.filterByInertia = False
#     blob_detector = cv2.SimpleBlobDetector_create(blob_params)

# # Video input/output and logging setup
# cap = cv2.VideoCapture('cityscape_clip_1.mp4')
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Create video writer with faster playback speed
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('optimized_output.mp4', fourcc, fps*OUTPUT_FPS_MULTIPLIER, (width, height))
# log = open('optimized_log.csv', 'w')
# log.write('Frame,' + ','.join(color_defs.keys()) + '\n')

# # Performance tracking
# frame_idx = 0
# start_time = time.time()
# kernel_open = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
# kernel_close = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)

# # Pre-compute color arrays for speed
# color_arrays = {name: (np.array(params['hsv_lo']), np.array(params['hsv_hi']), params['bgr']) 
#                 for name, params in color_defs.items()}

# print("Starting video processing with optimized settings...")
# print(f"- Min contour area: {MIN_CONTOUR_AREA}")
# print(f"- Processing scale: {PROCESSING_SCALE}")
# print(f"- Output speed: {OUTPUT_FPS_MULTIPLIER}x faster")
# print(f"- Blob detection: {'Enabled' if ENABLE_BLOB_DETECTION else 'Disabled'}")
# print(f"- Edge detection: {'Enabled' if ENABLE_EDGE_DETECTION else 'Disabled'}")
# print(f"- Hole filling: {'Enabled' if ENABLE_HOLE_FILLING else 'Disabled'}")
# print(f"- Color sensitivity: {COLOR_SENSITIVITY}")
# print(f"- Min blob area: {MIN_BLOB_AREA}")

# while True:
#     # Measure per-frame time
#     frame_start = time.time()
    
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame_idx += 1
    
#     # Scale down for faster processing
#     if PROCESSING_SCALE != 1.0:
#         proc_frame = cv2.resize(frame, (0, 0), fx=PROCESSING_SCALE, fy=PROCESSING_SCALE)
#     else:
#         proc_frame = frame
    
#     # Convert to HSV for color masks (just once)
#     hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
    
#     # Create display frame
#     display = frame.copy()
    
#     # Process edge detection if enabled
#     if ENABLE_EDGE_DETECTION:
#         edges = cv2.Canny(proc_frame, 100, 200)
#         edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#         display = cv2.addWeighted(display, 0.8, cv2.resize(edge_bgr, (width, height)), 0.2, 0)
    
#     # Process each color
#     counts = []
    
#     for name, (lo, hi, bgr_color) in color_arrays.items():
#         # Create color mask
#         mask = cv2.inRange(hsv, lo, hi)
        
#         # Optimized morphological operations
#         if np.count_nonzero(mask) > 0:  # Only process if mask has non-zero pixels
#             # Use enhanced mask processing from Code B
#             mask = enhance_mask(mask, min_area=MIN_BLOB_AREA)
            
#             # Find contours on the enhanced mask
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # Count and display
#             count = 0
#             for cnt in contours:
#                 area = cv2.contourArea(cnt)
#                 if area < MIN_CONTOUR_AREA:
#                     continue
                    
#                 # Scale coordinates if we processed at lower resolution
#                 if PROCESSING_SCALE != 1.0:
#                     cnt = cnt * (1/PROCESSING_SCALE)
                    
#                 x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
#                 cv2.rectangle(display, (x, y), (x+w, y+h), bgr_color, 2)
#                 cv2.putText(display, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.6, bgr_color, 2)
#                 count += 1
#         else:
#             count = 0
            
#         counts.append(count)
    
#     # Add blob detection if enabled
#     blob_count = 0
#     if ENABLE_BLOB_DETECTION:
#         keypoints = blob_detector.detect(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY))
#         display = cv2.drawKeypoints(display, keypoints, None, (0, 255, 255),
#                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         blob_count = len(keypoints)
    
#     # Overlay counts efficiently
#     total = sum(counts)
#     cv2.putText(display, f'Total: {total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (255, 255, 255), 2)
    
#     # Show all colors (FIXED: especially making sure Orange is shown)
#     y = 60
#     for idx, (name, count) in enumerate(zip(color_defs.keys(), counts)):
#         # Always show if detected or one of the primary colors
#         color_bgr = color_arrays[name][2]
#         cv2.putText(display, f'{name}: {count}', (10, y), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
#         y += 25
    
#     # Calculate and show FPS
#     frame_time = time.time() - frame_start
#     fps_text = f'FPS: {1.0/frame_time:.1f}'
#     cv2.putText(display, fps_text, (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (0, 255, 0), 2)
    
#     # Write output and show preview (scaled for display if needed)
#     out.write(display)
    
#     if DISPLAY_SCALE != 1.0:
#         display_resize = cv2.resize(display, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
#         cv2.imshow('Optimized Detection', display_resize)
#     else:
#         cv2.imshow('Optimized Detection', display)
    
#     # Log to CSV
#     log.write(f'{frame_idx},' + ','.join(map(str, counts)) + '\n')
    
#     # Print progress every 30 frames
#     if frame_idx % 30 == 0:
#         elapsed = time.time() - start_time
#         avg_fps = frame_idx / elapsed
#         print(f"Frame {frame_idx}: {avg_fps:.1f} FPS average")
    
#     # Exit on ESC or 'q'
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27 or key == ord('q'):
#         break

# # Cleanup
# total_time = time.time() - start_time
# print(f"Processed {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.1f} FPS)")
# cap.release()
# out.release()
# log.close()
# cv2.destroyAllWindows()


# enhanced_color_detect_cityscape.py
"""
Real-Time Multi-Color Object & Blob Detection with Enhanced Speed and Accuracy
"""

import cv2
import numpy as np
import time

# Performance configuration
ENABLE_BLOB_DETECTION = False   # Turn off for better performance
ENABLE_EDGE_DETECTION = False   # Turn off for better performance
PROCESSING_SCALE = 0.75         # Scale down frames for faster processing
MIN_CONTOUR_AREA = 2500         # Minimum area to be considered an object
DISPLAY_SCALE = 1.0             # Display frame scale (1.0 = original size)
OUTPUT_FPS_MULTIPLIER = 3       # Make output video play 3x faster than original

# Step 1: Define HSV thresholds and BGR display colors for each target
# IMPROVED: Wider thresholds especially for orange and better color separation
color_defs = {
    'Red':       {'hsv_lo': (0,   160, 120), 'hsv_hi': (7,   255, 255), 'bgr': (0, 0, 255)},
    'Red2':      {'hsv_lo': (173, 160, 120), 'hsv_hi': (180, 255, 255), 'bgr': (0, 0, 200)},
    'Orange':    {'hsv_lo': (8,  120, 130), 'hsv_hi': (22,  255, 255), 'bgr': (0, 165, 255)}, # Wider gap from red
    'Yellow':    {'hsv_lo': (23,  130, 130), 'hsv_hi': (32,  255, 255), 'bgr': (0, 255, 255)},
    'Green':     {'hsv_lo': (40,  70,  70),  'hsv_hi': (80,  255, 255), 'bgr': (0, 255, 0)},
    'Blue':      {'hsv_lo': (100, 100, 70),  'hsv_hi': (130, 255, 255), 'bgr': (255, 0, 0)},
    'DarkGreen': {'hsv_lo': (45,  70,  40),  'hsv_hi': (75,  255, 100), 'bgr': (0, 100, 0)},
    'Purple':    {'hsv_lo': (130, 75,  50),  'hsv_hi': (155, 255, 255), 'bgr': (128, 0, 128)},
    'Pink':      {'hsv_lo': (145, 80,  100), 'hsv_hi': (170, 255, 255), 'bgr': (203, 192, 255)},
    'Brown':     {'hsv_lo': (5,   50,  20),  'hsv_hi': (15,  150, 150), 'bgr': (42, 42, 165)},
    'LightBlue': {'hsv_lo': (85,  40,  150), 'hsv_hi': (95,  150, 255), 'bgr': (255, 255, 0)},
}

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

# Create video writer with faster playback speed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('optimized_output.mp4', fourcc, fps*OUTPUT_FPS_MULTIPLIER, (width, height))
log = open('optimized_log.csv', 'w')
log.write('Frame,' + ','.join(color_defs.keys()) + '\n')

# Performance tracking
frame_idx = 0
start_time = time.time()
kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((11, 11), np.uint8)

# Pre-compute color arrays for speed
color_arrays = {name: (np.array(params['hsv_lo']), np.array(params['hsv_hi']), params['bgr']) 
                for name, params in color_defs.items()}

print("Starting video processing with optimized settings...")
print(f"- Min contour area: {MIN_CONTOUR_AREA}")
print(f"- Processing scale: {PROCESSING_SCALE}")
print(f"- Output speed: {OUTPUT_FPS_MULTIPLIER}x faster")
print(f"- Blob detection: {'Enabled' if ENABLE_BLOB_DETECTION else 'Disabled'}")
print(f"- Edge detection: {'Enabled' if ENABLE_EDGE_DETECTION else 'Disabled'}")

while True:
    # Measure per-frame time
    frame_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    
    # Scale down for faster processing
    if PROCESSING_SCALE != 1.0:
        proc_frame = cv2.resize(frame, (0, 0), fx=PROCESSING_SCALE, fy=PROCESSING_SCALE)
    else:
        proc_frame = frame
    
    # Convert to HSV for color masks (just once)
    hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
    
    # Create display frame
    display = frame.copy()
    
    # Process edge detection if enabled
    if ENABLE_EDGE_DETECTION:
        edges = cv2.Canny(proc_frame, 100, 200)
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        display = cv2.addWeighted(display, 0.8, cv2.resize(edge_bgr, (width, height)), 0.2, 0)
    
    # Process each color
    counts = []
    
    for name, (lo, hi, bgr_color) in color_arrays.items():
        # Create color mask
        mask = cv2.inRange(hsv, lo, hi)
        
        # Optimized morphological operations
        if np.count_nonzero(mask) > 0:  # Only process if mask has non-zero pixels
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Find contours (use RETR_EXTERNAL for faster processing)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count and display
            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue
                    
                # Scale coordinates if we processed at lower resolution
                if PROCESSING_SCALE != 1.0:
                    cnt = cnt * (1/PROCESSING_SCALE)
                    
                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
                cv2.rectangle(display, (x, y), (x+w, y+h), bgr_color, 2)
                cv2.putText(display, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
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
    
    # Overlay counts efficiently
    total = sum(counts)
    cv2.putText(display, f'Total: {total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    
    # Show all colors (FIXED: especially making sure Orange is shown)
    y = 60
    for idx, (name, count) in enumerate(zip(color_defs.keys(), counts)):
        # Always show if detected or one of the primary colors
        color_bgr = color_arrays[name][2]
        cv2.putText(display, f'{name}: {count}', (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        y += 25
    
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
    log.write(f'{frame_idx},' + ','.join(map(str, counts)) + '\n')
    
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