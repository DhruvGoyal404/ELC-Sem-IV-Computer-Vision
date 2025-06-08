import cv2
import numpy as np
import imutils

# Configuration
MIN_AREA = 1500  # Minimum area for contour detection
ENABLE_SAVE = False  # Set to True if you want to save the output

# Define color ranges in HSV with names and BGR drawing colors
COLOR_RANGES = {
    'Red':       {'lower1': (0, 50, 50),    'upper1': (10, 255, 255),    'lower2': (170, 120, 70), 'upper2': (180, 255, 255), 'bgr': (0, 0, 255)},
    'Orange':    {'lower1': (11, 50, 50),   'upper1': (20, 255, 255),    'lower2': (181, 120, 70), 'upper2': (190, 255, 255), 'bgr': (0, 165, 255)},
    'Yellow':    {'lower1': (21, 50, 50),   'upper1': (30, 255, 255),    'lower2': (191, 120, 70), 'upper2': (200, 255, 255), 'bgr': (0, 255, 255)},
    'Green':     {'lower1': (31, 50, 50),   'upper1': (80, 255, 255),    'lower2': (201, 120, 70), 'upper2': (250, 255, 255), 'bgr': (0, 255, 0)},
    'LightBlue': {'lower1': (81, 50, 50),   'upper1': (110, 255, 255),   'lower2': (251, 120, 70), 'upper2': (290, 255, 255), 'bgr': (255, 255, 0)},
    'Blue':      {'lower1': (111, 50, 50),  'upper1': (130, 255, 255),   'lower2': (291, 120, 70), 'upper2': (300, 255, 255), 'bgr': (255, 0, 0)},
    'Violet':    {'lower1': (131, 50, 50),  'upper1': (140, 255, 255),   'lower2': (301, 120, 70), 'upper2': (310, 255, 255), 'bgr': (238, 130, 238)},
    'Purple':    {'lower1': (141, 50, 50),  'upper1': (160, 255, 255),   'lower2': (311, 120, 70), 'upper2': (330, 255, 255), 'bgr': (128, 0, 128)},
    'Pink':      {'lower1': (161, 50, 50),  'upper1': (170, 255, 255),   'lower2': (331, 120, 70), 'upper2': (340, 255, 255), 'bgr': (203, 192, 255)},
    'Gray':      {'lower1': (0, 0, 10),     'upper1': (0, 0, 225),       'lower2': (361, 120, 70), 'upper2': (380, 255, 255), 'bgr': (120, 120, 120)},
    'Black':     {'lower1': (0, 0, 0),      'upper1': (180, 255, 30),    'lower2': None,           'upper2': None,            'bgr': (0, 0, 0)},
    'White':     {'lower1': (0, 0, 200),    'upper1': (180, 30, 255),    'lower2': None,           'upper2': None,            'bgr': (255, 255, 255)},
}

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Create output video writer if saving is enabled
if ENABLE_SAVE:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('color_detection_output.mp4', fourcc, 20.0, (640, 480))

while True:
    _, frame = cap.read()
    if frame is None:
        print("Failed to capture frame. Check camera connection.")
        break

    # Convert to HSV once
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create panel for displaying counts
    panel_height = 30 + len(COLOR_RANGES) * 20
    panel = np.zeros((panel_height, 200, 3), dtype=np.uint8)
    panel[:] = (50, 50, 50)  # Dark gray background
    frame[0:panel_height, 0:200] = cv2.addWeighted(frame[0:panel_height, 0:200], 0.4, panel, 0.6, 0)
    
    # Process each color
    total_count = 0
    y_offset = 25
    
    for color_name, ranges in COLOR_RANGES.items():
        # Create mask for primary range
        mask1 = cv2.inRange(hsv, np.array(ranges['lower1']), np.array(ranges['upper1']))
        
        # Add secondary range if it exists
        if ranges['lower2'] is not None:
            mask2 = cv2.inRange(hsv, np.array(ranges['lower2']), np.array(ranges['upper2']))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
            
        # Apply morphological operations for noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Count and process valid contours
        count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_AREA:
                # Draw contour
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
                
                # Calculate and mark centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:  # Avoid division by zero
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw a circle at centroid
                    cv2.circle(frame, (cx, cy), 3, ranges['bgr'], -1)
                    
                    # Label the color
                    cv2.putText(frame, color_name, (cx-20, cy-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ranges['bgr'], 2)
                    
                count += 1
                
        # Update total count
        total_count += count
        
        # Display color count on panel
        cv2.putText(frame, f"{color_name}: {count}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ranges['bgr'], 1)
        y_offset += 20
    
    # Display total count
    cv2.putText(frame, f"Total: {total_count}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Write frame to output if saving is enabled
    if ENABLE_SAVE:
        out.write(frame)
    
    # Show result
    cv2.imshow("Color Detection", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
if ENABLE_SAVE:
    out.release()
cv2.destroyAllWindows()