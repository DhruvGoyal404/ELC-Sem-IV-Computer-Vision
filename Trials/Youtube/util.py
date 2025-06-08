import cv2
import numpy as np

def get_combined_mask(frame):
    """
    Returns a binary mask of 'pink' regions by:
      1) Hue-based HSV threshold
      2) 'a'-channel threshold from Lab color space
      3) Morphological open+close to remove noise
    """
    # --- 1) HSV filter for pink-ish hues ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([145,  60,  60], dtype=np.uint8)  # tweak H, S, V min
    upper_hsv = np.array([175, 255, 255], dtype=np.uint8)  # tweak H max
    hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # --- 2) Lab a-channel filter for pink vs green ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(lab)
    # any pixel with a_channel > 145 is more pink/magenta than green
    _, lab_mask = cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY)

    # --- 3) combine and clean up ---
    mask = cv2.bitwise_and(hsv_mask, lab_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask
