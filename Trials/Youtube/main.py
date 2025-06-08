import cv2
from util import get_combined_mask

# open your camera (0, 1, or 2...)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # get the pink mask
    mask = get_combined_mask(frame)

    # find contours on that mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # pick largest area
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 2000:    # ignore tiny blobs
            x,y,w,h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    # show results
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask',  mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
