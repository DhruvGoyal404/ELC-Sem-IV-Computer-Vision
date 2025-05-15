import cv2
import numpy as np
import csv

# --- CONFIGURATION ---

INPUT_VIDEO  = 'fruit-and-vegetable-detection.mp4'
OUTPUT_VIDEO = 'fruit_veg_counted_2.mp4'
CSV_LOG      = 'counts_log_2.csv'

SCALE      = 0.5    # Process at 50% resolution for speed
MIN_AREA   = 1000   # Min blob area at scaled size
PANEL_BG   = (50,50,50)  # dark gray background
PANEL_ALPHA= 0.6        # transparency of panel

# Define HSV color ranges and BGR draw colors
COLOR_RANGES = {
    'Red':        ((0, 120, 70),   (10, 255,255), (0,0,255)),
    'Red2':       ((170,120,70),   (180,255,255),(0,0,200)),
    'Green':      ((36, 25, 25),   (86, 255,255), (0,255,0)),
    'LightGreen': ((25, 40, 40),   (50, 255,255), (144,238,144)),  # catches paler greens
    'Yellow':     ((15,150,150),   (35, 255,255), (0,255,255)),
    'Blue':       ((94,  80, 2),   (126,255,255), (255,0,0)),
    'Orange':     ((5,   50, 50),  (15, 255,255), (0,165,255)),
}

# Morphology kernels
K_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# --- VIDEO & CSV SETUP ---

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (w,h))

csv_file = open(CSV_LOG, 'w', newline='')
writer   = csv.writer(csv_file)
writer.writerow(['frame'] + list(COLOR_RANGES.keys()) + ['Total'])

frame_idx = 0

# --- PROCESSING LOOP ---

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # 1) Downscale + HSV convert
    small = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    counts = []
    # 2) Per‑color masking, morphology, contouring
    for name, (lo, hi, bgr) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  K_OPEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CLOSE)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = 0
        for cnt in cnts:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            x,y,wc,hc = cv2.boundingRect(cnt)
            # scale to full‑res
            x0 = int(x/SCALE); y0 = int(y/SCALE)
            x1 = int((x+wc)/SCALE); y1 = int((y+hc)/SCALE)
            cv2.rectangle(frame, (x0,y0),(x1,y1), bgr, 2)
            cv2.putText(frame, name, (x0, y0-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
            c += 1
        counts.append(c)

    total = sum(counts)

    # 3) Draw translucent panel & text
    panel_h = 20 + 25*(len(COLOR_RANGES)+1)
    panel = np.zeros((panel_h, 200, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    overlay = frame.copy()
    overlay[0:panel_h, 0:200] = cv2.addWeighted(frame[0:panel_h,0:200],
                                                1-PANEL_ALPHA,
                                                panel,
                                                PANEL_ALPHA, 0)
    frame[0:panel_h,0:200] = overlay[0:panel_h,0:200]

    # write counts
    y0 = 20
    cv2.putText(frame, f'Total: {total}', (10,y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    for i,(name,_) in enumerate(COLOR_RANGES.items()):
        cv2.putText(frame,
                    f'{name}: {counts[i]}',
                    (10, y0 + 25*(i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)

    # 4) Write & log
    out.write(frame)
    writer.writerow([frame_idx] + counts + [total])

    # 5) Show live
    cv2.imshow('Color Counting', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- CLEAN UP ---

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
