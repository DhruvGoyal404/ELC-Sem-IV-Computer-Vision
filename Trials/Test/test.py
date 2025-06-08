import cv2                                    # OpenCV for video I/O
import torch                                 # PyTorch + YOLOv5
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracker
import pandas as pd                          # For CSV I/O and logging
import tkinter as tk                         # Simple GUI for class toggles
import time

# --- Configuration ---
VIDEO_INPUT       = 'cityscape_clip.mp4'
VIDEO_OUTPUT      = 'june_yolo_deepsort_output_final_june.mp4'
GROUND_TRUTH_CSV  = 'detections.csv'        # your 30-row CSV
LOG_CSV           = 'june_detection_log_final_june.csv'
CONFIDENCE_THRESH = 0.3
IOU_THRESH        = 0.5
TRACK_MAX_AGE     = 30

# --- 1) Load YOLOv5 Model ---
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = CONFIDENCE_THRESH

# --- 2) Initialize DeepSORT Tracker ---
print("Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=TRACK_MAX_AGE)

# --- 3) Video I/O Setup ---
cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# --- 4) Load & Normalize Ground Truth CSV ---
gt = pd.read_csv(GROUND_TRUTH_CSV)
print("GT columns before rename:", gt.columns.tolist())

# Rename if needed
if 'color' in gt.columns:
    gt.rename(columns={'color': 'class'}, inplace=True)
# Ensure 'frame' is present
if 'Frame' in gt.columns:
    gt.rename(columns={'Frame': 'frame'}, inplace=True)
elif 'frame_index' in gt.columns:
    gt.rename(columns={'frame_index': 'frame'}, inplace=True)

print("GT columns after rename:", gt.columns.tolist())

# --- 5) Prepare Logging ---
log = []
columns = ['frame','track_id','class','x1','y1','x2','y2','conf','tp','fp','fn']
start_time = time.time()

# --- 6) Tkinter GUI to Toggle Classes ---
root = tk.Tk()
root.title("Classes to Detect")
vars = {}
for cls in set(gt['class'].tolist()):
    var = tk.IntVar(value=1)
    tk.Checkbutton(root, text=cls, variable=var).pack(anchor='w')
    vars[cls] = var
tk.Button(root, text="Start", command=root.quit).pack()
root.mainloop()

# --- 7) Main Processing Loop ---
frame_idx = -1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # YOLOv5 inference (BGR→RGB)
    results = model(frame[..., ::-1])
    detections = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class_id]

    # Filter by GUI selection
    filtered = []
    for *box, conf, cls_id in detections:
        cls_name = model.names[int(cls_id)]
        if vars.get(cls_name, tk.IntVar(value=0)).get() == 1:
            filtered.append((box, conf, cls_name))

    # DeepSORT tracking
    tracks = tracker.update_tracks(filtered, frame=frame)

    # Draw & log each confirmed track
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cls_name = track.get_det_class()
        conf = track.det_conf

        # Draw bounding box + label
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f'{cls_name}-{tid}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Compute TP/FP/FN by matching to ground truth for this frame
        gt_frame = gt[gt['frame'] == frame_idx]
        tp = fp = fn = 0
        matched = set()
        for idx, g in gt_frame.iterrows():
            # Intersection‐over‐Union
            ix = max(0, min(x2, g.x2) - max(x1, g.x1))
            iy = max(0, min(y2, g.y2) - max(y1, g.y1))
            inter = ix * iy
            union = ((x2-x1)*(y2-y1) + (g.x2-g.x1)*(g.y2-g.y1) - inter) + 1e-6
            iou = inter / union
            if iou > IOU_THRESH and cls_name == g['class']:
                tp += 1
                matched.add(idx)
        fp = 1 if tp == 0 else 0
        fn = len(gt_frame) - len(matched)

        log.append([frame_idx, tid, cls_name, x1, y1, x2, y2, conf, tp, fp, fn])

    # Write frame to output & display
    out.write(frame)
    cv2.imshow("YOLOv5 + DeepSORT", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

# --- 8) Cleanup & Metrics ---
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detection log
df_log = pd.DataFrame(log, columns=columns)
df_log.to_csv(LOG_CSV, index=False)

# Compute overall precision, recall, accuracy
tp_sum = df_log.tp.sum()
fp_sum = df_log.fp.sum()
fn_sum = df_log.fn.sum()
precision = tp_sum / (tp_sum + fp_sum + 1e-6)
recall    = tp_sum / (tp_sum + fn_sum + 1e-6)
accuracy  = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-6)

print(f"\nResults on {time.strftime('%Y-%m-%d')}:\n"
      f"  Precision: {precision:.3f}\n"
      f"  Recall:    {recall:.3f}\n"
      f"  Accuracy:  {accuracy:.3f}\n")
