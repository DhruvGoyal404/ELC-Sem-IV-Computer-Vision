
import cv2                                    # OpenCV for video I/O :contentReference[oaicite:0]{index=0}
import torch                                 # PyTorch for YOLOv5 :contentReference[oaicite:1]{index=1}
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT for tracking 
import pandas as pd                          # For logging and precision/recall 
import tkinter as tk                         # Simple GUI 
import time

# --- Configuration ---
VIDEO_INPUT      = 'cityscape_clip.mp4'
VIDEO_OUTPUT     = 'yolo_deepsort_output_final.mp4'
GROUND_TRUTH_CSV = 'ground_truth_main.csv'       # format: frame,x1,y1,x2,y2,class
LOG_CSV          = 'detection_log_final.csv'
CONFIDENCE_THRESH= 0.3
IOU_THRESH       = 0.5
TRACK_MAX_AGE    = 30

# --- Day 1: Load YOLOv5 Model ---
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # :contentReference[oaicite:5]{index=5}
model.conf = CONFIDENCE_THRESH

# --- Day 3: Initialize DeepSORT Tracker ---
print("Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=TRACK_MAX_AGE)

# --- Video I/O Setup ---
cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

# --- Logging Setup ---
gt = pd.read_csv(GROUND_TRUTH_CSV)  # your ground truth
log = []
columns = ['frame','track_id','class','x1','y1','x2','y2','conf','tp','fp','fn']
# tp/fp/fn will be computed Day 4
year = time.strftime("%Y")

# --- Day 5: Simple Tkinter GUI to toggle classes ---
root = tk.Tk()
root.title("Classes to Detect")
vars = {}
for cls in ['person','car','truck','bus']:
    var = tk.IntVar(value=1)
    cb = tk.Checkbutton(root, text=cls, variable=var)
    cb.pack(anchor='w')
    vars[cls] = var
tk.Button(root, text="Start", command=root.quit).pack()
root.mainloop()

# --- Main Processing Loop ---
frame_idx = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Day 1: Run YOLOv5 inference
    results = model(frame[..., ::-1])  # BGR→RGB
    detections = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class]

    # Filter by GUI selection
    dets = []
    for *box, conf, cls in detections:
        cls_name = model.names[int(cls)]
        if vars.get(cls_name, tk.IntVar(value=0)).get() == 1:
            dets.append((box, conf, cls_name))

    # Day 3: Update DeepSORT with detections
    tracks = tracker.update_tracks(dets, frame=frame)

    # Draw and log
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        ltrb = track.to_ltrb()  # [x1,y1,x2,y2]
        cls = track.get_det_class()  # class name
        conf = track.det_conf

        # Draw
        x1,y1,x2,y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f'{cls}-{tid}',(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # Day 4: Precision/Recall logic (match with gt)
        gt_frame = gt[gt['frame']==frame_idx]
        # simplest IoU match
        tp=fp=fn=0
        matched = []
        for _,g in gt_frame.iterrows():
            iou = (
                max(0, min(x2,g.x2)-max(x1,g.x1))*
                max(0, min(y2,g.y2)-max(y1,g.y1))
            )/((x2-x1)*(y2-y1)+(g.x2-g.x1)*(g.y2-g.y1)-0.0001)
            if iou>IOU_THRESH and cls==g['class']:
                tp+=1
                matched.append(_)
        fp = 1 if tp==0 else 0
        fn = len(gt_frame)-len(matched)
        
        log.append([frame_idx, tid, cls, x1,y1,x2,y2,conf,tp,fp,fn])

    out.write(frame)
    cv2.imshow("YOLOv5 + DeepSORT", frame)
    if cv2.waitKey(1)==27:
        break

# Cleanup
cap.release(); out.release(); cv2.destroyAllWindows()

# Day 4: Save log & compute overall metrics
df = pd.DataFrame(log, columns=columns)
df.to_csv(LOG_CSV, index=False)
tp = df.tp.sum(); fp = df.fp.sum(); fn = df.fn.sum()
precision = tp/(tp+fp+1e-6)
recall    = tp/(tp+fn+1e-6)
accuracy  = tp/(tp+fp+fn+1e-6)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}")
