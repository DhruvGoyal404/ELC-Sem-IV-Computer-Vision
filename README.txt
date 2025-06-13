===============================
COLOR-BASED OBJECT DETECTION
===============================

Structure:
    /Code/
    ├── MATLAB/
    │   └── color_detect.m
    ├── Python/
    │   ├── color_detect_main.py
    │   ├── color_detect_webcam.py
    ├── Videos/
    │   ├── cityscape_clip.mp4
    │   └── fruit-and-vegetable-detection.mp4
    └── README.txt  ← (this file)

How to Run:

1. MATLAB Demo (color_detect_final.m)
   - Open MATLAB R2024a or later
   - Place 'fruit-and-vegetable-detection.mp4' or any test video in the same folder
   - Run 'color_detect_final.m'
   - Annotated video and log will be saved as 'output.mp4' and 'log.csv'

2. Python Batch Processing (color_detect_main.py)
   - Requires Python 3.10+
   - Install dependencies:
       pip install opencv-python numpy imutils
   - Place a test video (e.g. 'fruit-and-vegetable-detection.mp4') in the working directory
   - Run:
       python color_detect_main.py
   - Output video and CSV log will be generated

3. Python Real-Time Detection (color_detect_webcam.py)
   - Ensure a webcam is connected
   - Run:
       python color_detect_webcam.py
   - Live window opens showing detection overlays

4. Videos Used
   - 'cityscape_clip.mp4'
   - 'fruit-and-vegetable-detection.mp4'
   (Stored in /Videos/ folder)

--------------------------------------
Team Members:
- Dhruv Goyal led the design of the color-segmentation pipeline and MATLAB implementation.
- Shree Mishra developed and tested the Python batch script and handled API-based image acquisition.
- Dhruv Goyal built the real-time webcam detector, optimized performance, and created the annotation tool for ground-truth generation.

Institution:
- Thapar Institute of Engineering & Technology

--------------------------------------
End of README
--------------------------------------
