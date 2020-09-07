import cv2, sys
import numpy as np

if len(sys.argv) != 5:
    sys.exit("Usage: input_video input_resolution output_video output_resolution")

input_video = sys.argv[1]
input_resolution = str(sys.argv[2])
output_video = sys.argv[3]
output_resolution = str(sys.argv[4])

resolution_dict = {'2160p': (3840, 2160), 
     '1440p': (2560, 1440),
     '1080p': (1920, 1080),
     '720p':  (1280, 720),
     '480p':  (854, 480),
     '360p':  (640, 360),
     '240p':  (426, 240)}


FPS = 30
INPUT_SCALE = resolution_dict[input_resolution]
TARGET_SCALE = resolution_dict[output_resolution]
print(f"Changing resolution from {input_resolution}: {INPUT_SCALE} to {output_resolution}: {TARGET_SCALE}")

cap = cv2.VideoCapture(input_video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, FPS, INPUT_SCALE)

idx = 0
while True:
    ret, frame = cap.read()
    if idx % 500 == 0:
        print(f"Processed {idx} frames")
    if ret == True:
        b = cv2.resize(frame, TARGET_SCALE, fx=0, fy=0, interpolation = cv2.INTER_LINEAR)
        b = cv2.resize(b, INPUT_SCALE, fx=0, fy=0, interpolation = cv2.INTER_LINEAR)
        out.write(b)
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()