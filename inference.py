import glob 
import cv2 
import time 

import numpy as np 
from yolov5s import YoLov5TRT

images = glob.glob('sample_images/*.jpg')
yolo = YoLov5TRT(checkpoint_path='build/yolov5s.engine', device_num=0)

timestamps = []

for image in images:
    img = cv2.imread(image)
    
    start = time.time()
    results = yolo.infer(raw_image=img)
    end = time.time()
    
    timestamps.append(end-start)

print(f"FPS: {1/(np.mean(timestamps)):.2f}")