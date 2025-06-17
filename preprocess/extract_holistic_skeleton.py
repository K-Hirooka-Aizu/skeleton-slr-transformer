import os
from shutil import rmtree
from tqdm import tqdm
import json
from pathlib import Path

# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# from pose_module.Mediapipe import Model
from pose_module.RTMWPose import Model

import numpy as np
import cv2

OUTPUT_DIR = "/mnt/content/DATASET/WLASL-master/skeleton_mmpose"
VIDEO_DATA_DIR = "/mnt/content/DATASET/WLASL-master/WLASL2000"

model = Model()
    
if os.path.exists(OUTPUT_DIR):
    rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR,exist_ok=True)

for root,dirs,files in os.walk(VIDEO_DATA_DIR):
    for file in tqdm(files):
        if file.endswith(".mp4") and "checkpoint" not in os.path.join(root,file):

            data = {
                "keypoints":[],
                "img_size":[],
            }
            video_path = os.path.join(root,file)
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                keypoint, H, W = model(frame)
                data["keypoints"].append(keypoint.tolist())
                data["img_size"].append((H,W))

            cap.release()
            with open(f"{OUTPUT_DIR}/{Path(video_path).stem}.json", 'w') as f:
                    json.dump(data, f)
    