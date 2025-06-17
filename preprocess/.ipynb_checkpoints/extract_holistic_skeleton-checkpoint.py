import os
from natsort import natsorted
from shutil import rmtree
from tqdm import tqdm
import json

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2

OUTPUT_DIR = "../datasets/wlasl_skeleton"
VIDEO_DATA_DIR = "/mnt/JapaneseDB/WLASL-master/start_kit/videos"


mp_holistic = mp.solutions.holistic


def extract_keypoints_data(detection_result):
    data = {
        "pose_landmarks":None,
        "left_hand_landmarks":None,
        "right_hand_landmarks":None,
    }
    if detection_result.pose_landmarks != None:
        data["pose_landmarks"] = np.array([[detection_result.pose_landmarks.landmark[i].x,detection_result.pose_landmarks.landmark[i].y,detection_result.pose_landmarks.landmark[i].z,detection_result.pose_landmarks.landmark[i].visibility] for i in range(0,33)]).flatten().tolist()

    if detection_result.left_hand_landmarks != None:
        data["left_hand_landmarks"] = np.array([[detection_result.left_hand_landmarks.landmark[i].x,detection_result.left_hand_landmarks.landmark[i].y,detection_result.left_hand_landmarks.landmark[i].z] for i in range(0,21)]).flatten().tolist()

    if detection_result.right_hand_landmarks != None:
        data["right_hand_landmarks"] = np.array([[detection_result.right_hand_landmarks.landmark[i].x,detection_result.right_hand_landmarks.landmark[i].y,detection_result.right_hand_landmarks.landmark[i].z] for i in range(0,21)]).flatten().tolist()

    return data
    

if os.path.exists(OUTPUT_DIR):
    rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR,exist_ok=True)

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=2) as holistic:
    for file in tqdm(os.listdir(VIDEO_DATA_DIR)):
        if file.endswith(".mp4"):
            data = {
                "pose_landmarks":[],
                "left_hand_landmarks":[],
                "right_hand_landmarks":[],
            }
    
            video_path = os.path.join(VIDEO_DATA_DIR,file)
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)
                keypoints_data = extract_keypoints_data(results)

                for key in data.keys():
                    data[key].append(keypoints_data[key])
        
            cap.release()
            with open(f"{OUTPUT_DIR}/{file.split('.')[0]}.json", 'w') as f:
                    json.dump(data, f)
        