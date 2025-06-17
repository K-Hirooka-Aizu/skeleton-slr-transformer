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


def extract_keypoints_data(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    # Loop through the detected hands to visualize.
    keypoints_data = None
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        keypoints_data = [[lm.x,lm.y,lm.z] for lm in hand_landmarks_proto.landmark]

    return keypoints_data


DATASET_DIR = "/mnt/JapaneseDB/research_archive/ksl0/KoreanSignDataset/KSL0_dataset"
OUTPUT_DIR = "../datasets/ksl0_skeleton_json_files"

if os.path.exists(OUTPUT_DIR):
    rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR,exist_ok=True)

for subject_dir in os.listdir(DATASET_DIR):
    if subject_dir.startswith("KSL0"):

        os.makedirs(os.path.join(OUTPUT_DIR,subject_dir),exist_ok=True)
        for video_name in tqdm(natsorted(os.listdir(os.path.join(DATASET_DIR,subject_dir)))):
            if video_name.endswith(".mp4"):
                data = {"xyz":[],"label":None}
                data["label"] = int(video_name.split(".")[0])-1
                
                video_path = os.path.join(DATASET_DIR,subject_dir,video_name)

                # STEP 2: Create an HandLandmarker object.
                base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
                options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=1)
                detector = vision.HandLandmarker.create_from_options(options)

                cap = cv2.VideoCapture(video_path)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame,1)
                    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    detection_result = detector.detect(frame)
                    
                    keypoints = extract_keypoints_data(detection_result)
                    data["xyz"].append(np.array(keypoints).flatten().tolist() if keypoints != None else None)

            with open(f"{OUTPUT_DIR}/{subject_dir}/{video_name.split('.')[0]}.json", 'w') as f:
                json.dump(data, f)

        cap.release()
        

                    