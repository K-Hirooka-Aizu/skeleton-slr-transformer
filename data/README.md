# Datasets

Place the datasets in this directory. The terms of use of each dataset are summarized in the [main README](../README.md#terms-of-use-of-the-data).

## Directory Structure
```terminal
data/
├── jslv2
│   ├── skeleton_files_jslv2.zip
│   └── skeleton_files
├── ksl0
│   ├── skeleton_files_ksl0.zip
│   └── skeleton_files
├── official_wlasl
│   ├── pose_per_individual_videos
│   ├── pose_per_individual_videos.zip
│   └── splits
└──  README.md
```

## Dataset List
- [WLASL](https://dxli94.github.io/WLASL/) Please visit official WLASL Github page. We used split config and keypoint data they provided.
    - [splits](https://drive.google.com/file/d/16CWkbMLyEbdBkrxAPaxSXFP_aSxKzNN4/view)
    - [body_keypoints](https://drive.google.com/file/d/1k5mfrc2g4ZEzzNjW6CEVjLvNTZcmPanB/view)
- JSLv2 (Japanese Fingerspelling dataset)
    - [Google Drive](https://drive.google.com/file/d/1BOtXEcBvuF2ZzgBDa4LQJj58xUQKHk_M/view?usp=sharing)
- KSL0 (Korean Fingerspelling dataset)
    - [Google Drive](https://drive.google.com/file/d/1fxoqa1WgLqtyh0N6RSHCB-mH6bcpt8LI/view?usp=sharing)

## KSL0 file format

`skeleton_files_ksl0.zip` extracts to `skeleton_files/Subject01` ... `skeleton_files/Subject20`.
Each subject folder contains `1.json` ... `31.json`, one file per character (620 files in total).

- **Signers**: each `SubjectXX` folder is one signer. Each signer performed each of the 31 characters once, so there are no separate sessions or repetitions, and each JSON comes from a different source video. No augmented or derived copies are included.
- **Classes**: the file number is the class. The characters are shown in Figure 1 of [Shin et al., IEEE Access 2024](https://ieeexplore.ieee.org/document/10529244); we believe the file numbers follow that figure, but this has not been verified.
- **Format**: each JSON is a dict `{"<frame number>": [[x, y, z] × 21]}`. The keys are consecutive integers starting from 1. Each video has 65–152 frames (median 101).
- **Keypoints**: 21 hand landmarks extracted with MediaPipe, in the MediaPipe Hand Landmarker order (0 = wrist, 1–4 = thumb, 5–8 = index, 9–12 = middle, 13–16 = ring, 17–20 = pinky).
- **Coordinates**: `x` and `y` are normalized by the video width and height (range [0, 1]); `z` is the relative depth with the wrist as the origin.
- **Missing detections**: frames where no hand was detected are NaN for all 21 × 3 values (317 of 60,803 frames). No frame has only some joints missing.
- **Unknown**: the MediaPipe version, the video resolution and frame rate, and whether frames were flipped are not recorded here; see the paper above for the recording conditions.
- **Split**: there is no official split. In our paper, the subjects were split at random into train/valid/test at 8:1:1 (signer-independent; see `KSL0LightningDataModule` in `src/sstan/datamodule.py`).

Note: `preprocess/extract_hand_skeleton.py` is a legacy script and was not used to produce the released ZIP; its output format differs.
