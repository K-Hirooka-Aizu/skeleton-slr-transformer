import json
import os
from pathlib import Path
from typing import List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
from tqdm import tqdm

from .utils.sampling_func import (
    sequential_sampling,
    rand_start_sampling,
    k_copies_fixed_length_sequential_sampling
)

# Type alias for the wlasl video dataset item structure : (vid, gloss, video_path, start_frame_index, end_frame_index)
DatasetItem = Tuple[str, str, str, int, int]

class WLASLVideoDataset(data_utl.Dataset):
    gloss2index:dict[str, int]|None = None

    def __init__(self, split_file_path:str, split:str, video_dir_path:str, seq_len:int, sampling_strategy:str, transforms=None)->None:
        self.split_file_path = split_file_path
        self.split = split
        self.video_dir_path = video_dir_path
        self.seq_len = seq_len
        self.sampling_strategy = sampling_strategy
        self.transforms = transforms
        self.num_copies = 4

        self.samples:list[DatasetItem] = make_datast_remake(
            split_file=self.split_file_path,
            split=self.split,
            root=self.video_dir_path,
        )
        
        glosses_set = get_gloss_set(self.samples)
        self.num_classes = len(glosses_set)
        if WLASLVideoDataset.gloss2index is None:
            WLASLVideoDataset.gloss2index = {gloss:i for i, gloss in enumerate(glosses_set)}

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        vid, gloss, video_path, start_frame_index, end_frame_index = self.samples[idx]

        # one-hot encode
        onehot_ndarray = np.identity(self.num_classes,dtype=np.float32)
        onehot_tensor = torch.from_numpy(onehot_ndarray[WLASLVideoDataset.gloss2index[gloss]])

        frames_ndarray = self._load_video(video_path=video_path, start_frame_index=start_frame_index, end_frmae_index=end_frame_index)
        if self.sampling_strategy == 'rnd_start':
            frames_to_sample = rand_start_sampling(0, end_frame_index-start_frame_index, self.seq_len)
        elif self.sampling_strategy == 'seq':
            frames_to_sample = sequential_sampling(0, end_frame_index-start_frame_index, self.seq_len)
        elif self.sampling_strategy == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(0, end_frame_index-start_frame_index, self.seq_len, self.num_copies)
        else:
            raise RuntimeError('Unimplemented sample strategy found: {}.'.format(self.sampling_strategy))
        print(frames_ndarray.shape)
        frames_ndarray = frames_ndarray[frames_to_sample]

        frames_ndarray /= 255.
        frames_tensor = video_to_tensor(frames_ndarray)

        if self.transforms:
            frames_tensor = self.transforms(frames_tensor)
        return frames_tensor, onehot_tensor

    @staticmethod
    def _load_video(video_path:str, start_frame_index:int|None=None, end_frmae_index:int|None=None)->np.ndarray:
        if start_frame_index is not None and end_frmae_index is not None and end_frmae_index < start_frame_index:
            raise RuntimeError(f"end_frame_index should be greater than start_frame_index. Currently end_frmae_index:({end_frmae_index}) < start_frame_index:({start_frame_index}) were assigned.")
        vidcap = cv2.VideoCapture(video_path)

        frames: List[np.ndarray] = []

        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        vid = Path(video_path).stem
        save_dir_path = f"./images/{vid}"
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path,exist_ok=True)
            print(f"DEBUG {total_frames}")
            for i in range(total_frames):
                success, img = vidcap.read()
                if not success:
                    print("Failed to read video or ")
                    break

                cv2.imwrite(os.path.join(save_dir_path,f"image_{str(i).zfill(5)}.jpg"), img)
        vidcap.release()
        
        for i in range(end_frmae_index-start_frame_index+1):
            if not os.path.exists(os.path.join(save_dir_path,f"image_{str(i).zfill(5)}.jpg")):
                raise RuntimeError(f"{os.path.join(save_dir_path,f'image_{str(i).zfill(5)}.jpg')} is not exists.")
            
            img = cv2.imread(os.path.join(save_dir_path,f"image_{str(i).zfill(5)}.jpg"))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)

        frames_ndarray = np.stack(frames,dtype=np.float32)

        return frames_ndarray
    

def get_gloss_set(dataset:List[DatasetItem]):

    glosses = set()
    for vid, gloss, video_path, start_index, end_index in dataset:
        glosses.add(gloss)

    return glosses



def make_datast_remake(split_file: str, split: str, root:str, mode:str|None=None, num_classes:int|None=None) -> List[DatasetItem]:
    """
    Collect the meta information for data matching the specified split and store it in a List.

    
    :param split_file: json file path for wlasl dataset
    :type split_file:  str
    :param split: The category for the dataset being created (assigning “train” will build a dataset containing only instances assigned to the “train” split), {"train", "val", "test"}
    :type split: str 
    :param root: Directory path where MP4 files are stored.
    :type root: str
    :param mode: (this param currently unused)
    :type mode: str
    :param num_classes:  (this param currently unused)
    :type num_classes: int
    :return: The list of one sample data information (vid, label[gloss], video_path, frame_start, frame_end).
    :rtype: List[DatasetItem] (list[str, str, str, int, int])
    """
    dataset: List[DatasetItem] = []
    with open(split_file, "r") as f:
        meta_info: list[dict[str,Any]] = json.load(f)

    for one_glose_meta_info in tqdm(meta_info):

        gloss = one_glose_meta_info.get("gloss",None)
        if gloss is None:
            raise RuntimeError("Encounting an None in gloss information.")
        
        instances:list[dict[str,Any]] = one_glose_meta_info.get("instances",None)
        if instances is None:
            raise RuntimeError(f"gloss [{gloss}] has no instance.")
        
        for _instance in instances:
            frame_start = _instance.get("frame_start", None)
            frame_end = _instance.get("frame_end", None)
            split_category = _instance.get("split", None)
            vid = _instance.get("video_id", None)

            if frame_start is None or frame_end is None or split_category is None or vid is None:
                raise RuntimeError(f"Found Missing information. {vid}: {split_category}: {frame_start} ~ {frame_end}")
            
            if split_category not in ["train", "val", "test"]:
                raise RuntimeError(f"Encountering an unknown split category. {split_category}")

            num_frames = frame_end - frame_start
            video_path = os.path.join(root,f"{vid}.mp4")
            cap = cv2.VideoCapture(video_path)
            real_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release() # Best practice to release resource

            if real_num_frames < num_frames:
                RuntimeError(f"{vid}.mp4 has {real_num_frames} frames. But, this video has {num_frames} frames according to json file.")

            if os.path.exists(os.path.join(root,f"{vid}.mp4")):
                RuntimeError(f"{os.path.join(root,f'{vid}.mp4')} is not found.")
            
            if split_category == split:
                dataset.append((vid, gloss, os.path.join(root,f"{vid}.mp4"), frame_start, frame_end))
            else:
                continue

    return dataset


def video_to_tensor(pic: np.ndarray) -> torch.Tensor:
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
          pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
          Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))



if __name__=="__main__":
    # dataset = NSLT(
    #     split_file="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/splits/asl100.json",
    #     split="train",
    #     root="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/video",
    #     mode="rgb",
    #     transforms=None,
    #     )
    # make_datast_remake(
    #     split_file="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/splits/asl100.json",
    #     split="val",
    #     root="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/video",
    #     mode="rgb",
    #     num_classes=0
    #     )

    dataset = WLASLVideoDataset(
        split_file_path="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/splits/asl100.json",
        split="train",
        video_dir_path="/home/hirooka/transformer-based-sign-language-recognition/data/official_wlasl/video",
        seq_len=64,
        sampling_strategy="rnd_start",
        transforms=None
    )
    print(WLASLVideoDataset.gloss2index.keys())
    print(len(dataset))
    for i, (data, label) in enumerate(tqdm(dataset)):
        print(f"{type(data)}: {data.shape}")
        print(f"{type(label)}: {label.shape}")