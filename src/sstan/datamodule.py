import os
import json
from functools import partial
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf

from .dataset import (
    Sign_Dataset,
    WLASL_Dataset,
    KSL0_Skeleton_Dataset,
    JSLV2_Skeleton_Dataset,
)
from .augmentation_tools import JointMixAug
from .video_dataset import (
    WLASLVideoDataset,
    WLASLVideoDatasetWithHuggingFace
)
from .video_transforms import build_transforms_from_config

def build_lightning_data_module(cfg):
    dataset_name = cfg.data.dataset

    if dataset_name in ["wlasl100", "wlasl300", "wlasl1000", "wlasl2000"]:
        return WLASLOpenposeLightningDataModule(
            subset=cfg.data.subset,
            seq_len=cfg.data.seq_len,
            num_copies=cfg.data.num_copies,
            sampling_strategy=cfg.data.sampling_strategy,
            train_data_augmentation=cfg.data.train_data_augmentation,
            batch_size=cfg.batch_size,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers
        )
    elif dataset_name in ["wlasl100_mmpose", "wlasl300_mmpose", "wlasl1000_mmpose", "wlasl2000_mmpose"]:
        return WLASLMMPoseLightningDataModule(
            subset=cfg.data.subset,
            seq_len=cfg.data.seq_len,
            num_copies=cfg.data.num_copies,
            sampling_strategy=cfg.data.sampling_strategy,
            train_data_augmentation=cfg.data.train_data_augmentation,
            batch_size=cfg.batch_size,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers
        )

    elif dataset_name in ["ksl0"]:
        return KSL0LightningDataModule(
            dataset_dir_path=cfg.data.dataset_dir_path,
            split_ratio=cfg.data.split_ratio,
            seq_len=cfg.data.seq_len,
            num_copies=cfg.data.num_copies,
            sampling_strategy=cfg.data.sampling_strategy,
            batch_size=cfg.batch_size,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
    
    elif dataset_name in ["jsl0"]:
        return JSL0LightningDataModule(
            dataset_dir_path=cfg.data.dataset_dir_path,
            split_ratio=cfg.data.split_ratio,
            seq_len=cfg.data.seq_len,
            num_copies=cfg.data.num_copies,
            sampling_strategy=cfg.data.sampling_strategy,
            batch_size=cfg.batch_size,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
    elif dataset_name in ["wlasl100_rgb", "wlasl300_rgb", "wlasl1000_rgb", "wlasl2000_rgb"]:
        return WLASLVideolightningDataModule(
            subset=cfg.data.subset,
            seq_len=cfg.data.seq_len,
            num_copies=cfg.data.num_copies,
            sampling_strategy=cfg.data.sampling_strategy,
            transforms_config=cfg.data.transforms,
            batch_size=cfg.batch_size,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers
        )
    elif dataset_name in ["wlasl100_rgb_hf", "wlasl300_rgb_hf", "wlasl1000_rgb_hf", "wlasl2000_rgb_hf"]:
        return WLASLVideoWithHuggingFacelightningDataModule(
            huggingface_model_name=cfg.model.fuggingface_model_name,
            subset=cfg.data.subset,
            seq_len=cfg.data.seq_len,
            num_copies=cfg.data.num_copies,
            sampling_strategy=cfg.data.sampling_strategy,
            transforms_config=cfg.data.transforms,
            batch_size=cfg.batch_size,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers
        )

    else:
        raise RuntimeError(f"Dataset [{dataset_name}] is not implemented.")

class WLASLOpenposeLightningDataModule(L.LightningDataModule):
    def __init__(self,subset:str, seq_len:int=50, num_copies:int=4, sampling_strategy:dict[str,str]={"train":"rnd_start"}, train_data_augmentation:bool=False, batch_size:int=16, pin_memory:bool=False, num_workers:int=4):
        super().__init__()
        self.subset = subset
        self.seq_len = seq_len
        self.num_copies = num_copies
        self.sampling_strategy = sampling_strategy
        self.train_data_augmentation = train_data_augmentation

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.split_file =  './data/official_wlasl/splits/{}.json'.format(subset)
        self.pose_data_root = "./data/official_wlasl/pose_per_individual_videos"

        with open(self.split_file, 'r') as f:
            content = json.load(f)
        glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

        self.num_classes = len(glosses)

        self.collate_fn = partial(collate_fn, num_classes=self.num_classes)
        

    def setup(self, stage):
        self.train_dataset = Sign_Dataset(index_file_path=self.split_file, pose_root=self.pose_data_root, split="train", num_samples=self.seq_len, num_copies=self.num_copies, sample_strategy=self.sampling_strategy["train"], skeleton_augmentation=self.train_data_augmentation)
        self.valid_dataset = Sign_Dataset(index_file_path=self.split_file, pose_root=self.pose_data_root, split="val", num_samples=self.seq_len, num_copies=self.num_copies, sample_strategy=self.sampling_strategy["valid"], skeleton_augmentation=False)
        self.test_dataset = Sign_Dataset(index_file_path=self.split_file, pose_root=self.pose_data_root, split="test", num_samples=self.seq_len, num_copies=self.num_copies, sample_strategy=self.sampling_strategy["test"], skeleton_augmentation=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
    
    

class WLASLMMPoseLightningDataModule(L.LightningDataModule):
    def __init__(self,subset:str, seq_len:int=50, num_copies:int=4, sampling_strategy:dict[str,str]={"train":"rnd_start"}, train_data_augmentation:bool=False, batch_size:int=16, pin_memory:bool=False, num_workers:int=4):
        super().__init__()
        self.subset = subset
        self.seq_len = seq_len
        self.num_copies = num_copies
        self.sampling_strategy = sampling_strategy
        self.train_data_augmentation = train_data_augmentation

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.split_file =  './data/official_wlasl/splits/{}.json'.format(subset)
        self.pose_data_root = "./data/official_wlasl/skeleton_mmpose"

        with open(self.split_file, 'r') as f:
            content = json.load(f)
        glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

        self.num_classes = len(glosses)

        self.collate_fn = partial(collate_fn, num_classes=self.num_classes)
        

    def setup(self, stage):
        self.train_dataset = WLASL_Dataset(index_file_path=self.split_file, pose_root=self.pose_data_root, split="train", num_samples=self.seq_len, num_copies=self.num_copies, sample_strategy=self.sampling_strategy["train"], skeleton_augmentation=self.train_data_augmentation)
        self.valid_dataset = WLASL_Dataset(index_file_path=self.split_file, pose_root=self.pose_data_root, split="val", num_samples=self.seq_len, num_copies=self.num_copies, sample_strategy=self.sampling_strategy["valid"], skeleton_augmentation=False)
        self.test_dataset = WLASL_Dataset(index_file_path=self.split_file, pose_root=self.pose_data_root, split="test", num_samples=self.seq_len, num_copies=self.num_copies, sample_strategy=self.sampling_strategy["test"], skeleton_augmentation=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
    
class KSL0LightningDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir_path:str, split_ratio:list[float|int, float|int, float|int]=[0.6,0.2,0.2],seq_len:int=50, num_copies:int=4, sampling_strategy:dict[str,str]={"train":"rnd_start"}, train_data_augmentation:bool=False, batch_size:int=16, pin_memory:bool=False, num_workers:int=4, seed:int=42):
        super().__init__()

        assert len(split_ratio)==3 and sum(split_ratio)==1, f"args : split_ration should have 3 flaot or int elements. e.g. [0.6, 0.2, 0.2]"

        self.dataset_dir_path = dataset_dir_path
        self.split_ratio = split_ratio
        self.seq_len = seq_len
        self.num_copies = num_copies
        self.sampling_strategy = sampling_strategy
        self.train_data_augmentation = train_data_augmentation
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.seed = seed
       

    def setup(self, stage):
        # DATASET_PATH = "../datasets/ksl0/skeleton_files"
        DATASET_PATH = self.dataset_dir_path
        remove_target = []
        subject_list = [os.path.join(DATASET_PATH,subject_dir) for subject_dir in os.listdir(DATASET_PATH) if subject_dir.startswith("Subject") and "checkpoint" not in os.path.join(DATASET_PATH,subject_dir)]

        train_set, other_set = train_test_split(subject_list,test_size=sum(self.split_ratio[1:])/sum(self.split_ratio),shuffle=True,random_state=self.seed)
        valid_set, test_set = train_test_split(other_set,test_size=self.split_ratio[-1]/sum(self.split_ratio[1:]),shuffle=True,random_state=self.seed)

        sign_name = list(set([os.path.splitext(os.path.basename(file))[0] for root,dirs,files in os.walk(DATASET_PATH) for file in files if file.endswith(".json") and "checkpoint" not in file]))
        label2num = {sign_name:i for i,sign_name in enumerate(sign_name)}

        train_json_file_list = [os.path.join(root,file) for subject_path in train_set for root,dirs,files in os.walk(subject_path) for file in files if file.endswith(".json") and "checkpoint" not in os.path.join(root,file)]
        train_json_file_list = [path for path in train_json_file_list if os.path.splitext(os.path.basename(path))[0] not in remove_target]
        train_label = [label2num[os.path.splitext(os.path.basename(json_path))[0]] for json_path in train_json_file_list]
        
        valid_json_file_list = [os.path.join(root,file) for subject_path in valid_set for root,dirs,files in os.walk(subject_path) for file in files if file.endswith(".json") and "checkpoint" not in os.path.join(root,file)]
        valid_json_file_list = [path for path in valid_json_file_list if os.path.splitext(os.path.basename(path))[0] not in remove_target]
        valid_label = [label2num[os.path.splitext(os.path.basename(json_path))[0]] for json_path in valid_json_file_list]
        
        test_json_file_list = [os.path.join(root,file) for subject_path in test_set for root,dirs,files in os.walk(subject_path) for file in files if file.endswith(".json") and "checkpoint" not in os.path.join(root,file)]
        test_json_file_list = [path for path in test_json_file_list if os.path.splitext(os.path.basename(path))[0] not in remove_target]
        test_label = [label2num[os.path.splitext(os.path.basename(json_path))[0]] for json_path in test_json_file_list]

        self.train_dataset = KSL0_Skeleton_Dataset(train_json_file_list, train_label, sample_strategy=self.sampling_strategy["train"], seq_len=self.seq_len, augmentation=self.train_data_augmentation)
        self.valid_dataset = KSL0_Skeleton_Dataset(valid_json_file_list, valid_label, sample_strategy=self.sampling_strategy["valid"], seq_len=self.seq_len, augmentation=False)
        self.test_dataset = KSL0_Skeleton_Dataset(test_json_file_list, test_label, sample_strategy=self.sampling_strategy["test"], seq_len=self.seq_len, augmentation=False)

        self.num_classes = len(label2num)
        self.collate_fn = partial(collate_fn, num_classes=self.num_classes)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
    

class JSL0LightningDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir_path:str, split_ratio:list[float|int, float|int, float|int]=[0.6,0.2,0.2],seq_len:int=50, num_copies:int=4, sampling_strategy:dict[str,str]={"train":"rnd_start"}, train_data_augmentation:bool=False, batch_size:int=16, pin_memory:bool=False, num_workers:int=4, seed:int=42):
        super().__init__()

        assert len(split_ratio)==3 and sum(split_ratio)==1, f"args : split_ration should have 3 flaot or int elements. e.g. [0.6, 0.2, 0.2]"

        self.dataset_dir_path = dataset_dir_path
        self.split_ratio = split_ratio
        self.seq_len = seq_len
        self.num_copies = num_copies
        self.sampling_strategy = sampling_strategy
        self.train_data_augmentation = train_data_augmentation
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.seed = seed
       

    def setup(self, stage):
        DATASET_PATH = self.dataset_dir_path
        remove_target = []
        # if you want to compare with Kakizaki et al. paper, uncomment below line.
        # remove_target = ["".join([boin,siin]) for boin in ["g","z","d","b","p"] for siin in ["a","i","u","e","o"]] + ["xya","xyu","xyo""xtsu","long"]
        print(remove_target)
            
        subject_list = [os.path.join(DATASET_PATH,subject_dir) for subject_dir in os.listdir(DATASET_PATH) if subject_dir.endswith("_jslv2") and "checkpoint" not in os.path.join(DATASET_PATH,subject_dir)]

        train_set, other_set = train_test_split(subject_list,test_size=sum(self.split_ratio[1:])/sum(self.split_ratio),shuffle=True,random_state=self.seed)
        valid_set, test_set = train_test_split(other_set,test_size=self.split_ratio[-1]/sum(self.split_ratio[1:]),shuffle=True,random_state=self.seed)

        sign_name = list(set([os.path.splitext(os.path.basename(file))[0] for root,dirs,files in os.walk(DATASET_PATH) for file in files if file.endswith(".json") and "checkpoint" not in file]))
        label2num = {sign_name:i for i,sign_name in enumerate(sign_name)}

        train_json_file_list = [os.path.join(root,file) for subject_path in train_set for root,dirs,files in os.walk(subject_path) for file in files if file.endswith(".json") and "checkpoint" not in os.path.join(root,file)]
        train_json_file_list = [path for path in train_json_file_list if os.path.splitext(os.path.basename(path))[0] not in remove_target]
        train_label = [label2num[os.path.splitext(os.path.basename(json_path))[0]] for json_path in train_json_file_list]
        
        valid_json_file_list = [os.path.join(root,file) for subject_path in valid_set for root,dirs,files in os.walk(subject_path) for file in files if file.endswith(".json") and "checkpoint" not in os.path.join(root,file)]
        valid_json_file_list = [path for path in valid_json_file_list if os.path.splitext(os.path.basename(path))[0] not in remove_target]
        valid_label = [label2num[os.path.splitext(os.path.basename(json_path))[0]] for json_path in valid_json_file_list]
        
        test_json_file_list = [os.path.join(root,file) for subject_path in test_set for root,dirs,files in os.walk(subject_path) for file in files if file.endswith(".json") and "checkpoint" not in os.path.join(root,file)]
        test_json_file_list = [path for path in test_json_file_list if os.path.splitext(os.path.basename(path))[0] not in remove_target]
        test_label = [label2num[os.path.splitext(os.path.basename(json_path))[0]] for json_path in test_json_file_list]

        self.train_dataset = JSLV2_Skeleton_Dataset(train_json_file_list, train_label, sample_strategy=self.sampling_strategy["train"], seq_len=self.seq_len, augmentation=self.train_data_augmentation)
        self.valid_dataset = JSLV2_Skeleton_Dataset(valid_json_file_list, valid_label, sample_strategy=self.sampling_strategy["valid"], seq_len=self.seq_len, augmentation=False)
        self.test_dataset = JSLV2_Skeleton_Dataset(test_json_file_list, test_label, sample_strategy=self.sampling_strategy["test"], seq_len=self.seq_len, augmentation=False)

        self.num_classes = len(label2num)
        self.collate_fn = partial(collate_fn, num_classes=self.num_classes)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
    

def collate_fn(batch, num_classes):
    data = torch.stack([b["skeleton_data"] for b in batch])
    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    if label.dim() == 1:
        label = torch.nn.functional.one_hot(
            label, num_classes=num_classes
        ).float()

    data, label = JointMixAug(data, label)

    return {
        "skeleton_data": data,
        "label": label
    }

class WLASLVideolightningDataModule(L.LightningDataModule):
    def __init__(self,subset:str, seq_len:int=50, num_copies:int=4, sampling_strategy:dict[str,str]={"train":"rnd_start"}, transforms_config:OmegaConf|None=None, batch_size:int=16, pin_memory:bool=False, num_workers:int=4):
        super().__init__()
        self.subset = subset
        self.seq_len = seq_len
        self.num_copies = num_copies
        self.sampling_strategy = sampling_strategy

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.split_file =  './data/official_wlasl/splits/{}.json'.format(subset)
        self.video_dir_path = "./data/official_wlasl/video"

        with open(self.split_file, 'r') as f:
            content = json.load(f)
        glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

        self.num_classes = len(glosses)

        self.transforms_dict = {
            "train":build_transforms_from_config(transforms_config.train),
            "val":build_transforms_from_config(transforms_config.val),
            "test":build_transforms_from_config(transforms_config.test),
        }

    def setup(self, stage):
        self.train_dataset = WLASLVideoDataset(split_file_path=self.split_file, video_dir_path=self.video_dir_path, split="train", seq_len=self.seq_len, num_copies=self.num_copies, sampling_strategy=self.sampling_strategy["train"], transforms=self.transforms_dict["train"])
        self.valid_dataset = WLASLVideoDataset(split_file_path=self.split_file, video_dir_path=self.video_dir_path, split="val", seq_len=self.seq_len, num_copies=self.num_copies, sampling_strategy=self.sampling_strategy["valid"], transforms=self.transforms_dict["val"])
        self.test_dataset = WLASLVideoDataset(split_file_path=self.split_file, video_dir_path=self.video_dir_path, split="test", seq_len=self.seq_len, num_copies=self.num_copies, sampling_strategy=self.sampling_strategy["test"], transforms=self.transforms_dict["test"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
    
class WLASLVideoWithHuggingFacelightningDataModule(L.LightningDataModule):
    def __init__(self,huggingface_model_name:str, subset:str, seq_len:int=50, num_copies:int=4, sampling_strategy:dict[str,str]={"train":"rnd_start"}, transforms_config:OmegaConf|None=None, batch_size:int=16, pin_memory:bool=False, num_workers:int=4):
        super().__init__()
        self.huggingface_model_name = huggingface_model_name
        self.subset = subset
        self.seq_len = seq_len
        self.num_copies = num_copies
        self.sampling_strategy = sampling_strategy

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.split_file =  './data/official_wlasl/splits/{}.json'.format(subset)
        self.video_dir_path = "./data/official_wlasl/video"

        with open(self.split_file, 'r') as f:
            content = json.load(f)
        glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

        self.num_classes = len(glosses)

        self.transforms_dict = {
            "train":build_transforms_from_config(transforms_config.train),
            "val":build_transforms_from_config(transforms_config.val),
            "test":build_transforms_from_config(transforms_config.test),
        }

    def setup(self, stage):
        self.train_dataset = WLASLVideoDatasetWithHuggingFace(model_name=self.huggingface_model_name, split_file_path=self.split_file, video_dir_path=self.video_dir_path, split="train", seq_len=self.seq_len, num_copies=self.num_copies, sampling_strategy=self.sampling_strategy["train"], transforms=self.transforms_dict["train"])
        self.valid_dataset = WLASLVideoDatasetWithHuggingFace(model_name=self.huggingface_model_name, split_file_path=self.split_file, video_dir_path=self.video_dir_path, split="val", seq_len=self.seq_len, num_copies=self.num_copies, sampling_strategy=self.sampling_strategy["valid"], transforms=self.transforms_dict["val"])
        self.test_dataset = WLASLVideoDatasetWithHuggingFace(model_name=self.huggingface_model_name, split_file_path=self.split_file, video_dir_path=self.video_dir_path, split="test", seq_len=self.seq_len, num_copies=self.num_copies, sampling_strategy=self.sampling_strategy["test"], transforms=self.transforms_dict["test"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
    