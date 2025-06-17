import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L

from dataset import (
    Sign_Dataset,
    WLASL_Dataset,
)
from augmentation_tools import JointMixAug

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

        self.split_file =  '../data/official_wlasl/splits/{}.json'.format(subset)
        self.pose_data_root = "../data/official_wlasl/pose_per_individual_videos"

        with open(self.split_file, 'r') as f:
            content = json.load(f)
        glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

        self.num_classes = len(glosses)


        def collate_fn(batch):
            data, label = list(zip(*batch))

            data = torch.stack(data)
            label = torch.tensor(label,dtype=torch.long)
            
            if label.dim()==1:
                label_onehot = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
            else:
                label_onehot = label

            data, label_onehot = JointMixAug(data, label_onehot)
            return data, label_onehot
        

        self.collate_fn = collate_fn
        

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

        self.split_file =  '../data/official_wlasl/splits/{}.json'.format(subset)
        self.pose_data_root = "../data/official_wlasl/skeleton_mmpose"

        with open(self.split_file, 'r') as f:
            content = json.load(f)
        glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

        self.num_classes = len(glosses)


        def collate_fn(batch):
            data, label = list(zip(*batch))

            data = torch.stack(data)
            label = torch.tensor(label,dtype=torch.long)
            
            if label.dim()==1:
                label_onehot = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
            else:
                label_onehot = label

            data, label_onehot = JointMixAug(data, label_onehot)
            return data, label_onehot
        

        self.collate_fn = collate_fn

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
    
    