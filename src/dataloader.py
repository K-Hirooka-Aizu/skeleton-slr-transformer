import os
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

# My library
from dataset import (
    Sign_Dataset
)
from augmentation_tools import JointMixAug

def build_dataloader(cfg: DictConfig, **kwargs):
    if cfg.data.dataset == "wlasl":
        datasets, cfg.data.num_classes = _build_wlasl_skeleton_official_dataloader(
            subset=cfg.data.subset, seq_len=cfg.data.seq_len, num_copies=cfg.data.num_copies, sampling_strategy=cfg.data.sampling_strategy, data_augmentation=cfg.data.data_augmentation
            )
    else:
        raise RuntimeError(f"Dataset [{cfg.data.dataset}] is not implemented.")

    def collate_fn(batch):
        data, label = list(zip(*batch))

        data = torch.stack(data)
        label = torch.tensor(label,dtype=torch.long)
        
        if label.dim()==1:
            label_onehot = torch.nn.functional.one_hot(label, num_classes=cfg.data.num_classes).float()
        else:
            label_onehot = label

        data, label_onehot = JointMixAug(data, label_onehot)
        return data, label_onehot
    
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    dataloaders=dict()
    for key in datasets.keys():
        dataloaders[key] = DataLoader(
                    datasets[key],
                    batch_size=cfg.batch_size,
                    shuffle=True if key=="train" else False,
                    num_workers=cfg.num_workers,
                    collate_fn=collate_fn if key=="train" else None,
                    pin_memory=cfg.pin_memory,
                    drop_last=True if key=="train" else False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
    return dataloaders

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _build_wlasl_skeleton_official_dataloader(subset:str='asl100',seq_len:int=50, num_copies:int=4, sampling_strategy:dict=None, data_augmentation:bool=True, **kwargs):
    
    split_file =  '../data/official_wlasl/splits/{}.json'.format(subset)
    pose_data_root = "../data/official_wlasl/pose_per_individual_videos"

    with open(split_file, 'r') as f:
        content = json.load(f)
    glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

    datasets = dict()
    for key in ["train","valid","test"]:
        datasets[key] = Sign_Dataset(
            index_file_path=split_file, 
            pose_root=pose_data_root,
            split=key if key != "valid" else "val",
            num_samples=seq_len,
            num_copies=num_copies,
            sample_strategy=sampling_strategy[key],
            skeleton_augmentation=data_augmentation if key=="train" else False
            )
        
    print("="*50)
    for key in datasets.keys():
        print(f"{key} : {len(datasets[key])}")
    print("="*50)
    
    return datasets, len(glosses)

if __name__=="__main__":
    _build_wlasl_skeleton_official_dataloader()
