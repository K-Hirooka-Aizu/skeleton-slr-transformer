import os
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

# My library
from dataset import (
    Sign_Dataset
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _build_wlasl_skeleton_official_dataloader(subset='asl100',num_samples=50,batch_size=32,num_workers=8,pin_memory=True,random_seed=42):
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    split_file =  '../data/official_wlasl/splits/{}.json'.format(subset)
    pose_data_root = "../data/official_wlasl/pose_per_individual_videos"

    with open(split_file, 'r') as f:
        content = json.load(f)

    glosses = sorted(np.unique([gloss_entry['gloss'] for gloss_entry in content]))

    dataloaders = dict()
    for key in ["train","valid","test"]:
        dataset = Sign_Dataset(index_file_path=split_file, 
                               split=key if key != "valid" else "val",
                               pose_root=pose_data_root,
                               img_transforms=None,
                               video_transforms=None,
                               num_samples=num_samples,
                               sample_strategy='rnd_start' if key == "train" else 'k_copies',
                               skeleton_augmentation=True if key=="train" else False
                              )
        dataloaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True if key=="train" else False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True if key=="train" else False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )

    print("="*50)
    for key in dataloaders.keys():
        print(f"{key} : {len(dataloaders[key].dataset)}")
    print("="*50)
    
    return dataloaders, len(glosses)

if __name__=="__main__":
    _build_wlasl_skeleton_official_dataloader()
