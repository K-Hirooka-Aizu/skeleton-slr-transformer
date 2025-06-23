import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# My library
from utils.sampling_func import (
    rand_start_sampling, 
    sequential_sampling, 
    k_copies_fixed_length_sequential_sampling
)
from augmentation_tools import augment_skeleton

class Sign_Dataset(Dataset):
    def __init__(self, index_file_path, split, pose_root, sample_strategy='rnd_start', num_samples=25, num_copies=4, img_transforms=None, video_transforms=None, test_index_file=None,skeleton_augmentation=True):
        
        assert os.path.exists(index_file_path), "Non-existent indexing file path: {}.".format(index_file_path)
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        self.data = []
        self.glosses = []
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(categories='auto')

        if type(split) == 'str':
            split = [split]

        self.test_index_file = test_index_file
        self._make_dataset(index_file_path, split)

        self.index_file_path = index_file_path
        self.pose_root = pose_root
        self.framename = 'image_{}_keypoints.json'
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples

        self.img_transforms = img_transforms
        self.video_transforms = video_transforms
        self.skeleton_augmentation = skeleton_augmentation

        self.num_copies = num_copies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end = self.data[index]

        x = self._load_poses(video_id, frame_start, frame_end, self.sample_strategy, self.num_samples)
        
        x = torch.reshape(x,(-1,self.num_samples if self.sample_strategy!='k_copies' else self.num_samples*self.num_copies ,2)).permute(2,1,0).unsqueeze(-1)
        
        if self.video_transforms:
            x = self.video_transforms(x)
        if self.skeleton_augmentation:
            x = torch.from_numpy(augment_skeleton(x.numpy()))
        
        # y = torch.Tensor(gloss_cat).to(torch.int64)
        y = gloss_cat
        return x, y

    def _make_dataset(self, index_file_path, split):
        with open(index_file_path, 'r') as f:
            content = json.load(f)

        # create label encoder
        glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])

        self.label_encoder.fit(glosses)
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))
            
        if self.test_index_file is not None:
            print('Trained on {}, tested on {}'.format(index_file_path, self.test_index_file))
            with open(self.test_index_file, 'r') as f:
                content = json.load(f)

        # make dataset
        for gloss_entry in content:
            gloss, instances = gloss_entry['gloss'], gloss_entry['instances']

            gloss_cat = labels2cat(self.label_encoder, [gloss])[0]
                
            for instance in instances:
                if instance['split'] not in split:
                    continue

                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.glosses.append(gloss)
                self.data.append(instance_entry)

    def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples):
        """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
         """
        poses = []

        if sample_strategy == 'rnd_start':
            frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'seq':
            frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
                                                                         self.num_copies)
        else:
            raise RuntimeError('Unimplemented sample strategy found: {}.'.format(sample_strategy))

        for i in frames_to_sample:
            pose_path = os.path.join(self.pose_root, video_id, self.framename.format(str(i).zfill(5)))
            pose = self._read_pose_file(pose_path)

            if pose is not None:
                if self.img_transforms:
                    pose = self.img_transforms(pose)

                poses.append(pose)
            else:
                try:
                    poses.append(poses[-1])
                except IndexError:
                    print(pose_path)

        pad = None

        if len(poses) < num_samples:
            num_padding = num_samples - len(frames_to_sample)
            last_pose = poses[-1]
            pad = last_pose.repeat(1, num_padding)

        poses_across_time = torch.cat(poses, dim=1)
        if pad is not None:
            poses_across_time = torch.cat([poses_across_time, pad], dim=1)

        return poses_across_time

    def _compute_difference(self,x):
        diff = []
        for i, xx in enumerate(x):
            temp = []
            for j, xxx in enumerate(x):
                if i != j:
                    temp.append(xx - xxx)
            diff.append(temp)
        return diff

    def _read_pose_file(self,filepath):
        body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

        try:
            content = json.load(open(filepath))["people"][0]
        except IndexError:
            return None

        path_parts = os.path.split(filepath)

        frame_id = path_parts[1][:11]
        vid = os.path.split(path_parts[0])[-1]

        save_to = os.path.join('./features', vid)

        try:
            ft = torch.load(os.path.join(save_to, frame_id + '_ft.pt'),weights_only=False)
            xy = ft[:, :2]
            return xy

        except FileNotFoundError:
            body_pose = content["pose_keypoints_2d"]
            left_hand_pose = content["hand_left_keypoints_2d"]
            right_hand_pose = content["hand_right_keypoints_2d"]

            body_pose.extend(left_hand_pose)
            body_pose.extend(right_hand_pose)

            x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
            y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]
            conf = [v for i, v in enumerate(body_pose) if i % 3 == 2 and i // 3 not in body_pose_exclude]

            x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
            y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)
            conf = torch.FloatTensor(conf)

            x_diff = torch.FloatTensor(self._compute_difference(x)) / 2
            y_diff = torch.FloatTensor(self._compute_difference(y)) / 2

            zero_indices = (x_diff == 0).nonzero()

            orient = y_diff / x_diff
            orient[zero_indices] = 0

            xy = torch.stack([x, y]).transpose_(0, 1)
            ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)
            path_parts = os.path.split(filepath)

            frame_id = path_parts[1][:11]
            vid = os.path.split(path_parts[0])[-1]

            save_to = os.path.join('./features', vid)

            os.makedirs(save_to,exist_ok=True)
            torch.save(ft, os.path.join(save_to, frame_id + '_ft.pt'))

            xy = ft[:, :2]

            return xy
        

class WLASL_Dataset(Dataset):
    def __init__(self,index_file_path, split, pose_root, sample_strategy='rnd_start', num_samples=25, num_copies=4, labeling_strategy="fasttext", img_transforms=None, video_transforms=None, test_index_file=None,skeleton_augmentation=True):
        
        assert os.path.exists(index_file_path), "Non-existent indexing file path: {}.".format(index_file_path)
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        self.data = []
        self.glosses = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        if type(split) == 'str':
            split = [split]

        self.test_index_file = test_index_file
        self.labeling_strategy = labeling_strategy
        self._make_dataset(index_file_path, split)

        self.index_file_path = index_file_path
        self.pose_root = pose_root
        self.framename = 'image_{}_keypoints.json'
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples

        self.img_transforms = img_transforms
        self.video_transforms = video_transforms
        self.skeleton_augmentation = skeleton_augmentation

        self.num_copies = num_copies

        self.selected_keypoint_index = [i for i in range(0,11)] + [ i for i in range(91,135)]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end = self.data[index]
        # frames of dimensions (T, H, W, C)
        x = self._load_poses(video_id, frame_start, frame_end, self.sample_strategy, self.num_samples)

        x = torch.reshape(x,(-1,self.num_samples if self.sample_strategy!='k_copies' else self.num_samples*self.num_copies ,2)).permute(2,1,0).unsqueeze(-1)
        # x = torch.from_numpy(z_score_normalize(x.numpy()))
        
        if self.video_transforms:
            x = self.video_transforms(x)
        if self.skeleton_augmentation:
            x = torch.from_numpy(augment_skeleton(x.numpy()))
        
        y = gloss_cat

        return x, y
        
    def _make_dataset(self, index_file_path, split):
        with open(index_file_path, 'r') as f:
            content = json.load(f)

        # create label encoder
        glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])

        self.label_encoder.fit(glosses)
        
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))
            
        if self.test_index_file is not None:
            print('Trained on {}, tested on {}'.format(index_file_path, self.test_index_file))
            with open(self.test_index_file, 'r') as f:
                content = json.load(f)

        # make dataset
        for gloss_entry in content:
            gloss, instances = gloss_entry['gloss'], gloss_entry['instances']

            gloss_cat = labels2cat(self.label_encoder, [gloss])[0]
                
            for instance in instances:
                if instance['split'] not in split:
                    continue

                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.glosses.append(gloss)
                self.data.append(instance_entry)

    def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples):
        """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
         """
        poses = []

        if sample_strategy == 'rnd_start':
            frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'seq':
            frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
                                                                         self.num_copies)
        else:
            raise RuntimeError('Unimplemented sample strategy found: {}.'.format(sample_strategy))

        frames_to_sample = np.array(frames_to_sample) - frame_start

        #skel, score = self._load_json_file(os.path.join(self.pose_root,f"{video_id}.json"))
        skel = self._load_json_file(os.path.join(self.pose_root,f"{video_id}.json"))
        
        poses = skel[frames_to_sample]
        #scores = score[frames_to_sample]

        pad_skel, pad_score = None, None
        # if len(frames_to_sample) < num_samples:
        if len(poses) < num_samples:
            num_padding = num_samples - len(frames_to_sample)
            last_pose = poses[-1][np.newaxis]
            #last_score = scores[-1][np.newaxis]
            pad_skel = last_pose.repeat(num_padding,axis=0)
            #pad_score = last_score.repeat(num_padding,axis=0)

        poses_across_time = torch.from_numpy(poses)
        #scores_across_time = torch.from_numpy(scores)
        if pad_skel is not None:
            pad_skel = torch.from_numpy(pad_skel)
            #pad_score = torch.from_numpy(pad_score)

            poses_across_time = torch.cat([poses_across_time, pad_skel], dim=0)
            #scores_across_time = torch.cat([scores_across_time, pad_score], dim=0)

        return poses_across_time.to(torch.float32) #, scores_across_time.to(torch.float32)
    
    def _load_json_file(self,json_path):
        selected_kps_for_body = list(range(11)) + [17,18]
        
        with open(json_path, 'r') as f:
            content = json.load(f)
        data = np.array(content["keypoints"])
        img_size = np.array(content["img_size"])
        try:
            body = data[:,:17]
            foot = data[:,17:23]
            face = data[:,23:91]
            rhand = data[:,91:112]
            lhand = data[:,112:]
            
            body = np.concatenate([body,np.mean(body[:,5:7],axis=1,keepdims=True),np.mean(body[:,11:13],axis=1,keepdims=True)],axis=1)
            body = body[:,selected_kps_for_body]
            
            # data = np.concatenate([body,face,lhand,rhand],axis=1) # 13 + 68 +21 + 21 = 123
            data = np.concatenate([body,lhand,rhand],axis=1) # 13 +21 + 21 = 13 + 42 = 55
            # for i in range(len(data)):
            #     data[i] = data[i] - data[i,11]
            
        except:
            data = np.zeros((self.num_samples, 55 , 3))

        return 2*(data[:,:,:2]/ 256.0 - 0.5)

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(onehot_encoder, label_encoder, labels):
    return onehot_encoder.transform(label_encoder.transform(labels).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()