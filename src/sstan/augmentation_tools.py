import numpy as np

def random_rotate(skeleton, max_angle=15):
    """Random Rotation."""
    angle = np.random.uniform(-max_angle, max_angle)  
    radian = np.deg2rad(angle)
    
    rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)],
                                [np.sin(radian),  np.cos(radian)]])
    
    rotated_skeleton = skeleton.copy()
    for t in range(skeleton.shape[1]):
        for v in range(skeleton.shape[2]):
            for m in range(skeleton.shape[3]):
                rotated_skeleton[:, t, v, m] = np.dot(rotation_matrix, skeleton[:, t, v, m])
    return rotated_skeleton

def random_scale(skeleton, scale_range=(0.9, 1.1)):
    """Random Scaling."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return skeleton * scale_factor

def random_translate(skeleton, max_translation=0.1):
    """Random shifting."""
    translation = np.random.uniform(-max_translation, max_translation, size=(2,))
    translated_skeleton = skeleton.copy()
    translated_skeleton[0, :, :, :] += translation[0]  # x-dim
    translated_skeleton[1, :, :, :] += translation[1]  # y-dim
    return translated_skeleton

def random_vertex_dropout(data, drop_rate):
    """
    Perform random vertex dropout for data augmentation.

    Parameters:
        data (numpy.ndarray): Input data of shape (C, T, V, M)
        drop_rate (float or int): Proportion or number of vertices to drop

    Returns:
        numpy.ndarray: Augmented data with vertices randomly set to zero
    """
    # Check input shape
    if not isinstance(data, np.ndarray) or data.ndim != 4:
        raise ValueError("Input data must be a numpy array of shape (C, T, V, M)")

    C, T, V, M = data.shape

    # Determine the number of vertices to drop
    if isinstance(drop_rate, float):
        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError("drop_rate as float must be in the range [0, 1]")
        num_vertices_to_drop = int(V * drop_rate)
    elif isinstance(drop_rate, int):
        if not (0 <= drop_rate <= V):
            raise ValueError("drop_rate as int must be in the range [0, V]")
        num_vertices_to_drop = drop_rate
    else:
        raise TypeError("drop_rate must be a float or an int")

    # Randomly select vertices to drop
    vertices_to_drop = np.random.choice(V, num_vertices_to_drop, replace=False)

    # Set the data for selected vertices to zero
    augmented_data = data.copy()
    augmented_data[:, :, vertices_to_drop, :] = 0

    return augmented_data

def augment_skeleton(skeleton, rotate=True, scale=True, translate=True,random_drop=True):
    """Apply the augmentation for skeleton data."""
    augmented_skeleton = skeleton.copy()
    
    if rotate:
        augmented_skeleton = random_rotate(augmented_skeleton)
    if scale:
        augmented_skeleton = random_scale(augmented_skeleton)
    if translate:
        augmented_skeleton = random_translate(augmented_skeleton)
    
    return augmented_skeleton 




"""
Joint Mixing Data Augmentation for Skeleton-Based Action recognition (ACM transaction on Multimedia Computing, Communications, and Applications)
https://dl.acm.org/doi/10.1145/3700878
"""

import random
import numpy as np
import torch

def _random_start(src_length, resized_length):
    start_idx = random.randint(0,src_length-resized_length)
    return np.arange(start_idx,start_idx+resized_length)

def _uniform_interval(src_length, resized_length):
    return np.linspace(0, src_length-1, resized_length, endpoint=True).astype(np.int16)

def Mix_Temporal(data, onehot_label, threshold=0.5, max_mixing_ratio=0.5):
    N,C,T,V,M = data.size()
    clone_index = torch.randperm(N)
    clone_data = data[clone_index].clone()
    clone_onehot = onehot_label[clone_index].clone()

    for i in range(N):
        strategy = random.choice([
            "mixcut",
            # "temporal_mix",
            "resize_mix",
            "no_mix",
            ])

        if strategy == "mixcut":
            replace_start_index = random.randint(0,T)
            replace_range = random.randint(1,int(T*max_mixing_ratio))
        
            if replace_start_index+replace_range >= T:
                replace_range = T-replace_start_index

            replace_ratio = replace_range/T

            if random.random() < threshold:
                data[i,:,replace_start_index:replace_start_index+replace_range] = clone_data[i,:,replace_start_index:replace_start_index+replace_range]
                onehot_label[i] = (1-replace_ratio)*onehot_label[i] + replace_ratio*clone_onehot[i]
        
        elif strategy == "temporal_mix":
            border_index = random.randint(0,T)
            ori_temporal_size, src_temporal_size = border_index, T - border_index

            ori_frame_select_func = random.choice([_random_start, _uniform_interval])
            ori_selected_frame_idx = ori_frame_select_func(T,ori_temporal_size)
            src_frame_select_func = random.choice([_random_start, _uniform_interval])
            src_selected_frame_idx = src_frame_select_func(T,src_temporal_size)

            if random.random() < threshold:
                data[i] = torch.cat([data[i,:,ori_selected_frame_idx],clone_data[i,:,src_selected_frame_idx]],dim=1)
                onehot_label[i] = (border_index/T)*onehot_label[i] + (1-border_index/T)*clone_onehot[i]

        elif strategy == "resize_mix":
            replace_start_index = random.randint(0,T)
            replace_range = random.randint(1,int(T*max_mixing_ratio))
        
            if replace_start_index+replace_range >= T:
                replace_range = T-replace_start_index

            replace_ratio = replace_range/T

            src_start_idx = random.randint(0,T-replace_range)

            if random.random() < threshold:
                data[i,:,replace_start_index:replace_start_index+replace_range] = clone_data[i,:,src_start_idx:src_start_idx+replace_range]
                onehot_label[i] = (1-replace_ratio)*onehot_label[i] + replace_ratio*clone_onehot[i]

    
    return data, onehot_label


def Mix_Vertex(data, onehot_label, threshold=0.5, max_mixing_ratio=0.5):
    N,C,T,V,M = data.size()
    clone_index = torch.randperm(N)
    clone_data = data[clone_index].clone()
    clone_onehot = onehot_label[clone_index].clone()

    for i in range(N):
        strategy = random.choice([
            "random",
            "part",
            "no_mix",
            ])

        if strategy=="random":
            num_terget_vertex = random.randint(1,int(V*max_mixing_ratio))
            terget_vertex = random.sample(list(range(V)),num_terget_vertex)
            replace_ratio = num_terget_vertex/V

            if random.random() < threshold:
                data[i,:,:,terget_vertex] = clone_data[i,:,:,terget_vertex]
                onehot_label[i] = (1-replace_ratio)*onehot_label[i] + replace_ratio*clone_onehot[i]

        elif strategy=="part":
            replace_start_index = random.randint(0,V)
            replace_range = random.randint(0,int(V*max_mixing_ratio))
        
            if replace_start_index+replace_range >= V:
                replace_range = V-replace_start_index

            replace_ratio = replace_range/V

            if random.random() < threshold:
                data[i,:,:,replace_start_index:replace_start_index+replace_range] = clone_data[i,:,:,replace_start_index:replace_start_index+replace_range]
                onehot_label[i] = (1-replace_ratio)*onehot_label[i] + replace_ratio*clone_onehot[i]
    
    return data, onehot_label

def JointMixAug(data, onehot_label):
    """
    data : (N,C,T,V,M)
    onehot_label : (N. num_classes)
    """
    N,C,T,V,M = data.size()
    augment_func_list = [Mix_Temporal, Mix_Vertex]
    augment_order = torch.randperm(len(augment_func_list))
    for i in augment_order:
        data, onehot_label = augment_func_list[i](data, onehot_label)
        # index = torch.randperm(N)
        # data, onehot_label = data[index], onehot_label[index]

    return data, onehot_label
