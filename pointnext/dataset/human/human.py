import os
import sys
import numpy as np
import logging
import torch
import json
from torch.utils.data import Dataset
from ..build import DATASETS
from pointnext.model.layers.subsample import fps


@DATASETS.register_module()
class HumanDataset(Dataset):
    classes = ["human", "false"]
    num_classes = len(classes)

    dir_to_class_idx = {
        "human_clusters": 0,
        "false_clusters": 1,
    }
    idx_to_class = {v: k for k, v in dir_to_class_idx.items()}

    def __init__(self, data_dir, split,
                 num_points=2048,
                 transform=None,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        self.classes = HumanDataset.classes

        logging.info(f"Directory to class index mapping: {self.dir_to_class_idx}")
        split_filename = os.path.join(data_dir, f"{split}_split.txt")
        if not os.path.isfile(split_filename):
            raise FileNotFoundError(f"Split file not found: {split_filename}")

        self.file_list = []
        self.label_list = []
        class_dirs = list(self.dir_to_class_idx.keys())

        logging.info(f'Successfully loaded {split} split. Number of samples: {len(self.file_list)}')
        logging.info(f'Dataset: {self.__class__.__name__}, Classes: {self.classes}, Num classes: {self.num_classes}')

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.label_list[idx]

        filepath = os.path.join(self.data_dir, filename)
        raw_points = np.fromfile(filepath, dtype=np.float32)
        points = raw_points.reshape(-1, 4)

        current_num_points = points.shape[0]

        sampled_points = None
        if current_num_points > self.num_points:
            indices = np.random.choice(current_num_points, self.num_points, replace=False)
            sampled_points = points[indices, :]
        elif current_num_points < self.num_points:
            indices = np.random.choice(current_num_points, self.num_points, replace=True)
            sampled_points = points[indices, :]
        else:
            sampled_points = points

        pos = sampled_points[:, :3]
        intensity = sampled_points[:, 3:]

        data = {'pos': torch.from_numpy(pos).float(),
                'y': torch.tensor(label).long(),
               }
        
        data['x'] = torch.cat((data['pos'], torch.from_numpy(intensity).float()), dim=1)

        if self.transform is not None:
            try:
                data = self.transform(data)
                if 'pos' not in data or 'x' not in data or 'y' not in data:
                     logging.warning(f"Transform may have removed essential keys for file {filename}. Data: {data.keys()}")
            except Exception as e:
                logging.error(f"Error during transform for file {filename}: {e}")
                return None

        return data

    def __len__(self):
        return len(self.file_list)
