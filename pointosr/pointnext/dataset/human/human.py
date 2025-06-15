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
    classes = [
        "human",
        "false"
    ]
    num_classes = len(classes)

    dir_to_class_idx = {
        "human_clusters": 0,
        "false_clusters": 1,
    }
    idx_to_class = {v: k for k, v in dir_to_class_idx.items()}

    def __init__(self, data_dir, split,
                 num_points=2048,
                 transform=None,
                 uniform_sample=True,
                 lower_percentile=1.0,
                 upper_percentile=99.0,
                 intensity_low=None,
                 intensity_high=None,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        self.classes = HumanDataset.classes
        self.uniform_sample = uniform_sample

        logging.info(f"Directory to class index mapping: {self.dir_to_class_idx}")
        split_filename = os.path.join(data_dir, f"{split}_split.txt")
        if not os.path.isfile(split_filename):
            raise FileNotFoundError(f"Split file not found: {split_filename}")

        self.file_list = []
        self.label_list = []
        class_dirs = list(self.dir_to_class_idx.keys())
        
        with open(split_filename, 'r') as f:
            for line in f:
                filename = line.strip()
                if not filename: continue
                
                for class_dir in class_dirs:
                    filepath = os.path.join(data_dir, class_dir, filename)
                    if os.path.isfile(filepath):
                        self.file_list.append(os.path.join(class_dir, filename))
                        self.label_list.append(self.dir_to_class_idx[class_dir])
                        break

        logging.info(f'Successfully loaded {split} split. Number of samples: {len(self.file_list)}')
        logging.info(f'Dataset: {self.__class__.__name__}, Classes: {self.classes}, Num classes: {self.num_classes}')
        
        if split == "train":
            all_ints = []
            for rel_fp in self.file_list:
                pts = np.fromfile(os.path.join(data_dir, rel_fp), dtype=np.float32).reshape(-1, 4)
                all_ints.append(pts[:, 3])
            all_ints = np.concatenate(all_ints)
            self.int_low = np.percentile(all_ints, lower_percentile)
            self.int_high = np.percentile(all_ints, upper_percentile)
            logging.info(f"Intensity percentiles p{lower_percentile}={self.int_low:.2f}, p{upper_percentile}={self.int_high:.2f}")
        else:
            if intensity_low is None or intensity_high is None:
                raise ValueError("For val/test splits, pass intensity_low & intensity_high computed from training.")
            self.int_low = float(intensity_low)
            self.int_high = float(intensity_high)
        self._eps = 1e-6
        
        if (split == 'val' or split == 'test') and uniform_sample:
            self.fps_cache = {}
            logging.info(f"Precomputing FPS points for {split} split...")
            for i, filepath in enumerate(self.file_list):
                full_path = os.path.join(self.data_dir, filepath)
                raw_points = np.fromfile(full_path, dtype=np.float32)
                points = raw_points.reshape(-1, 4)
                
                if points.shape[0] > self.num_points:
                    points_tensor = torch.from_numpy(points).to(torch.float32).cuda()
                    sampled_points = fps(points_tensor.unsqueeze(0), self.num_points).squeeze(0).cpu().numpy()
                    self.fps_cache[filepath] = sampled_points
            
            logging.info(f"Finished precomputing FPS for {len(self.fps_cache)} samples")

    @property
    def num_classes(self):
        return len(self.classes)

    def _sample_points(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        if n > self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
            return pts[idx]
        if n < self.num_points:
            idx = np.random.choice(n, self.num_points, replace=True)
            return pts[idx]
        return pts

    def _norm_intensity(self, ints: np.ndarray) -> np.ndarray:
        clipped = np.clip(ints, self.int_low, self.int_high)
        return (clipped - self.int_low) / (self.int_high - self.int_low + self._eps)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.label_list[idx]

        filepath = os.path.join(self.data_dir, filename)
        
        if hasattr(self, 'fps_cache') and filename in self.fps_cache:
            sampled_points = self.fps_cache[filename]
        else:
            raw_points = np.fromfile(filepath, dtype=np.float32)
            points = raw_points.reshape(-1, 4)
            sampled_points = self._sample_points(points)

        pos = sampled_points[:, :3]
        intensity_raw = sampled_points[:, 3]
        intensity_norm = self._norm_intensity(intensity_raw).reshape(-1, 1)

        data = {'pos': torch.from_numpy(pos).float(),
                'y': torch.tensor(label).long(),
               }
        
        data['x'] = torch.cat((data['pos'], torch.from_numpy(intensity_norm).float()), dim=1)

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
