import os
import sys
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from pointnext.model.layers.subsample import fps


@DATASETS.register_module()
class UnlabeledTestDataset(Dataset):
    classes = ["human", "misc"]  # Default classes
    num_classes = len(classes)

    def __init__(self, data_root, 
                 transform=None, 
                 num_points=2048,
                 uniform_sample=True,
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.num_points = num_points
        self.uniform_sample = uniform_sample
        self.classes = UnlabeledTestDataset.classes
        
        self.file_list = []
        for file in os.listdir(data_root):
            if file.endswith('.bin'):
                self.file_list.append(file)

        if len(self.file_list) == 0:
            raise RuntimeError(f"Found 0 files in {data_root}")
        
        logging.info(f"Found {len(self.file_list)} test files in {data_root}")
        
        if uniform_sample:
            self.fps_cache = {}
            logging.info(f"Precomputing FPS points for test data...")
            for i, filename in enumerate(self.file_list):
                full_path = os.path.join(self.data_root, filename)
                raw_points = np.fromfile(full_path, dtype=np.float32)
                points = raw_points.reshape(-1, 4)
                
                if points.shape[0] > self.num_points:
                    points_tensor = torch.from_numpy(points).to(torch.float32).cuda()
                    sampled_points = fps(points_tensor.unsqueeze(0), self.num_points).squeeze(0).cpu().numpy()
                    self.fps_cache[filename] = sampled_points
            
            logging.info(f"Finished precomputing FPS for {len(self.fps_cache)} samples")

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.data_root, filename)
        
        if hasattr(self, 'fps_cache') and filename in self.fps_cache:
            sampled_points = self.fps_cache[filename]
        else:
            raw_points = np.fromfile(filepath, dtype=np.float32)
            points = raw_points.reshape(-1, 4)
            
            if points.shape[0] > self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                sampled_points = points[indices, :]
            elif points.shape[0] < self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=True)
                sampled_points = points[indices, :]
            else:
                sampled_points = points

        pos = sampled_points[:, :3]
        intensity = sampled_points[:, 3:]

        data = {
            'pos': torch.from_numpy(pos).float(),
            'y': torch.tensor(0).long(),  # dummy label for unlabeled data
            'filename': filename
        }
        
        data['x'] = torch.cat((data['pos'], torch.from_numpy(intensity).float()), dim=1)

        if self.transform is not None:
            try:
                data = self.transform(data)
            except Exception as e:
                logging.error(f"Error during transform for file {filename}: {e}")
                return None

        return data
    
    def __len__(self):
        return len(self.file_list) 