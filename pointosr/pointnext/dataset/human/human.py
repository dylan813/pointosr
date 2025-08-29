import os
import sys
import numpy as np
import logging
import torch
import json
from torch.utils.data import Dataset
from ..build import DATASETS
from pointnext.model.layers.subsample import fps
from pointnext.features.geometry_features import GeometryFeatureExtractor
from pointnext.features.fpfh_features import FPFHFeatureExtractor


def pointnext_collate_fn(batch):
    batch_size = len(batch)
    
    num_points = batch[0]['pos'].shape[0]
    num_features = batch[0]['x'].shape[1]
    
    pos_batch = torch.zeros(batch_size, num_points, 3)
    x_batch = torch.zeros(batch_size, num_points, num_features)
    y_batch = torch.zeros(batch_size, dtype=torch.long)
    
    for i, data in enumerate(batch):
        pos_batch[i] = data['pos']
        x_batch[i] = data['x']
        y_batch[i] = data['y']
    
    return {
        'pos': pos_batch,
        'x': x_batch,
        'y': y_batch
    }


@DATASETS.register_module()
class HumanDataset(Dataset):
    classes = [
        "human",
        "fp"
    ]
    num_classes = len(classes)

    dir_to_class_idx = {
        "human_clusters": 0,
        "fp_clusters": 1,
    }
    idx_to_class = {v: k for k, v in dir_to_class_idx.items()}

    def __init__(self, data_dir, split,
                 num_points=2048,
                 transform=None,
                 uniform_sample=True,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.partition = split
        self.num_points = num_points
        self.transform = transform
        self.uniform_sample = uniform_sample
        
        self.collate_fn = pointnext_collate_fn

        class Config:
            def __init__(self):
                class Geometry:
                    def __init__(self):
                        self.normal_radius = 0.15
                        self.eigen_radii = [0.15]
                        self.fpfh_radii = [0.2]
                        self.min_neighbors = 25
                        self.fpfh_pca_k = 6
                        self.density_adaptive = True
                        self.base_density = 2000
                        self.min_radius = 0.05
                        self.max_radius = 0.5
                
                self.geometry = Geometry()
        
        config = Config()
        self.geom_extractor = GeometryFeatureExtractor(config)
        self.fpfh_extractor = FPFHFeatureExtractor(config)

        logging.info(f"Directory to class index mapping: {self.dir_to_class_idx}")
        split_filename = os.path.join(data_dir, "splits", f"id_{split}_split.txt")
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

    def _extract_per_point_features(self, points):
        num_points = points.shape[0]
        
        xyz = points[:, :3]
        
        distances = np.linalg.norm(xyz, axis=1, keepdims=True)
        
        height = xyz[:, 2:3]
        
        from sklearn.neighbors import NearestNeighbors
        k_neighbors = min(10, num_points-1)
        if k_neighbors > 0:
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(xyz)
            distances_to_neighbors, _ = nbrs.kneighbors(xyz)
            local_density = 1.0 / (np.mean(distances_to_neighbors, axis=1, keepdims=True) + 1e-6)
        else:
            local_density = np.ones((num_points, 1))
        
        if k_neighbors > 1:
            local_curvature = np.std(distances_to_neighbors, axis=1, keepdims=True)
        else:
            local_curvature = np.zeros((num_points, 1))
        
        xy_distances = np.linalg.norm(xyz[:, :2], axis=1, keepdims=True)
        angle_xy = np.arctan2(height, xy_distances + 1e-6)
        
        xz_distances = np.linalg.norm(xyz[:, [0, 2]], axis=1, keepdims=True)
        angle_xz = np.arctan2(xyz[:, 1:2], xz_distances + 1e-6)
        
        xyz_normalized = (xyz - np.mean(xyz, axis=0)) / (np.std(xyz, axis=0) + 1e-6)
        
        r = np.linalg.norm(xyz, axis=1, keepdims=True)
        theta = np.arccos(np.clip(xyz[:, 2:3] / (r + 1e-6), -1, 1))
        phi = np.arctan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6)
        
        if k_neighbors > 1:
            mean_neighbor_dist = np.mean(distances_to_neighbors, axis=1, keepdims=True)
            std_neighbor_dist = np.std(distances_to_neighbors, axis=1, keepdims=True)
            min_neighbor_dist = np.min(distances_to_neighbors, axis=1, keepdims=True)
            max_neighbor_dist = np.max(distances_to_neighbors, axis=1, keepdims=True)
        else:
            mean_neighbor_dist = np.zeros((num_points, 1))
            std_neighbor_dist = np.zeros((num_points, 1))
            min_neighbor_dist = np.zeros((num_points, 1))
            max_neighbor_dist = np.zeros((num_points, 1))
        
        centroid = np.mean(xyz, axis=0)
        dist_from_centroid = np.linalg.norm(xyz - centroid, axis=1, keepdims=True)
        
        height_rel = height - centroid[2]
        
        radial_xy = np.linalg.norm(xyz[:, :2] - centroid[:2], axis=1, keepdims=True)
        
        per_point_features = np.concatenate([
            local_density,
            local_curvature,
            angle_xy,
            angle_xz,
            xyz_normalized,
            theta, phi,
            mean_neighbor_dist,
            std_neighbor_dist,
            min_neighbor_dist,
            max_neighbor_dist,
            dist_from_centroid,
            height_rel,
            radial_xy,
        ], axis=1)
        
        return per_point_features

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
        per_point_features = self._extract_per_point_features(sampled_points)
        
        data = {
            'pos': torch.from_numpy(pos).float(),
            'y': torch.tensor(label).long(),
            'x': torch.from_numpy(per_point_features).float()
        }

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
