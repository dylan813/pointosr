import torch
import numpy as np
from pointnext.model.layers.subsample import fps
from sklearn.neighbors import NearestNeighbors


class OnlineDataloader:
    # add classes used in training
    classes = [
        "human",
        "fp"
    ]

    def __init__(self, num_points=2048, device='cuda'):
        self.num_points = num_points
        self.device = torch.device(device)
        
        self._sensor_max = 65_535
        self._log_max_val = np.log1p(self._sensor_max)

    def _norm_intensity(self, ints: np.ndarray) -> np.ndarray:
        return np.log1p(ints) / self._log_max_val

    def _extract_per_point_features(self, points):
        num_points = points.shape[0]
        
        xyz = points[:, :3]
        
        distances = np.linalg.norm(xyz, axis=1, keepdims=True)
        
        height = xyz[:, 2:3]
        
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

    def process(self, points, identifier=None):
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("Input point cloud must be a NumPy array with shape (N, 4)")

        # Point sampling (using FPS for online processing to maintain quality)
        if points.shape[0] > self.num_points:
            points_tensor = torch.from_numpy(points).to(torch.float32).to(self.device)
            sampled_points_tensor = fps(points_tensor.unsqueeze(0), self.num_points).squeeze(0)
            sampled_points = sampled_points_tensor.cpu().numpy()
        elif points.shape[0] < self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            sampled_points = points[indices, :]
        else:
            sampled_points = points

        # Extract position and per-point features
        pos = sampled_points[:, :3]
        per_point_features = self._extract_per_point_features(sampled_points)
        
        # Convert to tensors
        pos_tensor = torch.from_numpy(pos).float()
        features_tensor = torch.from_numpy(per_point_features).float()

        # PointCloudCenterAndNormalize (datatransform)
        pos_tensor = pos_tensor - torch.mean(pos_tensor, axis=0, keepdims=True)
        m = torch.max(torch.sqrt(torch.sum(pos_tensor ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
        pos_tensor = pos_tensor / m

        data = {
            'pos': pos_tensor,
            'x': features_tensor
        }
        
        if identifier is not None:
            data['id'] = identifier
            
        return data