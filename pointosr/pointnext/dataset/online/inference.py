import torch
import numpy as np

# It's assumed that the pointnext package is in the Python path.
# This import path is based on the one in human.py
from pointnext.model.layers.subsample import fps


class PointCloudProcessor:
    """
    Processes a single point cloud for inference.
    
    This class takes a raw point cloud (as a NumPy array) and prepares it
    to be fed into a PointNet-like model. It handles sampling to a fixed
    number of points and formatting the data into tensors.
    """
    def __init__(self, num_points=2048, device='cuda'):
        """
        Initializes the processor.
        
        Args:
            num_points (int): The target number of points to sample.
            device (str): The device to use for processing ('cuda' or 'cpu').
        """
        self.num_points = num_points
        self.device = torch.device(device)

    def process(self, points, identifier=None):
        """
        Processes a single point cloud.
        
        Args:
            points (np.ndarray): The input point cloud, shape (N, 4),
                                 where columns are [x, y, z, intensity].
            identifier (any, optional): An identifier for the point cloud,
                                        e.g., a timestamp or filename.
                                        Defaults to None.
        
        Returns:
            dict: A dictionary containing the processed data, ready for the model.
                  Keys are 'pos', 'x', and optionally 'id'.
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("Input `points` must be a NumPy array with shape (N, 4).")

        # Use Farthest Point Sampling (FPS) for deterministic and representative
        # sampling, which is good practice for inference.
        if points.shape[0] > self.num_points:
            points_tensor = torch.from_numpy(points).to(torch.float32).to(self.device)
            # fps expects a batch dimension, so we add one and then remove it.
            sampled_points_tensor = fps(points_tensor.unsqueeze(0), self.num_points).squeeze(0)
            sampled_points = sampled_points_tensor.cpu().numpy()
        # If the cloud has fewer points, we pad by duplicating existing points.
        elif points.shape[0] < self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            sampled_points = points[indices, :]
        else:
            sampled_points = points

        # Separate position and features, and convert to tensors.
        pos = torch.from_numpy(sampled_points[:, :3]).float()
        intensity = torch.from_numpy(sampled_points[:, 3:]).float()

        data = {
            'pos': pos,
            'x': torch.cat((pos, intensity), dim=1)
        }
        
        if identifier is not None:
            data['id'] = identifier
            
        return data 