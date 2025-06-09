import torch
import numpy as np
from pointnext.model.layers.subsample import fps


class PointCloudProcessor:
    def __init__(self, num_points=2048, device='cuda'):
        self.num_points = num_points
        self.device = torch.device(device)

    def process(self, points, identifier=None):
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("Input point cloud must be a NumPy array with shape (N, 4)")

        # Farthest Point Sampling (FPS)
        if points.shape[0] > self.num_points:
            points_tensor = torch.from_numpy(points).to(torch.float32).to(self.device)
            sampled_points_tensor = fps(points_tensor.unsqueeze(0), self.num_points).squeeze(0)
            sampled_points = sampled_points_tensor.cpu().numpy()
        elif points.shape[0] < self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            sampled_points = points[indices, :]
        else:
            sampled_points = points

        # PointsToTensor (datatransform)
        pos = torch.from_numpy(sampled_points[:, :3]).float()
        intensity = torch.from_numpy(sampled_points[:, 3:]).float()

        # PointCloudCenterAndNormalize (datatransform)
        pos = pos - torch.mean(pos, axis=0, keepdims=True)
        m = torch.max(torch.sqrt(torch.sum(pos ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
        pos = pos / m

        data = {
            'pos': pos,
            'x': torch.cat((pos, intensity), dim=1)
        }
        
        if identifier is not None:
            data['id'] = identifier
            
        return data