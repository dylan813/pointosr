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
    # Classes remain the same
    classes = ["human", "false"]
    num_classes = len(classes)

    # Map directory names to class indices
    dir_to_class_idx = {
        "human_clusters": 0,
        "falser_clusters": 1,
    }
    # Reverse map for logging/debugging if needed
    idx_to_class = {v: k for k, v in dir_to_class_idx.items()}

    def __init__(self, data_dir, split,
                 num_points=2048,
                 transform=None,
                 # split_dir parameter is removed as it's the same as data_dir
                 # split_dir=None,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        self.classes = HumanDataset.classes # Use the defined classes

        # --- Check for required arguments ---
        # Removed split_dir check
        # if split_dir is None:
        #     raise ValueError("split_dir must be provided")

        # 1. Define class mapping based on directory names
        # (Moved mapping to class level for clarity)
        logging.info(f"Directory to class index mapping: {self.dir_to_class_idx}")

        # 2. Read the split file (.txt) using data_dir
        # Use data_dir to find the split file
        split_filename = os.path.join(data_dir, f"{split}_split.txt")
        if not os.path.isfile(split_filename):
            raise FileNotFoundError(f"Split file not found: {split_filename}")

        self.file_list = []
        self.label_list = []
        with open(split_filename, 'r') as f:
            for line in f:
                # Each line contains a filename relative to data_dir
                # e.g., "human_clusters/sample1.bin" or "falser_clusters/sampleA.bin"
                relative_filepath = line.strip()
                if not relative_filepath: continue # Skip empty lines

                try:
                    # Extract the first directory component as the class indicator
                    # os.path.normpath handles potential different separators (/, \\)
                    # os.path.split splits into (head, tail), we want the first part of head
                    parts = os.path.normpath(relative_filepath).split(os.sep)
                    if len(parts) < 2: # Ensure there is at least one directory level
                        logging.warning(f"File path '{relative_filepath}' in split file '{split_filename}' does not seem to be in a class subdirectory (e.g., 'human_clusters/file.bin'). Skipping.")
                        continue

                    class_dir_name = parts[0] # e.g., "human_clusters" or "falser_clusters"

                    # Get the integer label index from the directory name
                    label_idx = self.dir_to_class_idx.get(class_dir_name)

                    if label_idx is not None:
                        self.file_list.append(relative_filepath) # Store the full relative path
                        self.label_list.append(label_idx)
                    else:
                        # Handle cases where the directory name is not in our mapping
                        logging.warning(f"Directory name '{class_dir_name}' extracted from path '{relative_filepath}' is not a recognized class directory {list(self.dir_to_class_idx.keys())}. Skipping.")

                except Exception as e:
                    logging.error(f"Error processing line '{relative_filepath}' from split file '{split_filename}': {e}")


        if len(self.file_list) == 0:
             raise RuntimeError(f"No valid files found for split {split} in {split_filename} with directory structure {list(self.dir_to_class_idx.keys())}")

        logging.info(f'Successfully loaded {split} split. Number of samples: {len(self.file_list)}')
        logging.info(f'Dataset: {self.__class__.__name__}, Classes: {self.classes}, Num classes: {self.num_classes}')


    @property # Keep if needed by external code
    def num_classes(self):
        # Return the dynamically calculated number of classes
        return len(self.classes)

    def __getitem__(self, idx):
        # 1. Get filename and label (label is already an integer index)
        # filename now includes the relative path, e.g., "human_clusters/sample1.bin"
        filename = self.file_list[idx]
        label = self.label_list[idx] # Correct label index retrieved

        # 2. Construct full path and load .bin file
        # The filename already contains the relative path from data_dir
        filepath = os.path.join(self.data_dir, filename)
        try:
            raw_points = np.fromfile(filepath, dtype=np.float32)
            points = raw_points.reshape(-1, 4)
        except FileNotFoundError:
             logging.error(f"File not found during loading: {filepath}")
             # Return None? Needs handling in dataloader's collate_fn
             return None
        except ValueError as e:
             logging.error(f"Error reshaping file {filepath}. Incorrect format or dtype? Size: {raw_points.size}. Error: {e}")
             return None
        except Exception as e:
            logging.error(f"Error loading file {filepath}: {e}")
            return None

        # 3. Handle point sampling/padding
        current_num_points = points.shape[0]
        if current_num_points == 0:
             logging.warning(f"File {filepath} contains 0 points.")
             # Handle empty point clouds - maybe return None or skip?
             # Returning None for now. Requires collate_fn handling.
             return None

        sampled_points = None # Initialize variable
        if current_num_points > self.num_points:
            # Sample points if too many
            if self.partition == 'train':
                # Training: Use random sampling
                indices = np.random.choice(current_num_points, self.num_points, replace=False)
                sampled_points = points[indices, :] # Keep as numpy array
            else:
                # Val/Test: Use Farthest Point Sampling (FPS)
                # Convert points to tensor for FPS
                points_tensor = torch.from_numpy(points).float().cuda() # Move to GPU if available
                # Add batch dimension, call FPS, remove batch dimension
                # The fps function expects input shape (B, N, C) and k (num_points)
                # It returns indices of shape (B, k)
                # Ensure fps function is correctly imported: from pointnext.model.layers.subsample import fps
                fps_indices = fps(points_tensor.unsqueeze(0), self.num_points) # Shape: (1, num_points)
                # Use the indices returned by FPS to select points from the original numpy array
                sampled_points = points[fps_indices.squeeze(0).cpu().numpy(), :] # Select points and move indices to CPU/numpy
        elif current_num_points < self.num_points:
            # Pad points if too few by repeating points
            indices = np.random.choice(current_num_points, self.num_points, replace=True)
            sampled_points = points[indices, :]
        else:
            sampled_points = points # Already a numpy array

        # Ensure sampled_points is assigned
        if sampled_points is None:
             logging.error(f"Point sampling failed for {filepath}")
             return None

        # 4. Separate coordinates (pos) and intensity feature from the sampled numpy array
        pos = sampled_points[:, :3] # x, y, z
        intensity = sampled_points[:, 3:] # Intensity feature

        # 5. Package data - Convert FINAL sampled data to tensors here
        data = {'pos': torch.from_numpy(pos).float(),
                'y': torch.tensor(label).long(), # Ensure label is a tensor
               }
        
        # Use pos + intensity as the input features 'x'
        data['x'] = torch.cat((data['pos'], torch.from_numpy(intensity).float()), dim=1) # [N, 4] tensor

        # 6. Apply transformations (if any)
        if self.transform is not None:
            try:
                data = self.transform(data)
                # Verify essential keys are still present after transform
                if 'pos' not in data or 'x' not in data or 'y' not in data:
                     logging.warning(f"Transform may have removed essential keys for file {filename}. Data: {data.keys()}")
                     # Decide how to handle this - skip? error? reconstruct?
            except Exception as e:
                logging.error(f"Error during transform for file {filename}: {e}")
                return None # Skip problematic sample

        # Final check before returning
        if not isinstance(data['pos'], torch.Tensor) or \
           not isinstance(data['x'], torch.Tensor) or \
           not isinstance(data['y'], torch.Tensor):
            logging.error(f"Data values are not tensors for file {filename}. Check loading & transforms.")
            return None

        return data

    def __len__(self):
        return len(self.file_list)
