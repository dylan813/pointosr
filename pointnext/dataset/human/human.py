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
    classes = ["human"]
    num_classes = 1

    def __init__(self, data_dir, split,
                 num_points=2048,
                 transform=None,
                 split_dir=None,
                 label_file=None,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        self.classes = HumanDataset.classes

        # --- Added checks for required arguments ---
        if split_dir is None:
            raise ValueError("split_dir must be provided")
        if label_file is None:
            raise ValueError("label_file must be provided")
        # --- End added checks ---

        # 1. Load label mapping from JSON
        if not os.path.isfile(label_file):
             raise FileNotFoundError(f"Label file not found: {label_file}")
        try:
            with open(label_file, 'r') as f:
                self.label_map = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading label file {label_file}: {e}")

        # 2. Read the split file (.txt) to get the list of files for this split
        split_filename = os.path.join(split_dir, f"{split}_split.txt")
        if not os.path.isfile(split_filename):
            raise FileNotFoundError(f"Split file not found: {split_filename}")
        
        self.file_list = []
        self.label_list = []
        with open(split_filename, 'r') as f:
            for line in f:
                # Assuming each line contains a filename relative to data_dir
                # This filename must be a key in your labels.json
                filename = line.strip()
                if not filename: continue # Skip empty lines

                if filename in self.label_map:
                    self.file_list.append(filename)
                    # Convert any string labels to integers (0 for human class)
                    label_value = self.label_map[filename]
                    if isinstance(label_value, str):
                        label_value = 0  # Convert all strings to class index 0 (human)
                    self.label_list.append(label_value)
                else:
                    logging.warning(f"Filename '{filename}' from split file '{split_filename}' not found in label map '{label_file}'. Skipping.")

        if len(self.file_list) == 0:
             raise RuntimeError(f"No valid files found for split {split} in {split_filename} with labels in {label_file}")
        
        logging.info(f'Successfully loaded {split} split. Number of samples: {len(self.file_list)}')
        logging.info(f'Dataset: {self.__class__.__name__}, Classes: {self.classes}, Num classes: {self.num_classes}')

    # --- Label determination is handled in __init__ now ---
    # def _get_label_from_filename(self, filename): ... (Removed)

    @property # Keep if needed by external code
    def num_classes(self):
        return 1  # Direct return of the integer value

    def __getitem__(self, idx):
        # 1. Get filename and label
        filename = self.file_list[idx]
        label = self.label_list[idx] # Correct label is now retrieved
        
        # Ensure label is an integer
        if isinstance(label, str):
            label = 0  # Convert string labels to 0 (human class)

        # 2. Construct full path and load .bin file
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
                indices = np.random.choice(current_num_points, self.num_points, replace=False)
                sampled_points = points[indices, :] # Keep as numpy array for now
            else: # Val/Test: Use Farthest Point Sampling (FPS) if possible
                # Perform FPS on CPU instead of GPU to avoid CUDA in forked subprocess error
                points_tensor = torch.from_numpy(points).float()
                # Add batch dimension, call FPS, remove batch dimension
                fps_indices = torch.zeros(1, self.num_points, dtype=torch.int32)
                # Use numpy-based sampling instead of GPU FPS
                # Simple random sampling as fallback
                indices = np.random.choice(current_num_points, self.num_points, replace=False)
                sampled_points = points[indices, :]
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
