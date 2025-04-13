import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle

# Add the parent directory to the path to import from point_osr
sys.path.append(str(Path(__file__).parent.parent))

from dataset.build import build_dataloader_from_cfg
from models.pointnext_wrapper import PointNeXtFeatureExtractor

def extract_and_save_features(model, data_loader, save_dir, subset_name="train"):
    """Extract features for all samples in the data loader and save them.
    
    Args:
        model: Feature extraction model
        data_loader: DataLoader containing the dataset to extract features from
        save_dir: Directory to save features to
        subset_name: Name of the subset (e.g., "train", "test")
    """
    features_list = []
    labels_list = []
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract features batch by batch
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting {subset_name} features"):
            # Move batch to the same device as model
            if isinstance(batch, dict):
                data, labels = batch, batch['y'].to(model.device)
            else:
                data, labels = batch
                if isinstance(labels, dict):
                    data, labels = batch[0], batch[1]['y']
                labels = labels.to(model.device)
            
            # Extract features
            features = model.extract_features(data)
            
            # Store features and labels
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    # Concatenate features and labels
    all_features = np.vstack(features_list)
    all_labels = np.concatenate(labels_list)
    
    # Save features and labels
    save_path = os.path.join(save_dir, f"{subset_name}_features.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump({'features': all_features, 'labels': all_labels}, f)
    
    print(f"Saved {len(all_features)} {subset_name} features to {save_path}")
    return all_features, all_labels

def parse_args():
    parser = argparse.ArgumentParser('Feature extraction for open-set recognition')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--pretrained', type=str, required=True, help='pretrained model path')
    parser.add_argument('--save_dir', type=str, default='data/features', help='directory to save features')
    parser.add_argument('--subset', type=str, default='train', help='subset to extract features from (train, test, or all)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Import config
    sys.path.append(os.path.join(os.path.dirname(args.cfg), '..'))
    config = __import__(os.path.basename(args.cfg).split('.')[0], fromlist=[''])
    
    # Build model
    model = PointNeXtFeatureExtractor(model_cfg=config.model, pretrained_path=args.pretrained)
    model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(model.device)
    
    # Process subsets
    if args.subset in ['train', 'all']:
        train_loader = build_dataloader_from_cfg(config.data.train, config.data.train_batch_size, 
                                              config.data.num_workers, config.data.transform)
        extract_and_save_features(model, train_loader, args.save_dir, "train")
    
    if args.subset in ['test', 'all']:
        test_loader = build_dataloader_from_cfg(config.data.test, config.data.test_batch_size, 
                                             config.data.num_workers, config.data.transform)
        extract_and_save_features(model, test_loader, args.save_dir, "test")

if __name__ == "__main__":
    main() 