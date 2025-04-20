import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import yaml
from easydict import EasyDict as edict

# Add the parent directory to the path to import from point_osr
sys.path.append(str(Path(__file__).parent.parent))

from dataset.build import build_dataloader_from_cfg
from model.pointnext_wrapper import PointNeXtFeatureExtractor

def extract_and_save_features(model, data_loader, save_dir, subset_name="train"):
    """Extract features for all samples in the data loader and save them.
    
    Args:
        model: Feature extraction model (PointNeXtFeatureExtractor)
        data_loader: DataLoader containing the dataset to extract features from
        save_dir: Directory to save features to
        subset_name: Name of the subset (e.g., "train", "test", "val")
    """
    features_list = []
    labels_list = []
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract features batch by batch
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting {subset_name} features"):
            # Prepare data for the model
            # Assuming the default concat_collate_fn is used, 
            # the batch is a dictionary with 'pos', 'x', 'y', etc.
            if isinstance(batch, dict):
                # Data for the model should be a dictionary or object expected by model.extract_features
                # Typically PointNeXt expects point positions ('pos') and maybe initial features ('x')
                model_input = { 
                    'pos': batch['pos'].to(model.device),
                    'x': batch['x'].to(model.device) 
                }
                # Labels are usually separate
                labels = batch['y'] # Keep labels on CPU for numpy conversion
            else:
                # Fallback for unexpected batch format - may need adjustment
                # This assumes batch is a tuple (data_part, label_part)
                # And data_part itself needs to be moved to device
                # You might need to adjust this based on your specific non-dict dataloader
                print("Warning: Unexpected batch format (not dict). Trying tuple format.")
                data_part, labels = batch 
                # Assuming data_part needs to be moved to device; Adapt as necessary!
                model_input = data_part.to(model.device) 
            
            # Extract features (expecting normalized features from PointNeXtFeatureExtractor)
            features = model.extract_features(model_input)
            
            # Store features (CPU) and labels (CPU)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy()) # Ensure labels are numpy arrays
    
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
    parser.add_argument('--cfg', type=str, required=True, 
                        help='Path to the primary config file (e.g., model architecture)')
    parser.add_argument('--data_cfg', type=str, required=True, 
                        help='Path to the data config file (e.g., datasets, dataloaders, default.yaml)')
    parser.add_argument('--pretrained', type=str, required=True, help='pretrained model path')
    parser.add_argument('--save_dir', type=str, default='data/features', help='directory to save features')
    parser.add_argument('--subset', type=str, default='train', 
                        help='subset to extract features from (train, val, test, or all)')
    return parser.parse_args()

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config) # Convert to EasyDict for attribute access

def main():
    args = parse_args()
    
    # Load configurations
    model_config = load_config(args.cfg)
    data_config = load_config(args.data_cfg)
    
    # --- Combine configs if necessary (example assumes simple merge, adjust if needed) ---
    # If model config needs parts of data config or vice-versa, merge them here.
    # For now, assume model_config.model and data_config sections are sufficient.
    # config = edict({**model_config, **data_config}) # Example merge
    # For clarity, we'll use model_config and data_config directly
    
    # Build model
    # Assuming model definition is in model_config (e.g., args.cfg like pointnext-s.yaml)
    model = PointNeXtFeatureExtractor(model_cfg=model_config.model, pretrained_path=args.pretrained)
    model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(model.device)
    
    # Define which subsets to process
    subsets_to_process = []
    if args.subset == 'all':
        subsets_to_process = ['train', 'val', 'test']
    elif args.subset in ['train', 'val', 'test']:
        subsets_to_process = [args.subset]
    else:
        raise ValueError(f"Invalid subset specified: {args.subset}. Choose from train, val, test, all.")

    # Process subsets
    # Assumes data definitions are in data_config (e.g., args.data_cfg like default.yaml)
    for subset in subsets_to_process:
        print(f"\nProcessing subset: {subset}")
        if not hasattr(data_config, 'dataset') or not hasattr(data_config.dataset, subset):
             print(f"Warning: Dataset config for subset '{subset}' not found in {args.data_cfg}. Skipping.")
             continue
        if not hasattr(data_config, 'dataloader'):
             print(f"Warning: Dataloader config not found in {args.data_cfg}. Skipping {subset}.")
             continue
            
        # Determine batch size for the current subset
        batch_size = data_config.get('val_batch_size', data_config.batch_size) if subset in ['val', 'test'] else data_config.batch_size
        
        print(f"Building dataloader for {subset}...")
        try:
            data_loader = build_dataloader_from_cfg(
                batch_size=batch_size,
                dataset_cfg=data_config.dataset, 
                dataloader_cfg=data_config.dataloader, 
                datatransforms_cfg=data_config.get('datatransforms'), # Use get for optional transforms
                split=subset, 
                distributed=False # Assuming non-distributed extraction for simplicity
            )
        except Exception as e:
            print(f"Error building dataloader for subset '{subset}': {e}")
            continue
            
        print(f"Extracting features for {subset}...")
        extract_and_save_features(model, data_loader, args.save_dir, subset)

if __name__ == "__main__":
    main() 