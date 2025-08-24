#!/usr/bin/env python3
"""
Phase 2: Generate class-conditional prototypes using K-means clustering.

This script:
1. Loads the trained PointNeXt-S model
2. Extracts penultimate layer embeddings from training set
3. Creates K_H=6 prototypes for human class
4. Creates K_F=4 prototypes for false-positive class
5. Saves prototypes and metadata for OSR system
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import json
from tqdm import tqdm
import logging

# Add the pointosr package to Python path
pointosr_path = '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr'
sys.path.insert(0, pointosr_path)

from pointnext.utils import EasyConfig
from pointnext.dataset.human.human import HumanDataset
from pointnext.model import build_model_from_cfg
from easydict import EasyDict as edict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    """Extract penultimate layer embeddings from trained model."""
    
    def __init__(self, model_path, config_path, device='cuda'):
        self.device = device
        self.model = None
        self.embeddings = []
        self.labels = []
        self.sample_ids = []
        
        # Load model
        self.load_model(model_path, config_path)
        
    def load_model(self, model_path, config_path):
        """Load the trained model."""
        logger.info(f"Loading model from {model_path}")
        
        # Load config
        cfg = EasyConfig()
        cfg.load(config_path, recursive=True)
        
        # Build model
        model = build_model_from_cfg(cfg.model)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        # Register hook to extract embeddings
        self.register_hook(model)
        
        self.model = model
        logger.info("Model loaded successfully")
        
    def register_hook(self, model):
        """Register forward hook to extract penultimate layer features."""
        def hook_fn(module, input, output):
            # Store the embeddings (encoder output before classification head)
            if isinstance(output, torch.Tensor):
                self.current_embeddings = output.detach().cpu()
        
        # Hook into the encoder to get global features before classification
        target_layer = model.encoder
        target_layer.register_forward_hook(hook_fn)
        logger.info(f"Registered hook on encoder: {target_layer}")
        
        # Also need to override the forward method to capture the encoder output
        original_forward = model.forward
        def hooked_forward(data):
            global_feat = model.encoder.forward_cls_feat(data)
            self.current_embeddings = global_feat.detach().cpu()
            return model.prediction(global_feat)
        
        model.forward = hooked_forward
    
    def extract_embeddings(self, dataset, batch_size=32):
        """Extract embeddings from dataset."""
        logger.info(f"Extracting embeddings from {len(dataset)} samples")
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        all_embeddings = []
        all_labels = []
        all_ids = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
                # Move batch to device
                pos = batch['pos'].to(self.device)
                x = batch['x'].to(self.device)
                labels = batch['y']
                
                # Prepare input for model (transpose for conv1d: batch, features, points)
                data = {'pos': pos, 'x': x.transpose(1, 2).contiguous()}
                
                # Forward pass (this triggers our hook)
                _ = self.model(data)
                
                # Store embeddings and labels
                if hasattr(self, 'current_embeddings'):
                    all_embeddings.append(self.current_embeddings)
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Generate sample IDs
                    batch_start = i * batch_size
                    batch_ids = [f"train_sample_{batch_start + j}" for j in range(len(labels))]
                    all_ids.extend(batch_ids)
        
        # Concatenate all embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings, np.array(all_labels), all_ids
        else:
            raise ValueError("No embeddings were extracted. Check hook registration.")

def generate_class_prototypes(embeddings, labels, sample_ids, class_idx, k_prototypes, class_name):
    """Generate K prototypes for a specific class using K-means."""
    logger.info(f"Generating {k_prototypes} prototypes for class {class_name} (idx={class_idx})")
    
    # Filter embeddings for this class
    class_mask = labels == class_idx
    class_embeddings = embeddings[class_mask]
    class_sample_ids = [sample_ids[i] for i in range(len(sample_ids)) if class_mask[i]]
    
    logger.info(f"Found {len(class_embeddings)} samples for class {class_name}")
    
    if len(class_embeddings) < k_prototypes:
        logger.warning(f"Only {len(class_embeddings)} samples available, but {k_prototypes} prototypes requested")
        k_prototypes = len(class_embeddings)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k_prototypes, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(class_embeddings)
    
    # Get prototype centers
    prototypes = kmeans.cluster_centers_
    
    # Get sample IDs used for each prototype (closest samples to centers)
    prototype_sample_ids = []
    for i in range(k_prototypes):
        cluster_mask = cluster_assignments == i
        cluster_embeddings = class_embeddings[cluster_mask]
        cluster_sample_ids = [class_sample_ids[j] for j in range(len(class_sample_ids)) if cluster_mask[j]]
        
        if len(cluster_embeddings) > 0:
            # Find closest sample to prototype center
            distances = np.linalg.norm(cluster_embeddings - prototypes[i], axis=1)
            closest_idx = np.argmin(distances)
            prototype_sample_ids.append(cluster_sample_ids[closest_idx])
        else:
            prototype_sample_ids.append(f"prototype_{i}_empty")
    
    return prototypes, prototype_sample_ids, cluster_assignments

def save_prototypes(prototypes, sample_ids, metadata, output_dir):
    """Save prototypes and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prototypes as numpy array
    prototype_path = os.path.join(output_dir, 'prototypes.npy')
    np.save(prototype_path, prototypes)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(prototypes)} prototypes to {output_dir}")
    logger.info(f"  - Prototypes: {prototype_path}")
    logger.info(f"  - Metadata: {metadata_path}")
    
    return prototype_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description='Generate class-conditional prototypes for OSR')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/cerlab/Documents/data/pointosr/human',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
                       help='Output directory for prototypes')
    parser.add_argument('--k_human', type=int, default=6,
                       help='Number of human prototypes (K_H)')
    parser.add_argument('--k_false', type=int, default=4,
                       help='Number of false-positive prototypes (K_F)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    logger.info("Starting prototype generation for OSR")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"K_H (human): {args.k_human}, K_F (fp): {args.k_false}")
    
    # Create embedding extractor
    extractor = EmbeddingExtractor(args.model_path, args.config_path, args.device)
    
    # Load training dataset
    logger.info("Loading training dataset...")
    train_dataset = HumanDataset(
        data_dir=args.data_dir,
        split='train',
        num_points=2048,
        transform=None,  # No augmentation for prototype generation
        uniform_sample=False
    )
    
    # Extract embeddings
    embeddings, labels, sample_ids = extractor.extract_embeddings(train_dataset, args.batch_size)
    
    # Generate human prototypes (class 0)
    human_prototypes, human_sample_ids, human_clusters = generate_class_prototypes(
        embeddings, labels, sample_ids, class_idx=0, k_prototypes=args.k_human, class_name="human"
    )
    
    # Generate false-positive prototypes (class 1)
    fp_prototypes, fp_sample_ids, fp_clusters = generate_class_prototypes(
        embeddings, labels, sample_ids, class_idx=1, k_prototypes=args.k_false, class_name="fp"
    )
    
    # Save human prototypes
    human_metadata = {
        'class_name': 'human',
        'class_idx': 0,
        'num_prototypes': len(human_prototypes),
        'embedding_dim': human_prototypes.shape[1],
        'prototype_sample_ids': human_sample_ids,
        'total_samples_used': int(np.sum(labels == 0)),
        'generation_config': {
            'k_means_random_state': 42,
            'k_means_n_init': 10
        }
    }
    
    human_dir = os.path.join(args.output_dir, 'human_k6')
    save_prototypes(human_prototypes, human_sample_ids, human_metadata, human_dir)
    
    # Save false-positive prototypes
    fp_metadata = {
        'class_name': 'false_positive',
        'class_idx': 1,
        'num_prototypes': len(fp_prototypes),
        'embedding_dim': fp_prototypes.shape[1],
        'prototype_sample_ids': fp_sample_ids,
        'total_samples_used': int(np.sum(labels == 1)),
        'generation_config': {
            'k_means_random_state': 42,
            'k_means_n_init': 10
        }
    }
    
    fp_dir = os.path.join(args.output_dir, 'fp_k4')
    save_prototypes(fp_prototypes, fp_sample_ids, fp_metadata, fp_dir)
    
    # Create summary
    summary = {
        'human_prototypes': {
            'path': human_dir,
            'count': len(human_prototypes),
            'embedding_dim': human_prototypes.shape[1]
        },
        'fp_prototypes': {
            'path': fp_dir,
            'count': len(fp_prototypes), 
            'embedding_dim': fp_prototypes.shape[1]
        },
        'total_training_samples': len(embeddings),
        'embedding_extraction': {
            'model_path': args.model_path,
            'config_path': args.config_path,
            'data_dir': args.data_dir
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'prototype_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 Prototype generation completed successfully!")
    logger.info(f"Human prototypes (K_H={args.k_human}): {human_dir}")
    logger.info(f"False-positive prototypes (K_F={args.k_false}): {fp_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
