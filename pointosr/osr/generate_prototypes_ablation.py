#!/usr/bin/env python3
"""
Ablation Study: Generate class-conditional prototypes using K-means clustering.

This script generates prototypes for multiple configurations:
- Human prototypes: 4, 8, 16
- False-positive prototypes: 0, 1, 2

Usage:
    python generate_prototypes_ablation.py --model_path /path/to/model.pth --config_path /path/to/config.yaml
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
        """Load trained model and configuration."""
        logger.info(f"Loading model from {model_path}")
        
        # Load config
        cfg = EasyConfig()
        cfg.load(config_path, recursive=True)
        
        # Build model
        self.model = build_model_from_cfg(cfg.model).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def extract_embeddings(self, dataset, batch_size=32):
        """Extract embeddings from dataset."""
        logger.info(f"Extracting embeddings from {len(dataset)} samples...")
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        self.embeddings = []
        self.labels = []
        self.sample_ids = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
                # Handle batch data structure from HumanDataset
                if isinstance(batch, list):
                    # Collate batch data
                    pos = torch.stack([item['pos'] for item in batch]).to(self.device)
                    x = torch.stack([item['x'] for item in batch]).to(self.device)
                    labels = torch.stack([item['y'] for item in batch]).to(self.device)
                else:
                    # Direct batch
                    pos = batch['pos'].to(self.device)
                    x = batch['x'].to(self.device)
                    labels = batch['y'].to(self.device)
                
                # Prepare data in the format expected by the model
                # pos: [B, N, 3] -> [B, N, 3] (keep as is)
                # x: [B, N, 4] -> [B, 4, N] (transpose for model)
                data = {
                    'pos': pos.contiguous(),  # [B, N, 3]
                    'x': x.transpose(1, 2).contiguous()  # [B, N, 4] -> [B, 4, N]
                }
                
                sample_ids = [f"batch_{batch_idx}_{i}" for i in range(len(pos))]
                
                # Forward pass to get embeddings (penultimate layer)
                # Use the encoder's forward_cls_feat method to get global features
                embeddings = self.model.encoder.forward_cls_feat(data)
                
                self.embeddings.append(embeddings.cpu().numpy())
                self.labels.append(labels.cpu().numpy())
                self.sample_ids.extend(sample_ids)
        
        # Concatenate all embeddings
        self.embeddings = np.concatenate(self.embeddings, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        logger.info(f"Extracted {len(self.embeddings)} embeddings with shape {self.embeddings.shape}")
        return self.embeddings, self.labels

def generate_prototypes_for_config(embeddings, labels, k_human, k_fp, output_dir, config_name):
    """Generate prototypes for a specific configuration."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Generating prototypes for {config_name}")
    logger.info(f"Human prototypes: K={k_human}, False-positive prototypes: K={k_fp}")
    logger.info(f"{'='*50}")
    
    # Separate embeddings by class
    human_mask = labels == 0
    fp_mask = labels == 1
    
    human_embeddings = embeddings[human_mask]
    fp_embeddings = embeddings[fp_mask]
    
    logger.info(f"Human samples: {len(human_embeddings)}")
    logger.info(f"False-positive samples: {len(fp_embeddings)}")
    
    # Generate human prototypes
    if len(human_embeddings) >= k_human:
        human_kmeans = KMeans(n_clusters=k_human, random_state=42, n_init=10)
        human_kmeans.fit(human_embeddings)
        human_prototypes = human_kmeans.cluster_centers_
        human_inertia = human_kmeans.inertia_
        logger.info(f"Human prototypes generated (inertia: {human_inertia:.4f})")
    else:
        logger.warning(f"Not enough human samples ({len(human_embeddings)}) for {k_human} prototypes")
        human_prototypes = human_embeddings[:k_human] if len(human_embeddings) > 0 else np.zeros((k_human, embeddings.shape[1]))
    
    # Generate false-positive prototypes (handle k_fp = 0 case)
    if k_fp == 0:
        logger.info("Skipping FP prototype generation (k_fp = 0)")
        fp_prototypes = np.array([]).reshape(0, embeddings.shape[1])  # Empty array with correct shape
        fp_inertia = None
    elif len(fp_embeddings) >= k_fp:
        fp_kmeans = KMeans(n_clusters=k_fp, random_state=42, n_init=10)
        fp_kmeans.fit(fp_embeddings)
        fp_prototypes = fp_kmeans.cluster_centers_
        fp_inertia = fp_kmeans.inertia_
        logger.info(f"False-positive prototypes generated (inertia: {fp_inertia:.4f})")
    else:
        logger.warning(f"Not enough FP samples ({len(fp_embeddings)}) for {k_fp} prototypes")
        fp_prototypes = fp_embeddings[:k_fp] if len(fp_embeddings) > 0 else np.zeros((k_fp, embeddings.shape[1]))
    
    # Create output directories for this specific configuration
    config_dir = os.path.join(output_dir, config_name)
    human_dir = os.path.join(config_dir, f'human_k{k_human}')
    fp_dir = os.path.join(config_dir, f'fp_k{k_fp}')
    
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)
    
    # Save prototypes
    human_prototypes_path = os.path.join(human_dir, 'prototypes.npy')
    fp_prototypes_path = os.path.join(fp_dir, 'prototypes.npy')
    
    np.save(human_prototypes_path, human_prototypes)
    np.save(fp_prototypes_path, fp_prototypes)
    
    # Save metadata
    human_metadata = {
        'class': 'human',
        'k_prototypes': k_human,
        'prototype_shape': human_prototypes.shape,
        'generation_method': 'kmeans',
        'training_samples': len(human_embeddings),
        'inertia': float(human_inertia) if 'human_inertia' in locals() else None,
        'config_name': config_name
    }
    
    fp_metadata = {
        'class': 'false_positive',
        'k_prototypes': k_fp,
        'prototype_shape': fp_prototypes.shape,
        'generation_method': 'kmeans' if k_fp > 0 else 'none',
        'training_samples': len(fp_embeddings),
        'inertia': float(fp_inertia) if fp_inertia is not None else None,
        'config_name': config_name,
        'note': 'No FP prototypes generated (k_fp = 0)' if k_fp == 0 else None
    }
    
    with open(os.path.join(human_dir, 'metadata.json'), 'w') as f:
        json.dump(human_metadata, f, indent=2)
    
    with open(os.path.join(fp_dir, 'metadata.json'), 'w') as f:
        json.dump(fp_metadata, f, indent=2)
    
    logger.info(f"Prototypes saved:")
    logger.info(f"  Human: {human_prototypes_path}")
    logger.info(f"  False-positive: {fp_prototypes_path}")
    
    return human_prototypes, fp_prototypes

def main():
    parser = argparse.ArgumentParser(description='Generate prototypes for ablation study')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/cerlab/Documents/data/pointosr/dataset',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
                       help='Output directory for prototypes')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Prototype Generation for Ablation Study")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Output: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info("📂 Loading training dataset...")
    train_dataset = HumanDataset(
        data_dir=args.data_dir,
        split='train',
        num_points=2048,
        transform=None,  # No augmentation for prototype generation
        uniform_sample=False
    )
    logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
    
    # Extract embeddings
    extractor = EmbeddingExtractor(args.model_path, args.config_path, args.device)
    embeddings, labels = extractor.extract_embeddings(train_dataset, args.batch_size)
    
    # Ablation study configurations (including 0 FP prototypes)
    configurations = [
        {'k_human': 4, 'k_fp': 0, 'name': 'human_k4_fp_k0'},
        {'k_human': 4, 'k_fp': 1, 'name': 'human_k4_fp_k1'},
        {'k_human': 4, 'k_fp': 2, 'name': 'human_k4_fp_k2'},
        {'k_human': 8, 'k_fp': 0, 'name': 'human_k8_fp_k0'},
        {'k_human': 8, 'k_fp': 1, 'name': 'human_k8_fp_k1'},
        {'k_human': 8, 'k_fp': 2, 'name': 'human_k8_fp_k2'},
        {'k_human': 16, 'k_fp': 0, 'name': 'human_k16_fp_k0'},
        {'k_human': 16, 'k_fp': 1, 'name': 'human_k16_fp_k1'},
        {'k_human': 16, 'k_fp': 2, 'name': 'human_k16_fp_k2'},
    ]
    
    # Generate prototypes for each configuration
    all_results = {}
    
    for config in configurations:
        k_human = config['k_human']
        k_fp = config['k_fp']
        config_name = config['name']
        
        human_prototypes, fp_prototypes = generate_prototypes_for_config(
            embeddings, labels, k_human, k_fp, args.output_dir, config_name
        )
        
        all_results[config_name] = {
            'k_human': k_human,
            'k_fp': k_fp,
            'human_prototypes_shape': human_prototypes.shape,
            'fp_prototypes_shape': fp_prototypes.shape,
            'human_dir': f'human_k{k_human}',
            'fp_dir': f'fp_k{k_fp}'
        }
    
    # Save ablation study summary
    ablation_summary = {
        'ablation_study': {
            'description': 'K-means prototype ablation study (including 0 FP prototypes)',
            'total_configurations': len(configurations),
            'configurations': all_results,
            'embedding_extraction': {
                'model_path': args.model_path,
                'config_path': args.config_path,
                'data_dir': args.data_dir,
                'total_training_samples': len(embeddings),
                'embedding_dim': embeddings.shape[1]
            }
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'ablation_prototype_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 Ablation Study Prototype Generation Completed!")
    logger.info(f"Generated prototypes for {len(configurations)} configurations:")
    for config_name, result in all_results.items():
        logger.info(f"  {config_name}: Human K={result['k_human']}, FP K={result['k_fp']}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
