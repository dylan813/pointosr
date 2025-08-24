#!/usr/bin/env python3
"""
Phase 4: Define the two raw signals (Energy + Cosine)

This script implements and computes:
1. Energy Score: Temperature-scaled confidence from logits
   - Uses optimal temperature T from Phase 3
   - Higher values = more ID-like (confident predictions)
   
2. Cosine Score: Class-conditional similarity to prototypes  
   - For predicted class c, compute max cosine to class c prototypes
   - Higher values = more ID-like (similar to training patterns)

Both scores follow "higher is better" direction for ID samples.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Add the pointosr package to Python path
pointosr_path = '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr'
sys.path.insert(0, pointosr_path)

from pointnext.utils import EasyConfig
from pointnext.dataset.human.human import HumanDataset
from pointnext.model import build_model_from_cfg

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyScorer:
    """Compute temperature-scaled energy scores."""
    
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        
    def compute_energy(self, logits):
        """
        Compute energy score from logits.
        
        Energy = T * log(sum(exp(z_i / T))) where z_i are logits
        
        Higher energy = more confident = more ID-like
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Compute energy (logsumexp)
        energy = self.temperature * torch.logsumexp(scaled_logits, dim=1)
        
        return energy

class CosineScorer:
    """Compute class-conditional cosine similarity to prototypes."""
    
    def __init__(self, human_prototypes, fp_prototypes):
        """
        Args:
            human_prototypes: (K_H, embed_dim) array of human prototypes
            fp_prototypes: (K_F, embed_dim) array of false-positive prototypes
        """
        self.human_prototypes = human_prototypes
        self.fp_prototypes = fp_prototypes
        
    def compute_cosine(self, embeddings, predicted_classes):
        """
        Compute class-conditional cosine similarity.
        
        For each sample with predicted class c:
        - If c == 0 (human): compute max cosine to human prototypes
        - If c == 1 (false): compute max cosine to FP prototypes
        
        Args:
            embeddings: (N, embed_dim) embeddings
            predicted_classes: (N,) predicted class labels
            
        Returns:
            cosine_scores: (N,) max cosine similarities
        """
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
        predicted_classes_np = predicted_classes.cpu().numpy() if torch.is_tensor(predicted_classes) else predicted_classes
        
        cosine_scores = np.zeros(len(embeddings_np))
        
        for i, (embedding, pred_class) in enumerate(zip(embeddings_np, predicted_classes_np)):
            embedding = embedding.reshape(1, -1)
            
            if pred_class == 0:  # Human class
                # Compute cosine similarity to all human prototypes
                similarities = cosine_similarity(embedding, self.human_prototypes)[0]
                cosine_scores[i] = np.max(similarities)
            else:  # False-positive class
                # Compute cosine similarity to all FP prototypes
                similarities = cosine_similarity(embedding, self.fp_prototypes)[0]
                cosine_scores[i] = np.max(similarities)
        
        return torch.tensor(cosine_scores, dtype=torch.float32)

class ScoreExtractor:
    """Extract both energy and cosine scores from a model."""
    
    def __init__(self, model_path, config_path, fusion_config_path, prototypes_dir, device='cuda'):
        self.device = device
        
        # Load model
        self.model = self.load_model(model_path, config_path)
        
        # Load fusion config for temperature
        with open(fusion_config_path, 'r') as f:
            fusion_config = json.load(f)
        self.temperature = fusion_config['T']
        
        # Load prototypes
        human_prototypes = np.load(os.path.join(prototypes_dir, 'human_k6', 'prototypes.npy'))
        fp_prototypes = np.load(os.path.join(prototypes_dir, 'fp_k4', 'prototypes.npy'))
        
        # Initialize scorers
        self.energy_scorer = EnergyScorer(self.temperature)
        self.cosine_scorer = CosineScorer(human_prototypes, fp_prototypes)
        
        logger.info(f"ScoreExtractor initialized with T={self.temperature:.4f}")
        logger.info(f"Human prototypes: {human_prototypes.shape}")
        logger.info(f"FP prototypes: {fp_prototypes.shape}")
        
    def load_model(self, model_path, config_path):
        """Load the trained model with embedding extraction capability."""
        logger.info(f"Loading model from {model_path}")
        
        # Load config
        cfg = EasyConfig()
        cfg.load(config_path, recursive=True)
        
        # Build model
        model = build_model_from_cfg(cfg.model)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
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
        
        # Override forward method to extract both logits and embeddings
        original_forward = model.forward
        def enhanced_forward(data):
            # Get embeddings from encoder
            global_feat = model.encoder.forward_cls_feat(data)
            # Get logits from classifier
            logits = model.prediction(global_feat)
            # Return both
            return logits, global_feat
        
        model.forward = enhanced_forward
        
        logger.info("Model loaded successfully with enhanced forward pass")
        return model
        
    def extract_scores(self, dataset, batch_size=32):
        """Extract energy and cosine scores from dataset."""
        logger.info(f"Extracting scores from {len(dataset)} samples")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        all_energy_scores = []
        all_cosine_scores = []
        all_logits = []
        all_embeddings = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting scores"):
                # Move batch to device
                pos = batch['pos'].to(self.device)
                x = batch['x'].to(self.device)
                labels = batch['y']
                
                # Prepare input for model
                data = {'pos': pos, 'x': x.transpose(1, 2).contiguous()}
                
                # Forward pass - get both logits and embeddings
                logits, embeddings = self.model(data)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Compute energy scores
                energy_scores = self.energy_scorer.compute_energy(logits)
                
                # Compute cosine scores
                cosine_scores = self.cosine_scorer.compute_cosine(embeddings.cpu(), predictions.cpu())
                
                # Store results
                all_energy_scores.append(energy_scores.cpu())
                all_cosine_scores.append(cosine_scores)
                all_logits.append(logits.cpu())
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                all_predictions.append(predictions.cpu())
        
        # Concatenate all results
        results = {
            'energy_scores': torch.cat(all_energy_scores, dim=0),
            'cosine_scores': torch.cat(all_cosine_scores, dim=0),
            'logits': torch.cat(all_logits, dim=0),
            'embeddings': torch.cat(all_embeddings, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'predictions': torch.cat(all_predictions, dim=0)
        }
        
        logger.info(f"Extracted scores:")
        logger.info(f"  Energy scores shape: {results['energy_scores'].shape}")
        logger.info(f"  Cosine scores shape: {results['cosine_scores'].shape}")
        logger.info(f"  Accuracy: {(results['predictions'] == results['labels']).float().mean():.4f}")
        
        return results

def analyze_scores(results):
    """Analyze the extracted scores."""
    energy_scores = results['energy_scores'].numpy()
    cosine_scores = results['cosine_scores'].numpy()
    labels = results['labels'].numpy()
    predictions = results['predictions'].numpy()
    
    logger.info("\n📊 Score Analysis:")
    
    # Overall statistics
    logger.info(f"Energy scores - min: {energy_scores.min():.4f}, max: {energy_scores.max():.4f}, mean: {energy_scores.mean():.4f}")
    logger.info(f"Cosine scores - min: {cosine_scores.min():.4f}, max: {cosine_scores.max():.4f}, mean: {cosine_scores.mean():.4f}")
    
    # Per-class statistics
    for class_idx, class_name in enumerate(['human', 'fp']):
        class_mask = labels == class_idx
        if np.sum(class_mask) > 0:
            logger.info(f"\n{class_name.upper()} class (n={np.sum(class_mask)}):")
            logger.info(f"  Energy - mean: {energy_scores[class_mask].mean():.4f}, std: {energy_scores[class_mask].std():.4f}")
            logger.info(f"  Cosine - mean: {cosine_scores[class_mask].mean():.4f}, std: {cosine_scores[class_mask].std():.4f}")
    
    # Prediction accuracy per class
    for class_idx, class_name in enumerate(['human', 'fp']):
        class_mask = labels == class_idx
        if np.sum(class_mask) > 0:
            accuracy = (predictions[class_mask] == labels[class_mask]).mean()
            logger.info(f"{class_name} classification accuracy: {accuracy:.4f}")
    
    # Score correlation
    correlation = np.corrcoef(energy_scores, cosine_scores)[0, 1]
    logger.info(f"\nEnergy-Cosine correlation: {correlation:.4f}")
    
    return {
        'energy_stats': {
            'min': float(energy_scores.min()),
            'max': float(energy_scores.max()),
            'mean': float(energy_scores.mean()),
            'std': float(energy_scores.std())
        },
        'cosine_stats': {
            'min': float(cosine_scores.min()),
            'max': float(cosine_scores.max()),
            'mean': float(cosine_scores.mean()),
            'std': float(cosine_scores.std())
        },
        'correlation': float(correlation),
        'accuracy': float((predictions == labels).mean())
    }

def main():
    parser = argparse.ArgumentParser(description='Compute raw Energy and Cosine scores for OSR')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--fusion_config', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration/fusion_config.json',
                       help='Path to fusion config with temperature')
    parser.add_argument('--prototypes_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
                       help='Directory containing prototype files')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/cerlab/Documents/data/pointosr/human',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration/scores',
                       help='Output directory for score files')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    logger.info("Starting raw score computation for OSR")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Fusion config: {args.fusion_config}")
    logger.info(f"Prototypes: {args.prototypes_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize score extractor
    extractor = ScoreExtractor(
        args.model_path, 
        args.config_path, 
        args.fusion_config,
        args.prototypes_dir,
        args.device
    )
    
    # Load calibration dataset
    cal_dataset = HumanDataset(
        data_dir=args.data_dir,
        split='cal',
        num_points=2048,
        transform=None,
        uniform_sample=False
    )
    
    logger.info(f"Loaded calibration dataset: {len(cal_dataset)} samples")
    
    # Extract scores
    results = extractor.extract_scores(cal_dataset, args.batch_size)
    
    # Analyze scores
    score_analysis = analyze_scores(results)
    
    # Save raw scores
    scores_data = {
        'energy_scores': results['energy_scores'].numpy().tolist(),
        'cosine_scores': results['cosine_scores'].numpy().tolist(),
        'labels': results['labels'].numpy().tolist(),
        'predictions': results['predictions'].numpy().tolist(),
        'sample_indices': list(range(len(results['labels']))),
        'score_analysis': score_analysis,
        'metadata': {
            'temperature': extractor.temperature,
            'num_samples': len(cal_dataset),
            'score_directions': {
                'energy': 'higher_is_better',
                'cosine': 'higher_is_better'
            }
        }
    }
    
    scores_file = os.path.join(args.output_dir, 'raw_scores_calibration.json')
    with open(scores_file, 'w') as f:
        json.dump(scores_data, f, indent=2)
    
    # Save score definitions documentation
    score_definitions = {
        "energy_score": {
            "definition": "T * log(sum(exp(z_i / T))) where z_i are logits and T is temperature",
            "temperature": extractor.temperature,
            "direction": "higher_is_better",
            "interpretation": "Higher energy indicates more confident predictions (more ID-like)"
        },
        "cosine_score": {
            "definition": "max cosine similarity to prototypes of predicted class",
            "num_human_prototypes": 6,
            "num_fp_prototypes": 4,
            "direction": "higher_is_better", 
            "interpretation": "Higher cosine indicates more similar to training patterns (more ID-like)"
        },
        "score_combination": {
            "fusion_approach": "weighted_combination",
            "initial_weights": [0.5, 0.5],
            "normalization": "per_class_percentile_to_[0,1]",
            "next_phase": "Phase 5 - Score normalization"
        }
    }
    
    definitions_file = os.path.join(args.output_dir, 'score_definitions.json')
    with open(definitions_file, 'w') as f:
        json.dump(score_definitions, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 Raw score computation completed successfully!")
    logger.info(f"Energy Score: T={extractor.temperature:.4f} * logsumexp(logits/T)")
    logger.info(f"Cosine Score: max cosine to predicted class prototypes")
    logger.info(f"Raw scores saved to: {scores_file}")
    logger.info(f"Score definitions saved to: {definitions_file}")
    logger.info(f"✅ Ready for Phase 5: Score normalization")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
