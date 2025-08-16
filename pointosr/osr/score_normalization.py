#!/usr/bin/env python3
"""
Phase 5: Score Normalization

This script implements per-class percentile normalization to bring Energy and Cosine 
scores onto a shared [0,1] scale for fair fusion.

Normalization Strategy:
1. For each class (human, false) and each score (energy, cosine):
   - Build empirical CDF from calibration data  
   - Map scores to [0,1] via percentile transformation
   - Handle out-of-range values with clipping

2. At inference time:
   - Use predicted class to select appropriate normalization
   - Apply class-specific mappings to both scores
   - Clip to [0,1] range for safety
"""

import os
import sys
import argparse
import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PercentileNormalizer:
    """Per-class percentile normalization for OSR scores."""
    
    def __init__(self, n_percentiles=100):
        """
        Args:
            n_percentiles: Number of percentile points to store (default: 100 for 1% resolution)
        """
        self.n_percentiles = n_percentiles
        self.normalizers = {}
        
    def fit(self, scores, labels, score_name):
        """
        Fit per-class percentile normalizers.
        
        Args:
            scores: (N,) array of scores
            labels: (N,) array of class labels
            score_name: str, name of the score ('energy' or 'cosine')
        """
        logger.info(f"Fitting percentile normalizer for {score_name} scores")
        
        self.normalizers[score_name] = {}
        
        # Fit normalizer for each class
        for class_idx in [0, 1]:  # Human, False
            class_name = 'human' if class_idx == 0 else 'false'
            class_mask = labels == class_idx
            class_scores = scores[class_mask]
            
            if len(class_scores) == 0:
                logger.warning(f"No samples found for class {class_name}")
                continue
                
            logger.info(f"  Class {class_name}: {len(class_scores)} samples")
            logger.info(f"    Score range: [{class_scores.min():.4f}, {class_scores.max():.4f}]")
            logger.info(f"    Score mean±std: {class_scores.mean():.4f}±{class_scores.std():.4f}")
            
            # Create percentile mapping
            percentiles = np.linspace(0, 100, self.n_percentiles + 1)
            percentile_values = np.percentile(class_scores, percentiles)
            
            # Store normalization info
            self.normalizers[score_name][class_idx] = {
                'class_name': class_name,
                'n_samples': len(class_scores),
                'raw_min': float(class_scores.min()),
                'raw_max': float(class_scores.max()),
                'raw_mean': float(class_scores.mean()),
                'raw_std': float(class_scores.std()),
                'percentiles': percentiles.tolist(),
                'percentile_values': percentile_values.tolist(),
                'clip_policy': 'clip_to_[0,1]'
            }
            
            logger.info(f"    Percentile normalizer fitted for class {class_name}")
    
    def transform(self, scores, predicted_classes, score_name):
        """
        Transform scores using fitted percentile normalizers.
        
        Args:
            scores: (N,) array of scores to normalize
            predicted_classes: (N,) array of predicted class labels
            score_name: str, name of the score ('energy' or 'cosine')
            
        Returns:
            normalized_scores: (N,) array of normalized scores in [0,1]
        """
        if score_name not in self.normalizers:
            raise ValueError(f"Normalizer for {score_name} not fitted")
            
        normalized_scores = np.zeros_like(scores)
        
        for class_idx in [0, 1]:
            if class_idx not in self.normalizers[score_name]:
                continue
                
            # Get samples predicted as this class
            class_mask = predicted_classes == class_idx
            if not np.any(class_mask):
                continue
                
            class_scores = scores[class_mask]
            normalizer = self.normalizers[score_name][class_idx]
            
            # Transform using percentile mapping
            percentile_values = np.array(normalizer['percentile_values'])
            percentiles = np.array(normalizer['percentiles'])
            
            # Use interpolation to map scores to percentiles, then to [0,1]
            normalized_percentiles = np.interp(class_scores, percentile_values, percentiles)
            normalized_class_scores = normalized_percentiles / 100.0  # Convert to [0,1]
            
            # Clip to [0,1] range
            normalized_class_scores = np.clip(normalized_class_scores, 0.0, 1.0)
            
            normalized_scores[class_mask] = normalized_class_scores
            
        return normalized_scores
    
    def save(self, output_path):
        """Save normalization mappings to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.normalizers, f, indent=2)
        logger.info(f"Normalization mappings saved to: {output_path}")
    
    def load(self, input_path):
        """Load normalization mappings from JSON."""
        with open(input_path, 'r') as f:
            self.normalizers = json.load(f)
        logger.info(f"Normalization mappings loaded from: {input_path}")

def analyze_normalization_quality(raw_scores, normalized_scores, labels, predictions, score_name):
    """Analyze the quality of normalization."""
    logger.info(f"\n📊 Normalization Analysis for {score_name.upper()} scores:")
    
    # Overall statistics
    logger.info(f"Raw scores - range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}], mean: {raw_scores.mean():.4f}")
    logger.info(f"Normalized - range: [{normalized_scores.min():.4f}, {normalized_scores.max():.4f}], mean: {normalized_scores.mean():.4f}")
    
    # Per-class analysis
    for class_idx, class_name in enumerate(['human', 'false']):
        # Use predicted classes for normalization analysis (as in real inference)
        pred_mask = predictions == class_idx
        if not np.any(pred_mask):
            continue
            
        raw_class = raw_scores[pred_mask]
        norm_class = normalized_scores[pred_mask]
        
        logger.info(f"\n{class_name.upper()} class (predicted, n={np.sum(pred_mask)}):")
        logger.info(f"  Raw scores - mean: {raw_class.mean():.4f}, std: {raw_class.std():.4f}")
        logger.info(f"  Normalized - mean: {norm_class.mean():.4f}, std: {norm_class.std():.4f}")
        logger.info(f"  Coverage: {np.sum((norm_class > 0) & (norm_class < 1)) / len(norm_class):.1%} in (0,1)")
        logger.info(f"  Clipped low: {np.sum(norm_class == 0.0)}, high: {np.sum(norm_class == 1.0)}")
    
    # Check for preservation of relative ordering
    correlation = np.corrcoef(raw_scores, normalized_scores)[0, 1]
    logger.info(f"\nRank correlation (Spearman): {stats.spearmanr(raw_scores, normalized_scores)[0]:.4f}")
    logger.info(f"Linear correlation (Pearson): {correlation:.4f}")
    
    return {
        'raw_range': [float(raw_scores.min()), float(raw_scores.max())],
        'normalized_range': [float(normalized_scores.min()), float(normalized_scores.max())],
        'raw_mean': float(raw_scores.mean()),
        'normalized_mean': float(normalized_scores.mean()),
        'rank_correlation': float(stats.spearmanr(raw_scores, normalized_scores)[0]),
        'linear_correlation': float(correlation)
    }

def main():
    parser = argparse.ArgumentParser(description='Score normalization for OSR')
    parser.add_argument('--scores_file', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration/scores/raw_scores_calibration.json',
                       help='Path to raw scores JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
                       help='Output directory for normalization results')
    parser.add_argument('--n_percentiles', type=int, default=100,
                       help='Number of percentile points for normalization mapping')
    
    args = parser.parse_args()
    
    logger.info("Starting score normalization for OSR")
    logger.info(f"Raw scores file: {args.scores_file}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    scores_output_dir = os.path.join(args.output_dir, 'scores')
    os.makedirs(scores_output_dir, exist_ok=True)
    
    # Load raw scores
    with open(args.scores_file, 'r') as f:
        scores_data = json.load(f)
    
    energy_scores = np.array(scores_data['energy_scores'])
    cosine_scores = np.array(scores_data['cosine_scores'])
    labels = np.array(scores_data['labels'])
    predictions = np.array(scores_data['predictions'])
    
    logger.info(f"Loaded {len(energy_scores)} samples from calibration set")
    logger.info(f"Class distribution - Human: {np.sum(labels == 0)}, False: {np.sum(labels == 1)}")
    logger.info(f"Prediction accuracy: {(predictions == labels).mean():.4f}")
    
    # Initialize normalizer
    normalizer = PercentileNormalizer(n_percentiles=args.n_percentiles)
    
    # Fit normalizers for both scores
    normalizer.fit(energy_scores, predictions, 'energy')  # Use predictions for normalization
    normalizer.fit(cosine_scores, predictions, 'cosine')
    
    # Transform scores
    logger.info("\n🔄 Applying normalization...")
    normalized_energy = normalizer.transform(energy_scores, predictions, 'energy')
    normalized_cosine = normalizer.transform(cosine_scores, predictions, 'cosine')
    
    # Analyze normalization quality
    energy_analysis = analyze_normalization_quality(
        energy_scores, normalized_energy, labels, predictions, 'energy'
    )
    cosine_analysis = analyze_normalization_quality(
        cosine_scores, normalized_cosine, labels, predictions, 'cosine'
    )
    
    # Save normalization mappings (stats.json)
    stats_file = os.path.join(args.output_dir, 'stats.json')
    normalizer.save(stats_file)
    
    # Save normalized scores
    normalized_scores_data = {
        'energy_scores_raw': energy_scores.tolist(),
        'energy_scores_normalized': normalized_energy.tolist(),
        'cosine_scores_raw': cosine_scores.tolist(), 
        'cosine_scores_normalized': normalized_cosine.tolist(),
        'labels': labels.tolist(),
        'predictions': predictions.tolist(),
        'sample_indices': list(range(len(labels))),
        'normalization_analysis': {
            'energy': energy_analysis,
            'cosine': cosine_analysis
        },
        'metadata': {
            'normalization_method': 'per_class_percentile',
            'n_percentiles': args.n_percentiles,
            'num_samples': len(labels),
            'clip_policy': 'clip_to_[0,1]',
            'ready_for_fusion': True
        }
    }
    
    normalized_scores_file = os.path.join(scores_output_dir, 'normalized_scores_calibration.json')
    with open(normalized_scores_file, 'w') as f:
        json.dump(normalized_scores_data, f, indent=2)
    
    # Create summary report
    summary = {
        'normalization_method': 'per_class_percentile_to_[0,1]',
        'energy_score_normalization': {
            'human_class': normalizer.normalizers['energy'][0],
            'false_class': normalizer.normalizers['energy'][1]
        },
        'cosine_score_normalization': {
            'human_class': normalizer.normalizers['cosine'][0], 
            'false_class': normalizer.normalizers['cosine'][1]
        },
        'quality_metrics': {
            'energy_rank_correlation': energy_analysis['rank_correlation'],
            'cosine_rank_correlation': cosine_analysis['rank_correlation'],
            'energy_range_normalized': energy_analysis['normalized_range'],
            'cosine_range_normalized': cosine_analysis['normalized_range']
        },
        'next_phase': 'Phase 6 - Learn fused score weights and threshold'
    }
    
    summary_file = os.path.join(args.output_dir, 'normalization_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 Score normalization completed successfully!")
    logger.info(f"Normalization mappings saved to: {stats_file}")
    logger.info(f"Normalized scores saved to: {normalized_scores_file}")
    logger.info(f"Summary report saved to: {summary_file}")
    logger.info("\n📊 Normalization Summary:")
    logger.info(f"  Energy scores: {energy_analysis['raw_range']} → [0,1]")
    logger.info(f"  Cosine scores: {cosine_analysis['raw_range']} → [0,1]")
    logger.info(f"  Rank preservation: Energy {energy_analysis['rank_correlation']:.3f}, Cosine {cosine_analysis['rank_correlation']:.3f}")
    logger.info(f"✅ Ready for Phase 6: Learn fused score weights and threshold")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
