#!/usr/bin/env python3
"""
Phase 6: Fusion Optimization

This script learns the optimal fusion weights and threshold for combining
normalized Energy and Cosine scores into a single OSR decision.

Fusion Strategy:
1. Linear fusion: score_fused = w1 * energy_norm + w2 * cosine_norm
2. Constraint: w1 + w2 = 1 (convex combination)
3. Grid search over w1 ∈ [0, 1] with step size (e.g., 0.05)
4. For each (w1, w2), find optimal threshold τ_fused targeting TPR_ID ≥ 99%
5. Select configuration minimizing FPR while maintaining TPR constraint
6. Evaluate with AUROC, FPR@TPR, and coverage metrics

Target: TPR_ID ≥ 99% (correctly accept 99%+ of ID samples)
Optimization: Minimize FPR_OOD (false positive rate on OOD samples)
"""

import os
import sys
import argparse
import numpy as np
import json
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FusionOptimizer:
    """Optimize fusion weights and threshold for OSR."""
    
    def __init__(self, target_tpr=0.99, weight_step=0.05):
        """
        Args:
            target_tpr: Target True Positive Rate for ID samples (default: 0.99)
            weight_step: Grid search step size for weights (default: 0.05)
        """
        self.target_tpr = target_tpr
        self.weight_step = weight_step
        self.results = []
        
    def compute_fused_scores(self, energy_norm, cosine_norm, w1):
        """
        Compute fused scores with given weight.
        
        Args:
            energy_norm: (N,) normalized energy scores
            cosine_norm: (N,) normalized cosine scores 
            w1: Weight for energy score (w2 = 1 - w1)
            
        Returns:
            fused_scores: (N,) fused scores
        """
        w2 = 1.0 - w1
        fused_scores = w1 * energy_norm + w2 * cosine_norm
        return fused_scores
    
    def find_optimal_threshold(self, fused_scores, labels, target_tpr=None):
        """
        Find threshold that achieves target TPR for ID samples.
        
        Args:
            fused_scores: (N,) fused scores (higher = more ID-like)
            labels: (N,) binary labels (0=Human/ID, 1=False/OOD)
            target_tpr: Target TPR (default: self.target_tpr)
            
        Returns:
            dict with threshold, achieved_tpr, fpr, and metrics
        """
        if target_tpr is None:
            target_tpr = self.target_tpr
            
        # ID samples have label=0, OOD samples have label=1
        id_mask = labels == 0
        ood_mask = labels == 1
        
        id_scores = fused_scores[id_mask]
        ood_scores = fused_scores[ood_mask]
        
        if len(id_scores) == 0 or len(ood_scores) == 0:
            logger.warning("Empty ID or OOD samples")
            return None
            
        # Find threshold for target TPR
        # TPR = P(score >= threshold | ID)
        id_scores_sorted = np.sort(id_scores)
        n_id = len(id_scores)
        
        # Find threshold that gives target TPR
        tpr_index = int((1 - target_tpr) * n_id)
        if tpr_index >= n_id:
            threshold = id_scores_sorted[0] - 1e-6  # Accept all
            achieved_tpr = 1.0
        else:
            threshold = id_scores_sorted[tpr_index]
            achieved_tpr = np.sum(id_scores >= threshold) / len(id_scores)
        
        # Compute FPR at this threshold
        # FPR = P(score >= threshold | OOD)
        fpr = np.sum(ood_scores >= threshold) / len(ood_scores)
        
        # Compute other metrics
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except:
            auroc = 0.5
            
        # Coverage (fraction of samples accepted as ID)
        all_accepted = np.sum(fused_scores >= threshold)
        coverage = all_accepted / len(fused_scores)
        
        return {
            'threshold': float(threshold),
            'achieved_tpr': float(achieved_tpr),
            'fpr': float(fpr),
            'auroc': float(auroc),
            'coverage': float(coverage),
            'n_id': len(id_scores),
            'n_ood': len(ood_scores),
            'id_accepted': int(np.sum(id_scores >= threshold)),
            'ood_accepted': int(np.sum(ood_scores >= threshold))
        }
    
    def grid_search_fusion(self, energy_norm, cosine_norm, labels):
        """
        Grid search over fusion weights to find optimal configuration.
        
        Args:
            energy_norm: (N,) normalized energy scores
            cosine_norm: (N,) normalized cosine scores
            labels: (N,) binary labels (0=ID, 1=OOD)
            
        Returns:
            list of results for each weight configuration
        """
        logger.info(f"Starting grid search with target TPR ≥ {self.target_tpr:.1%}")
        logger.info(f"Weight step size: {self.weight_step}")
        
        # Generate weight grid
        w1_values = np.arange(0.0, 1.0 + self.weight_step/2, self.weight_step)
        logger.info(f"Testing {len(w1_values)} weight combinations")
        
        results = []
        
        for w1 in tqdm(w1_values, desc="Grid search"):
            w2 = 1.0 - w1
            
            # Compute fused scores
            fused_scores = self.compute_fused_scores(energy_norm, cosine_norm, w1)
            
            # Find optimal threshold
            result = self.find_optimal_threshold(fused_scores, labels)
            if result is None:
                continue
                
            # Add weight information
            result.update({
                'w1_energy': float(w1),
                'w2_cosine': float(w2),
                'meets_tpr_target': result['achieved_tpr'] >= self.target_tpr
            })
            
            results.append(result)
            
        logger.info(f"Evaluated {len(results)} configurations")
        
        # Sort by performance: first by meeting TPR target, then by FPR (lower better)
        results.sort(key=lambda x: (-x['meets_tpr_target'], x['fpr'], -x['auroc']))
        
        return results
    
    def analyze_results(self, results):
        """Analyze and summarize grid search results."""
        if not results:
            logger.error("No results to analyze")
            return None
            
        logger.info("\n📊 Fusion Optimization Analysis:")
        
        # Count configurations meeting TPR target
        valid_configs = [r for r in results if r['meets_tpr_target']]
        logger.info(f"Configurations meeting TPR ≥ {self.target_tpr:.1%}: {len(valid_configs)}/{len(results)}")
        
        if not valid_configs:
            logger.warning("No configurations meet TPR target! Using best available.")
            best_result = results[0]  # Best by sort criteria
        else:
            best_result = valid_configs[0]  # Best among valid
            
        logger.info(f"\n🏆 OPTIMAL CONFIGURATION:")
        logger.info(f"  Weights: Energy={best_result['w1_energy']:.3f}, Cosine={best_result['w2_cosine']:.3f}")
        logger.info(f"  Threshold: τ_fused = {best_result['threshold']:.6f}")
        logger.info(f"  Achieved TPR: {best_result['achieved_tpr']:.4f} ({'✅' if best_result['meets_tpr_target'] else '❌'})")
        logger.info(f"  FPR: {best_result['fpr']:.4f}")
        logger.info(f"  AUROC: {best_result['auroc']:.4f}")
        logger.info(f"  Coverage: {best_result['coverage']:.4f}")
        logger.info(f"  ID accepted: {best_result['id_accepted']}/{best_result['n_id']}")
        logger.info(f"  OOD rejected: {best_result['n_ood'] - best_result['ood_accepted']}/{best_result['n_ood']}")
        
        # Show top 5 configurations
        logger.info(f"\n📋 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(results[:5]):
            status = "✅" if result['meets_tpr_target'] else "❌"
            logger.info(f"  {i+1}. w1={result['w1_energy']:.2f}, TPR={result['achieved_tpr']:.3f} {status}, "
                       f"FPR={result['fpr']:.3f}, AUROC={result['auroc']:.3f}")
        
        return best_result, results
    
    def save_results(self, best_result, all_results, output_dir):
        """Save optimization results and update fusion config."""
        
        # Update fusion_config.json with optimal parameters
        fusion_config_file = os.path.join(output_dir, 'fusion_config.json')
        
        if os.path.exists(fusion_config_file):
            with open(fusion_config_file, 'r') as f:
                fusion_config = json.load(f)
        else:
            fusion_config = {}
            
        # Update with fusion results
        fusion_config.update({
            'fused_weights': [best_result['w1_energy'], best_result['w2_cosine']],
            'fused_threshold': best_result['threshold'],
            'target_tpr': self.target_tpr,
            'achieved_tpr': best_result['achieved_tpr'],
            'achieved_fpr': best_result['fpr'],
            'achieved_auroc': best_result['auroc'],
            'optimization_status': 'completed_phase_6'
        })
        
        with open(fusion_config_file, 'w') as f:
            json.dump(fusion_config, f, indent=2)
        logger.info(f"Updated fusion config: {fusion_config_file}")
        
        # Save detailed optimization results
        optimization_results = {
            'best_configuration': best_result,
            'all_configurations': all_results,
            'optimization_settings': {
                'target_tpr': self.target_tpr,
                'weight_step': self.weight_step,
                'n_configurations_tested': len(all_results)
            },
            'summary': {
                'optimal_weights': [best_result['w1_energy'], best_result['w2_cosine']],
                'optimal_threshold': best_result['threshold'],
                'performance': {
                    'tpr': best_result['achieved_tpr'],
                    'fpr': best_result['fpr'],
                    'auroc': best_result['auroc'],
                    'coverage': best_result['coverage']
                }
            }
        }
        
        results_file = os.path.join(output_dir, 'fusion_optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        logger.info(f"Detailed results saved: {results_file}")
        
        return fusion_config_file, results_file

def main():
    parser = argparse.ArgumentParser(description='Fusion optimization for OSR')
    parser.add_argument('--scores_file', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration/scores/normalized_scores_calibration.json',
                       help='Path to normalized scores JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
                       help='Output directory for fusion results')
    parser.add_argument('--target_tpr', type=float, default=0.99,
                       help='Target True Positive Rate for ID samples (default: 0.99)')
    parser.add_argument('--weight_step', type=float, default=0.05,
                       help='Grid search step size for weights (default: 0.05)')
    
    args = parser.parse_args()
    
    logger.info("Starting fusion optimization for OSR")
    logger.info(f"Normalized scores file: {args.scores_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target TPR: {args.target_tpr:.1%}")
    logger.info(f"Weight step: {args.weight_step}")
    
    # Load normalized scores
    with open(args.scores_file, 'r') as f:
        scores_data = json.load(f)
    
    energy_norm = np.array(scores_data['energy_scores_normalized'])
    cosine_norm = np.array(scores_data['cosine_scores_normalized'])
    labels = np.array(scores_data['labels'])
    
    logger.info(f"Loaded {len(energy_norm)} samples")
    logger.info(f"Class distribution - ID (Human): {np.sum(labels == 0)}, OOD (False): {np.sum(labels == 1)}")
    
    # Validate score ranges
    logger.info(f"Energy scores - range: [{energy_norm.min():.3f}, {energy_norm.max():.3f}]")
    logger.info(f"Cosine scores - range: [{cosine_norm.min():.3f}, {cosine_norm.max():.3f}]")
    
    # Initialize optimizer
    optimizer = FusionOptimizer(target_tpr=args.target_tpr, weight_step=args.weight_step)
    
    # Run grid search
    results = optimizer.grid_search_fusion(energy_norm, cosine_norm, labels)
    
    # Analyze results
    best_result, all_results = optimizer.analyze_results(results)
    
    if best_result is None:
        logger.error("Optimization failed!")
        return 1
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    fusion_config_file, results_file = optimizer.save_results(best_result, all_results, args.output_dir)
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 Fusion optimization completed successfully!")
    logger.info(f"Optimal fusion: {best_result['w1_energy']:.3f} * Energy + {best_result['w2_cosine']:.3f} * Cosine")
    logger.info(f"Optimal threshold: τ_fused = {best_result['threshold']:.6f}")
    logger.info(f"Performance: TPR={best_result['achieved_tpr']:.4f}, FPR={best_result['fpr']:.4f}, AUROC={best_result['auroc']:.4f}")
    logger.info(f"Updated config: {fusion_config_file}")
    logger.info(f"Detailed results: {results_file}")
    logger.info(f"✅ Ready for Phase 7: Evaluate on test set")
    logger.info(f"{'='*50}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
