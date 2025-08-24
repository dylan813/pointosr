#!/usr/bin/env python3
"""
OSR Evaluation Module

This module performs comprehensive Open Set Recognition evaluation using:
1. Trained PointNeXt-S model
2. Optimized fusion configuration (weights + threshold)
3. Score normalization mappings
4. Class-conditional prototypes

Evaluation pipeline:
1. Load trained model and OSR configurations
2. Extract raw scores (Energy + Cosine) from test dataset
3. Apply normalization and fusion
4. Compute OSR metrics (AUROC, FPR@TPR, Coverage)
5. Generate detailed evaluation report
"""

import os
import sys
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

from pointnext.utils import load_checkpoint, setup_logger_dist, set_random_seed
from pointnext.dataset import build_dataloader_from_cfg
from pointnext.model import build_model_from_cfg

# Setup logging
logger = logging.getLogger(__name__)

class OSREvaluator:
    """Complete OSR evaluation pipeline."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f'cuda:{cfg.rank}' if torch.cuda.is_available() else 'cpu')
        
        # Load OSR configurations
        self.load_osr_configs()
        
        # Initialize model
        self.model = None
        self.normalizer = None
        
    def load_osr_configs(self):
        """Load all OSR configuration files."""
        osr_eval_cfg = self.cfg.osr_eval
        
        # Load fusion config
        with open(osr_eval_cfg.fusion_config_path, 'r') as f:
            self.fusion_config = json.load(f)
        logger.info(f"Loaded fusion config: T={self.fusion_config['T']:.4f}, "
                   f"weights=[{self.fusion_config['fused_weights'][0]:.3f}, {self.fusion_config['fused_weights'][1]:.3f}], "
                   f"threshold={self.fusion_config['fused_threshold']:.6f}")
        
        # Load normalization stats
        with open(osr_eval_cfg.stats_path, 'r') as f:
            self.normalization_stats = json.load(f)
        logger.info("Loaded score normalization mappings")
        
        # Load prototypes
        human_prototypes_path = os.path.join(osr_eval_cfg.prototypes_path, 'human_k6', 'prototypes.npy')
        fp_prototypes_path = os.path.join(osr_eval_cfg.prototypes_path, 'fp_k4', 'prototypes.npy')
        
        self.human_prototypes = torch.from_numpy(np.load(human_prototypes_path)).float()
        self.fp_prototypes = torch.from_numpy(np.load(fp_prototypes_path)).float()
        
        logger.info(f"Loaded prototypes: Human {self.human_prototypes.shape}, FP {self.fp_prototypes.shape}")
        
        # Create output directory
        os.makedirs(osr_eval_cfg.results_dir, exist_ok=True)
        
    def load_model(self):
        """Load and prepare the trained model for evaluation."""
        if not self.cfg.model.get('criterion_args', False):
            self.cfg.model.criterion_args = self.cfg.criterion_args
        
        self.model = build_model_from_cfg(self.cfg.model).to(self.device)
        
        if self.cfg.pretrained_path is None:
            raise ValueError("pretrained_path must be specified for OSR evaluation")
        
        # Load checkpoint
        epoch, best_val = load_checkpoint(self.model, self.cfg.pretrained_path)
        logger.info(f"Loaded model from {self.cfg.pretrained_path} (epoch {epoch}, best_val {best_val})")
        
        self.model.eval()
        
        # Override forward method to capture embeddings
        self._setup_embedding_extraction()
        
    def _setup_embedding_extraction(self):
        """Setup embedding extraction from penultimate layer."""
        self.current_embeddings = None
        
        def embedding_hook(module, input, output):
            self.current_embeddings = output.detach()
        
        # Register hook on encoder
        if hasattr(self.model, 'encoder'):
            self.model.encoder.register_forward_hook(embedding_hook)
        else:
            logger.warning("Model doesn't have 'encoder' attribute, trying alternative hook registration")
            
        # Override forward to ensure embeddings are captured
        original_forward = self.model.forward
        
        def enhanced_forward(data):
            if hasattr(self.model, 'encoder'):
                global_feat = self.model.encoder.forward_cls_feat(data)
                self.current_embeddings = global_feat
                logits = self.model.prediction(global_feat)
                return logits
            else:
                return original_forward(data)
        
        self.model.forward = enhanced_forward
        
    def extract_scores(self, dataloader):
        """Extract Energy and Cosine scores from the test dataset."""
        logger.info(f"Extracting scores from {len(dataloader.dataset)} test samples")
        
        all_energy_scores = []
        all_cosine_scores = []
        all_labels = []
        all_predictions = []
        all_logits = []
        
        # Move prototypes to device
        human_prototypes = self.human_prototypes.to(self.device)
        fp_prototypes = self.fp_prototypes.to(self.device)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(dataloader, desc="Extracting scores")):
                # Move data to device
                for key in data.keys():
                    data[key] = data[key].to(self.device, non_blocking=True)
                
                target = data['y']
                points = data['x']
                
                # Prepare input
                npoints = self.cfg.num_points
                if points.shape[1] > npoints:
                    points = points[:, :npoints]
                    
                data['pos'] = points[:, :, :3].contiguous()
                # Use encoder_args.in_channels or default to 4
                in_channels = self.cfg.model.encoder_args.get('in_channels', 4)
                data['x'] = points[:, :, :in_channels].transpose(1, 2).contiguous()
                
                # Forward pass
                logits = self.model(data)
                embeddings = self.current_embeddings
                
                if embeddings is None:
                    raise RuntimeError("Failed to extract embeddings. Check hook registration.")
                
                # Compute Energy scores (temperature-scaled)
                T = self.fusion_config['T']
                energy_scores = T * torch.logsumexp(logits / T, dim=1)
                
                # Compute predictions
                predictions = logits.argmax(dim=1)
                
                # Compute Cosine similarity scores
                cosine_scores = []
                for i in range(len(embeddings)):
                    emb = embeddings[i]  # (embedding_dim,)
                    pred_class = predictions[i].item()
                    
                    if pred_class == 0:  # Human class
                        prototypes = human_prototypes
                    else:  # False class
                        prototypes = fp_prototypes
                    
                    # Compute cosine similarity to all prototypes of predicted class
                    emb_norm = emb / (emb.norm() + 1e-8)
                    proto_norm = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)
                    cosine_sims = torch.mm(emb_norm.unsqueeze(0), proto_norm.t()).squeeze()
                    
                    # Take max cosine similarity
                    max_cosine = cosine_sims.max()
                    cosine_scores.append(max_cosine.item())
                
                # Store results
                all_energy_scores.extend(energy_scores.cpu().numpy())
                all_cosine_scores.extend(cosine_scores)
                all_labels.extend(target.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        return {
            'energy_scores': np.array(all_energy_scores),
            'cosine_scores': np.array(all_cosine_scores),
            'labels': np.array(all_labels),
            'predictions': np.array(all_predictions),
            'logits': np.array(all_logits)
        }
    
    def normalize_scores(self, energy_scores, cosine_scores, predictions):
        """Apply per-class percentile normalization to scores."""
        logger.info("Applying score normalization")
        
        normalized_energy = np.zeros_like(energy_scores)
        normalized_cosine = np.zeros_like(cosine_scores)
        
        for class_idx in [0, 1]:  # Human, False
            if str(class_idx) not in self.normalization_stats['energy']:
                continue
                
            class_mask = predictions == class_idx
            if not np.any(class_mask):
                continue
            
            # Normalize Energy scores
            energy_normalizer = self.normalization_stats['energy'][str(class_idx)]
            percentile_values = np.array(energy_normalizer['percentile_values'])
            percentiles = np.array(energy_normalizer['percentiles'])
            
            class_energy = energy_scores[class_mask]
            normalized_percentiles = np.interp(class_energy, percentile_values, percentiles)
            normalized_energy[class_mask] = np.clip(normalized_percentiles / 100.0, 0.0, 1.0)
            
            # Normalize Cosine scores
            cosine_normalizer = self.normalization_stats['cosine'][str(class_idx)]
            percentile_values = np.array(cosine_normalizer['percentile_values'])
            percentiles = np.array(cosine_normalizer['percentiles'])
            
            class_cosine = cosine_scores[class_mask]
            normalized_percentiles = np.interp(class_cosine, percentile_values, percentiles)
            normalized_cosine[class_mask] = np.clip(normalized_percentiles / 100.0, 0.0, 1.0)
        
        return normalized_energy, normalized_cosine
    
    def compute_fused_scores(self, energy_norm, cosine_norm):
        """Compute fused scores using optimized weights."""
        w1, w2 = self.fusion_config['fused_weights']
        fused_scores = w1 * energy_norm + w2 * cosine_norm
        return fused_scores
    
    def evaluate_osr_performance(self, fused_scores, labels):
        """Compute comprehensive OSR metrics."""
        logger.info("Computing OSR performance metrics")
        
        threshold = self.fusion_config['fused_threshold']
        target_tpr = self.fusion_config['target_tpr']
        
        # ID = Human (label 0), OOD = False (label 1)
        id_mask = labels == 0
        ood_mask = labels == 1
        
        id_scores = fused_scores[id_mask]
        ood_scores = fused_scores[ood_mask]
        
        # Compute metrics at optimized threshold
        id_accepted = np.sum(id_scores >= threshold)
        ood_rejected = np.sum(ood_scores < threshold)
        
        tpr = id_accepted / len(id_scores) if len(id_scores) > 0 else 0.0
        tnr = ood_rejected / len(ood_scores) if len(ood_scores) > 0 else 0.0
        fpr = 1.0 - tnr
        
        # Overall accuracy
        all_accepted = np.sum(fused_scores >= threshold)
        coverage = all_accepted / len(fused_scores)
        
        # AUROC
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except:
            auroc = 0.5
        
        # ROC curve
        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_scores)
        
        # Find FPR at different TPR levels
        target_tprs = [0.95, 0.99, 0.995]
        fpr_at_tprs = {}
        for target in target_tprs:
            idx = np.where(tpr_curve >= target)[0]
            if len(idx) > 0:
                fpr_at_tprs[f'FPR@TPR{target:.1%}'] = fpr_curve[idx[0]]
            else:
                fpr_at_tprs[f'FPR@TPR{target:.1%}'] = 1.0
        
        metrics = {
            'threshold_used': float(threshold),
            'target_tpr': float(target_tpr),
            'achieved_tpr': float(tpr),
            'achieved_fpr': float(fpr),
            'achieved_tnr': float(tnr),
            'auroc': float(auroc),
            'coverage': float(coverage),
            'n_id_samples': int(len(id_scores)),
            'n_ood_samples': int(len(ood_scores)),
            'id_accepted': int(id_accepted),
            'ood_rejected': int(ood_rejected),
            'fpr_at_tprs': {k: float(v) for k, v in fpr_at_tprs.items()},
            'roc_curve': {
                'fpr': fpr_curve.tolist(),
                'tpr': tpr_curve.tolist(),
                'thresholds': thresholds.tolist()
            }
        }
        
        return metrics
    
    def save_results(self, scores_data, metrics):
        """Save evaluation results and generate report."""
        results_dir = self.cfg.osr_eval.results_dir
        
        # Save detailed scores
        scores_file = os.path.join(results_dir, 'test_scores_detailed.json')
        with open(scores_file, 'w') as f:
            json.dump({
                'energy_scores_raw': scores_data['energy_scores'].tolist(),
                'cosine_scores_raw': scores_data['cosine_scores'].tolist(),
                'energy_scores_normalized': scores_data['energy_norm'].tolist(),
                'cosine_scores_normalized': scores_data['cosine_norm'].tolist(),
                'fused_scores': scores_data['fused_scores'].tolist(),
                'labels': scores_data['labels'].tolist(),
                'predictions': scores_data['predictions'].tolist(),
                'evaluation_metadata': {
                    'num_samples': len(scores_data['labels']),
                    'model_path': self.cfg.pretrained_path,
                    'eval_dataset': self.cfg.osr_eval.eval_dataset
                }
            }, f, indent=2)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, 'osr_evaluation_results.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate summary report
        summary = {
            'model_path': self.cfg.pretrained_path,
            'eval_dataset': self.cfg.osr_eval.eval_dataset,
            'num_test_samples': len(scores_data['labels']),
            'fusion_configuration': {
                'weights': self.fusion_config['fused_weights'],
                'threshold': self.fusion_config['fused_threshold'],
                'temperature': self.fusion_config['T']
            },
            'performance_summary': {
                'AUROC': metrics['auroc'],
                'TPR (ID acceptance)': metrics['achieved_tpr'],
                'FPR (OOD acceptance)': metrics['achieved_fpr'],
                'Coverage': metrics['coverage'],
                'Target TPR': metrics['target_tpr'],
                'Meets Target': metrics['achieved_tpr'] >= metrics['target_tpr']
            },
            'detailed_results_files': {
                'scores': scores_file,
                'metrics': metrics_file
            }
        }
        
        summary_file = os.path.join(results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to: {results_dir}")
        return summary_file, metrics_file, scores_file

def main(gpu, cfg, profile=False):
    """Main OSR evaluation function."""
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    
    # Setup logging
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    
    logger.info("="*60)
    logger.info("🚀 Starting OSR Evaluation")
    logger.info("="*60)
    logger.info(f"Evaluation dataset: {cfg.osr_eval.eval_dataset}")
    logger.info(f"Model: {cfg.pretrained_path}")
    
    # Initialize evaluator
    evaluator = OSREvaluator(cfg)
    
    # Load model
    evaluator.load_model()
    
    # Build evaluation dataloader
    eval_loader = build_dataloader_from_cfg(
        cfg.get('val_batch_size', cfg.batch_size),
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split=cfg.osr_eval.eval_dataset,  # Use eval dataset
        distributed=cfg.distributed
    )
    
    logger.info(f"Loaded evaluation dataset: {len(eval_loader.dataset)} samples")
    
    # Extract raw scores
    scores_data = evaluator.extract_scores(eval_loader)
    logger.info(f"Extracted scores from {len(scores_data['labels'])} samples")
    
    # Apply normalization
    energy_norm, cosine_norm = evaluator.normalize_scores(
        scores_data['energy_scores'], 
        scores_data['cosine_scores'], 
        scores_data['predictions']
    )
    
    # Compute fused scores
    fused_scores = evaluator.compute_fused_scores(energy_norm, cosine_norm)
    
    # Add normalized scores to data
    scores_data.update({
        'energy_norm': energy_norm,
        'cosine_norm': cosine_norm,
        'fused_scores': fused_scores
    })
    
    # Evaluate OSR performance
    metrics = evaluator.evaluate_osr_performance(fused_scores, scores_data['labels'])
    
    # Save results
    summary_file, metrics_file, scores_file = evaluator.save_results(scores_data, metrics)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("🎉 OSR EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"📊 Performance Metrics:")
    logger.info(f"  AUROC: {metrics['auroc']:.4f}")
    logger.info(f"  TPR (ID): {metrics['achieved_tpr']:.4f} (target: {metrics['target_tpr']:.4f})")
    logger.info(f"  FPR (OOD): {metrics['achieved_fpr']:.4f}")
    logger.info(f"  Coverage: {metrics['coverage']:.4f}")
    logger.info(f"  Target met: {'✅' if metrics['achieved_tpr'] >= metrics['target_tpr'] else '❌'}")
    
    logger.info(f"\n📈 FPR at different TPR levels:")
    for key, value in metrics['fpr_at_tprs'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"\n📁 Results saved:")
    logger.info(f"  Summary: {summary_file}")
    logger.info(f"  Detailed metrics: {metrics_file}")
    logger.info(f"  Scores: {scores_file}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ OSR Evaluation completed successfully!")
    logger.info("="*60)
    
    if cfg.distributed:
        dist.destroy_process_group()
    
    return True
