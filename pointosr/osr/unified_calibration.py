#!/usr/bin/env python3
"""
Unified OSR Calibration Pipeline

This script combines all OSR calibration steps into a single process:
1. Temperature Calibration - Find optimal temperature T
2. Raw Score Computation - Compute energy and cosine scores  
3. Score Normalization - Normalize scores to [0,1] range
4. Fusion Optimization - Find optimal fusion weights and threshold

Usage:
    python unified_calibration.py --model_path /path/to/model.pth --config_path /path/to/config.yaml
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import logging
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt

# Add the pointosr package to Python path
pointosr_path = '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr'
sys.path.insert(0, pointosr_path)

from pointnext.utils import EasyConfig
from pointnext.dataset.human.human import HumanDataset
from pointnext.model import build_model_from_cfg

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemperatureScaling:
    """Temperature scaling for model calibration."""
    
    def __init__(self):
        self.temperature = 1.0
        
    def set_temperature(self, logits, labels):
        """Find optimal temperature using validation set."""
        logger.info("🔧 Finding optimal temperature via cross-validation...")
        
        # Convert to numpy for optimization
        logits_np = logits.cpu().numpy() if torch.is_tensor(logits) else logits
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        def temperature_nll(T):
            """Negative log-likelihood as a function of temperature."""
            if T <= 0:
                return np.inf
            
            # Apply temperature scaling
            scaled_logits = logits_np / T
            
            # Convert to probabilities
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Compute negative log-likelihood
            nll = -np.mean(np.log(probs[np.arange(len(labels_np)), labels_np] + 1e-12))
            return nll
        
        # Optimize temperature
        result = minimize_scalar(temperature_nll, bounds=(0.1, 10.0), method='bounded')
        
        if result.success:
            self.temperature = float(result.x)
            logger.info(f"✅ Optimal temperature found: T = {self.temperature:.4f}")
            logger.info(f"   Final NLL: {result.fun:.4f}")
        else:
            logger.warning("Temperature optimization failed, using T = 1.0")
            self.temperature = 1.0
            
        return self.temperature

class EnergyScorer:
    """Compute temperature-scaled energy scores."""
    
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        
    def compute_energy(self, logits):
        """Compute energy score from logits."""
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Compute energy (logsumexp)
        energy = self.temperature * torch.logsumexp(scaled_logits, dim=1)
        
        return energy

class CosineScorer:
    """Compute class-conditional cosine similarity to prototypes."""
    
    def __init__(self, human_prototypes, fp_prototypes):
        self.human_prototypes = human_prototypes  # (K_H, D)
        self.fp_prototypes = fp_prototypes        # (K_F, D)
        
    def compute_cosine(self, embeddings, predictions):
        """Compute cosine similarity to prototypes based on predicted class."""
        batch_size = embeddings.shape[0]
        cosine_scores = torch.zeros(batch_size, device=embeddings.device)
        
        for i in range(batch_size):
            pred_class = predictions[i].item()
            embedding = embeddings[i].unsqueeze(0)  # (1, D)
            
            if pred_class == 0:  # Human
                # Compute cosine to human prototypes
                similarities = F.cosine_similarity(embedding, torch.from_numpy(self.human_prototypes).to(embedding.device), dim=1)
                cosine_scores[i] = torch.max(similarities)
            else:  # False Positive
                # Compute cosine to FP prototypes
                similarities = F.cosine_similarity(embedding, torch.from_numpy(self.fp_prototypes).to(embedding.device), dim=1)
                cosine_scores[i] = torch.max(similarities)
        
        return cosine_scores

class PercentileNormalizer:
    """Per-class percentile normalization for scores."""
    
    def __init__(self, n_percentiles=100):
        self.n_percentiles = n_percentiles
        self.mappings = {}
        
    def fit(self, scores, predictions, score_name):
        """Fit normalization mapping for each class."""
        logger.info(f"📊 Fitting {score_name} normalization...")
        
        for class_id in [0, 1]:  # Human, FP
            class_mask = predictions == class_id
            if np.sum(class_mask) == 0:
                continue
                
            class_scores = scores[class_mask]
            
            # Build empirical CDF
            percentiles = np.linspace(0, 100, self.n_percentiles)
            percentile_values = np.percentile(class_scores, percentiles)
            
            # Store mapping
            key = f"{score_name}_class_{class_id}"
            self.mappings[key] = {
                'percentiles': percentiles,
                'values': percentile_values,
                'min_score': np.min(class_scores),
                'max_score': np.max(class_scores)
            }
            
            logger.info(f"   Class {class_id}: range [{np.min(class_scores):.3f}, {np.max(class_scores):.3f}]")
    
    def transform(self, scores, predictions, score_name):
        """Transform scores using fitted mappings."""
        normalized_scores = np.zeros_like(scores)
        
        for class_id in [0, 1]:
            class_mask = predictions == class_id
            if np.sum(class_mask) == 0:
                continue
                
            key = f"{score_name}_class_{class_id}"
            if key not in self.mappings:
                continue
                
            mapping = self.mappings[key]
            class_scores = scores[class_mask]
            
            # Apply percentile transformation
            normalized_class_scores = np.interp(
                class_scores, 
                mapping['values'], 
                mapping['percentiles'] / 100.0
            )
            
            # Clip to [0, 1]
            normalized_class_scores = np.clip(normalized_class_scores, 0, 1)
            normalized_scores[class_mask] = normalized_class_scores
        
        return normalized_scores
    
    def save(self, filepath):
        """Save normalization mappings."""
        # Convert numpy arrays to lists for JSON serialization
        json_mappings = {}
        for key, mapping in self.mappings.items():
            json_mappings[key] = {
                'percentiles': mapping['percentiles'].tolist(),
                'values': mapping['values'].tolist(),
                'min_score': float(mapping['min_score']),
                'max_score': float(mapping['max_score'])
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_mappings, f, indent=2)

class FusionOptimizer:
    """Optimize fusion weights and threshold."""
    
    def __init__(self, target_tpr=0.99, weight_step=0.05):
        self.target_tpr = target_tpr
        self.weight_step = weight_step
        
    def compute_fused_scores(self, energy_norm, cosine_norm, w1):
        """Compute fused scores with given weight."""
        w2 = 1.0 - w1
        fused_scores = w1 * energy_norm + w2 * cosine_norm
        return fused_scores
    
    def find_optimal_threshold(self, fused_scores, labels, target_tpr=None):
        """Find optimal threshold for given TPR target."""
        if target_tpr is None:
            target_tpr = self.target_tpr
            
        # Sort scores and find threshold
        sorted_indices = np.argsort(fused_scores)[::-1]  # Higher is better
        sorted_scores = fused_scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Find threshold that achieves target TPR
        id_mask = sorted_labels == 0  # Human samples
        if np.sum(id_mask) == 0:
            return 0.5, 0.0, 0.0
            
        # Count how many ID samples we need to accept
        n_id_samples = np.sum(id_mask)
        n_accept = int(target_tpr * n_id_samples)
        
        if n_accept >= len(sorted_scores):
            threshold = sorted_scores[-1] - 1e-6
        else:
            threshold = sorted_scores[n_accept]
        
        # Compute metrics
        predictions = fused_scores >= threshold
        tpr = np.sum((predictions == 1) & (labels == 0)) / np.sum(labels == 0)
        fpr = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
        
        return threshold, tpr, fpr
    
    def optimize(self, energy_norm, cosine_norm, labels):
        """Run grid search for optimal fusion weights."""
        logger.info("🔍 Optimizing fusion weights and threshold...")
        
        best_result = None
        best_fpr = float('inf')
        all_results = []
        
        # Grid search over weights
        for w1 in np.arange(0, 1.01, self.weight_step):
            w2 = 1.0 - w1
            
            # Compute fused scores
            fused_scores = self.compute_fused_scores(energy_norm, cosine_norm, w1)
            
            # Find optimal threshold
            threshold, tpr, fpr = self.find_optimal_threshold(fused_scores, labels)
            
            # Compute AUROC
            try:
                auroc = roc_auc_score(labels, fused_scores)
            except:
                auroc = 0.5
            
            result = {
                'w1': w1,
                'w2': w2,
                'threshold': threshold,
                'tpr': tpr,
                'fpr': fpr,
                'auroc': auroc
            }
            all_results.append(result)
            
            # Check if this is the best result
            if tpr >= self.target_tpr and fpr < best_fpr:
                best_result = result
                best_fpr = fpr
        
        if best_result is None:
            logger.warning("No configuration achieved target TPR, using default")
            best_result = {
                'w1': 0.5, 'w2': 0.5, 'threshold': 0.5,
                'tpr': 0.0, 'fpr': 0.0, 'auroc': 0.5
            }
        
        logger.info(f"✅ Best configuration:")
        logger.info(f"   Weights: [{best_result['w1']:.2f}, {best_result['w2']:.2f}]")
        logger.info(f"   Threshold: {best_result['threshold']:.6f}")
        logger.info(f"   TPR: {best_result['tpr']:.3f}")
        logger.info(f"   FPR: {best_result['fpr']:.3f}")
        logger.info(f"   AUROC: {best_result['auroc']:.3f}")
        
        return best_result, all_results

def load_model(model_path, config_path, device='cuda'):
    """Load the trained model."""
    logger.info(f"📥 Loading model from {model_path}")
    
    # Load configuration
    cfg = EasyConfig()
    cfg.load(config_path, recursive=True)
    
    # Build model
    model = build_model_from_cfg(cfg.model).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("✅ Model loaded successfully")
    
    return model, cfg

def load_calibration_data(data_dir, calibration_metadata_path):
    """Load calibration dataset from metadata."""
    logger.info("📂 Loading calibration data...")
    
    # Load calibration metadata
    with open(calibration_metadata_path, 'r') as f:
        cal_metadata = json.load(f)
    
    # Load calibration dataset
    cal_dataset = HumanDataset(
        data_dir=data_dir,
        split='cal',
        num_points=2048,
        transform=None,  # No augmentation for calibration
        uniform_sample=False
    )
    
    logger.info(f"✅ Calibration dataset loaded: {len(cal_dataset)} samples")
    logger.info(f"   ID Human samples: {len(cal_metadata['cal_id_human'])}")
    logger.info(f"   ID False-positive samples: {len(cal_metadata['cal_id_fp'])}")
    logger.info(f"   OOD samples: {len(cal_metadata['cal_ood'])}")
    
    return cal_dataset, cal_metadata

def extract_logits_and_embeddings(model, dataset, batch_size=32, device='cuda'):
    """Extract logits and embeddings from model."""
    logger.info("🔄 Extracting logits and embeddings...")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_logits = []
    all_embeddings = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Handle new data structure: 'pos' and 'x' instead of 'point'
            if 'pos' in batch and 'x' in batch:
                # New data structure: pass as dictionary with pos and x
                # Features need to be transposed from (B, N, F) to (B, F, N) for the model
                points = {
                    'pos': batch['pos'].to(device),  # (B, N, 3)
                    'x': batch['x'].transpose(1, 2).to(device)  # (B, F, N) - transposed
                }
            elif 'point' in batch:
                # Old data structure
                points = batch['point'].to(device)
            else:
                raise KeyError(f"Expected 'pos'/'x' or 'point' in batch, got: {batch.keys()}")
            
            # Handle label key
            if 'y' in batch:
                labels = batch['y'].to(device)
            elif 'label' in batch:
                labels = batch['label'].to(device)
            else:
                raise KeyError(f"Expected 'y' or 'label' in batch, got: {batch.keys()}")
            
            # Forward pass - extract both logits and embeddings
            # Get embeddings from encoder before classification head
            embeddings = model.encoder.forward_cls_feat(points)  # (B, C) - global features
            # Get logits through the full model
            logits = model(points)  # (B, num_classes)
            
            all_logits.append(logits.cpu())
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    logger.info(f"✅ Extracted features: {logits.shape}, {embeddings.shape}, {labels.shape}")
    return logits, embeddings, labels

def load_prototypes(prototypes_path):
    """Load pre-computed prototypes."""
    logger.info("📂 Loading prototypes...")
    
    human_prototypes_path = os.path.join(prototypes_path, 'human_k6', 'prototypes.npy')
    fp_prototypes_path = os.path.join(prototypes_path, 'fp_k4', 'prototypes.npy')
    
    human_prototypes = np.load(human_prototypes_path)
    fp_prototypes = np.load(fp_prototypes_path)
    
    logger.info(f"✅ Loaded prototypes:")
    logger.info(f"   Human: {human_prototypes.shape}")
    logger.info(f"   FP: {fp_prototypes.shape}")
    
    return human_prototypes, fp_prototypes

def compute_calibration_metrics(logits, labels, temperature=1.0):
    """Compute calibration metrics."""
    # Apply temperature scaling
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    
    # Accuracy
    predictions = torch.argmax(probs, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    
    # Negative log-likelihood
    nll = F.cross_entropy(scaled_logits, labels).item()
    
    # Expected Calibration Error (ECE)
    ece = compute_ece(probs, labels)
    
    return {
        'accuracy': accuracy,
        'nll': nll,
        'ece': ece,
        'temperature': temperature
    }

def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def save_results(fusion_config, calibration_results, output_dir):
    """Save all calibration results."""
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert both configs
    fusion_config = convert_numpy_types(fusion_config)
    calibration_results = convert_numpy_types(calibration_results)
    
    # Save fusion config
    fusion_config_path = os.path.join(output_dir, 'fusion_config.json')
    with open(fusion_config_path, 'w') as f:
        json.dump(fusion_config, f, indent=2)
    
    # Save calibration results
    results_path = os.path.join(output_dir, 'calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    logger.info(f"💾 Results saved to {output_dir}")
    return fusion_config_path, results_path

def main():
    parser = argparse.ArgumentParser(description='Unified OSR Calibration Pipeline')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/cerlab/Documents/data/pointosr/human',
                       help='Path to data directory')
    parser.add_argument('--calibration_metadata', type=str,
                       default='/home/cerlab/Documents/data/pointosr/human/calibration_selection.json',
                       help='Path to calibration metadata JSON')
    parser.add_argument('--prototypes_path', type=str,
                       default='/home/cerlab/Documents/pointosr/pointosr/osr/prototypes',
                       help='Path to prototypes directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
                       help='Output directory for calibration results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--target_tpr', type=float, default=0.99,
                       help='Target True Positive Rate for ID samples')
    parser.add_argument('--weight_step', type=float, default=0.05,
                       help='Grid search step size for weights')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Unified OSR Calibration Pipeline")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Output: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    scores_dir = os.path.join(args.output_dir, 'scores')
    os.makedirs(scores_dir, exist_ok=True)
    
    # Step 1: Load model and data
    model, cfg = load_model(args.model_path, args.config_path, args.device)
    cal_dataset, cal_metadata = load_calibration_data(args.data_dir, args.calibration_metadata)
    
    # Step 2: Extract features
    logits, embeddings, labels = extract_logits_and_embeddings(
        model, cal_dataset, args.batch_size, args.device
    )
    
    # Step 3: Temperature Calibration
    logger.info("\n" + "="*50)
    logger.info("STEP 1: TEMPERATURE CALIBRATION")
    logger.info("="*50)
    
    # Compute metrics before calibration
    metrics_before = compute_calibration_metrics(logits, labels, temperature=1.0)
    logger.info("📊 Metrics before temperature calibration:")
    for key, value in metrics_before.items():
        logger.info(f"   {key}: {value:.4f}")
    
    # Perform temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.set_temperature(logits, labels)
    
    # Compute metrics after calibration
    metrics_after = compute_calibration_metrics(logits, labels, temperature=optimal_temp)
    logger.info("📊 Metrics after temperature calibration:")
    for key, value in metrics_after.items():
        logger.info(f"   {key}: {value:.4f}")
    
    # Step 4: Load prototypes and compute raw scores
    logger.info("\n" + "="*50)
    logger.info("STEP 2: RAW SCORE COMPUTATION")
    logger.info("="*50)
    
    human_prototypes, fp_prototypes = load_prototypes(args.prototypes_path)
    
    # Get predictions for score computation
    scaled_logits = logits / optimal_temp
    probs = F.softmax(scaled_logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    # Compute energy scores
    energy_scorer = EnergyScorer(temperature=optimal_temp)
    energy_scores = energy_scorer.compute_energy(logits)
    
    # Compute cosine scores
    cosine_scorer = CosineScorer(human_prototypes, fp_prototypes)
    cosine_scores = cosine_scorer.compute_cosine(embeddings, predictions)
    
    logger.info(f"✅ Raw scores computed:")
    logger.info(f"   Energy scores: range [{energy_scores.min():.3f}, {energy_scores.max():.3f}]")
    logger.info(f"   Cosine scores: range [{cosine_scores.min():.3f}, {cosine_scores.max():.3f}]")
    
    # Step 5: Score Normalization
    logger.info("\n" + "="*50)
    logger.info("STEP 3: SCORE NORMALIZATION")
    logger.info("="*50)
    
    normalizer = PercentileNormalizer(n_percentiles=100)
    
    # Fit normalizers
    normalizer.fit(energy_scores.numpy(), predictions.numpy(), 'energy')
    normalizer.fit(cosine_scores.numpy(), predictions.numpy(), 'cosine')
    
    # Transform scores
    energy_norm = normalizer.transform(energy_scores.numpy(), predictions.numpy(), 'energy')
    cosine_norm = normalizer.transform(cosine_scores.numpy(), predictions.numpy(), 'cosine')
    
    logger.info(f"✅ Scores normalized:")
    logger.info(f"   Energy normalized: range [{energy_norm.min():.3f}, {energy_norm.max():.3f}]")
    logger.info(f"   Cosine normalized: range [{cosine_norm.min():.3f}, {cosine_norm.max():.3f}]")
    
    # Save normalization stats
    stats_path = os.path.join(args.output_dir, 'stats.json')
    normalizer.save(stats_path)
    
    # Step 6: Fusion Optimization
    logger.info("\n" + "="*50)
    logger.info("STEP 4: FUSION OPTIMIZATION")
    logger.info("="*50)
    
    optimizer = FusionOptimizer(target_tpr=args.target_tpr, weight_step=args.weight_step)
    best_result, all_results = optimizer.optimize(energy_norm, cosine_norm, labels.numpy())
    
    # Step 7: Save Results
    logger.info("\n" + "="*50)
    logger.info("STEP 5: SAVE RESULTS")
    logger.info("="*50)
    
    # Create fusion config
    fusion_config = {
        "T": optimal_temp,
        "K_H": 6,
        "K_F": 4,
        "fused_weights": [best_result['w1'], best_result['w2']],
        "fused_threshold": best_result['threshold'],
        "target_tpr": args.target_tpr,
        "achieved_tpr": best_result['tpr'],
        "achieved_fpr": best_result['fpr'],
        "achieved_auroc": best_result['auroc'],
        "score_directions": {
            "energy": "higher_is_better",
            "cosine": "higher_is_better"
        },
        "calibration_metadata": {
            "model_path": args.model_path,
            "config_path": args.config_path,
            "data_dir": args.data_dir,
            "calibration_samples": len(cal_dataset),
            "metrics_before_calibration": metrics_before,
            "metrics_after_calibration": metrics_after
        }
    }
    
    # Create calibration results
    calibration_results = {
        "temperature_calibration": {
            "optimal_temperature": optimal_temp,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after
        },
        "raw_scores": {
            "energy_scores": energy_scores.numpy().tolist(),
            "cosine_scores": cosine_scores.numpy().tolist(),
            "labels": labels.numpy().tolist(),
            "predictions": predictions.numpy().tolist()
        },
        "normalized_scores": {
            "energy_scores_normalized": energy_norm.tolist(),
            "cosine_scores_normalized": cosine_norm.tolist()
        },
        "fusion_optimization": {
            "best_result": best_result,
            "all_results": all_results
        },
        "metadata": {
            "num_samples": len(cal_dataset),
            "target_tpr": args.target_tpr,
            "weight_step": args.weight_step,
            "calibration_complete": True
        }
    }
    
    # Save results
    fusion_config_path, results_path = save_results(fusion_config, calibration_results, args.output_dir)
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("🎉 UNIFIED CALIBRATION COMPLETE!")
    logger.info("="*50)
    logger.info(f"📁 Results saved to: {args.output_dir}")
    logger.info(f"🔧 Temperature: {optimal_temp:.4f}")
    logger.info(f"⚖️  Fusion weights: [{best_result['w1']:.2f}, {best_result['w2']:.2f}]")
    logger.info(f"🎯 Threshold: {best_result['threshold']:.6f}")
    logger.info(f"📊 TPR: {best_result['tpr']:.3f}, FPR: {best_result['fpr']:.3f}, AUROC: {best_result['auroc']:.3f}")
    logger.info(f"📈 NLL improvement: {metrics_before['nll'] - metrics_after['nll']:.4f}")
    logger.info(f"📈 ECE improvement: {metrics_before['ece'] - metrics_after['ece']:.4f}")

if __name__ == "__main__":
    main()
