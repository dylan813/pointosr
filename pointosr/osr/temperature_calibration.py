#!/usr/bin/env python3
"""
Phase 3: Temperature Calibration for OSR

This script:
1. Loads the calibration set (ID samples: Human + False Positive)
2. Extracts logits from the trained model
3. Finds optimal temperature T to minimize NLL (Negative Log-Likelihood) on ID data
4. Saves temperature to fusion_config.json for OSR system
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import json
from tqdm import tqdm
import logging
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

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
    """
    Temperature scaling for model calibration.
    
    Fits a single scalar parameter T that is applied to all logits:
    p_i = softmax(z_i / T)
    
    where z_i are the original logits and T > 0 is the temperature.
    """
    
    def __init__(self):
        self.temperature = 1.0
        
    def set_temperature(self, logits, labels):
        """
        Find optimal temperature using validation set.
        
        Args:
            logits: (N, C) tensor of logits
            labels: (N,) tensor of true labels
            
        Returns:
            optimal_temperature: float
        """
        logger.info("Finding optimal temperature via cross-validation...")
        
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
    
    def apply_temperature(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def get_calibrated_probabilities(self, logits):
        """Get temperature-calibrated probabilities."""
        scaled_logits = self.apply_temperature(logits)
        return F.softmax(scaled_logits, dim=1)

def load_model(model_path, config_path, device='cuda'):
    """Load the trained model."""
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
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model

def extract_logits(model, dataset, batch_size=32, device='cuda'):
    """Extract logits from model on dataset."""
    logger.info(f"Extracting logits from {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting logits"):
            # Move batch to device
            pos = batch['pos'].to(device)
            x = batch['x'].to(device)
            labels = batch['y']
            
            # Prepare input for model (transpose for conv1d: batch, features, points)
            data = {'pos': pos, 'x': x.transpose(1, 2).contiguous()}
            
            # Forward pass
            logits = model(data)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    # Concatenate all results
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    logger.info(f"Extracted logits shape: {logits.shape}, labels shape: {labels.shape}")
    return logits, labels

def load_calibration_data(data_dir, calibration_metadata_path):
    """Load calibration dataset from metadata."""
    logger.info("Loading calibration data...")
    
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
    
    logger.info(f"Calibration dataset loaded: {len(cal_dataset)} samples")
    logger.info(f"ID Human samples: {len(cal_metadata['cal_id_human'])}")
    logger.info(f"ID False-positive samples: {len(cal_metadata['cal_id_fp'])}")
    logger.info(f"OOD samples: {len(cal_metadata['cal_ood'])}")
    
    return cal_dataset, cal_metadata

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
    
    confidences = torch.max(probs, dim=1)[0]
    predictions = torch.argmax(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Select examples in this bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def main():
    parser = argparse.ArgumentParser(description='Temperature calibration for OSR')
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
    parser.add_argument('--output_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
                       help='Output directory for calibration results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    logger.info("Starting temperature calibration for OSR")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.config_path, args.device)
    
    # Load calibration data
    cal_dataset, cal_metadata = load_calibration_data(args.data_dir, args.calibration_metadata)
    
    # Extract logits on calibration set
    logits, labels = extract_logits(model, cal_dataset, args.batch_size, args.device)
    
    # Compute metrics before calibration
    logger.info("\n📊 Metrics before temperature calibration:")
    metrics_before = compute_calibration_metrics(logits, labels, temperature=1.0)
    for key, value in metrics_before.items():
        logger.info(f"   {key}: {value:.4f}")
    
    # Perform temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.set_temperature(logits, labels)
    
    # Compute metrics after calibration
    logger.info("\n📊 Metrics after temperature calibration:")
    metrics_after = compute_calibration_metrics(logits, labels, temperature=optimal_temp)
    for key, value in metrics_after.items():
        logger.info(f"   {key}: {value:.4f}")
    
    # Compute improvement
    logger.info("\n📈 Calibration improvement:")
    logger.info(f"   NLL reduction: {metrics_before['nll'] - metrics_after['nll']:.4f}")
    logger.info(f"   ECE reduction: {metrics_before['ece'] - metrics_after['ece']:.4f}")
    
    # Save temperature to fusion config
    fusion_config = {
        "T": optimal_temp,
        "K_H": 6,
        "K_F": 4,
        "fused_weights": [0.5, 0.5],
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
    
    fusion_config_path = os.path.join(args.output_dir, 'fusion_config.json')
    with open(fusion_config_path, 'w') as f:
        json.dump(fusion_config, f, indent=2)
    
    # Save detailed calibration results
    calibration_results = {
        "optimal_temperature": optimal_temp,
        "calibration_dataset_size": len(cal_dataset),
        "id_human_samples": len(cal_metadata['cal_id_human']),
        "id_fp_samples": len(cal_metadata['cal_id_fp']),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "improvement": {
            "nll_reduction": metrics_before['nll'] - metrics_after['nll'],
            "ece_reduction": metrics_before['ece'] - metrics_after['ece']
        }
    }
    
    results_path = os.path.join(args.output_dir, 'temperature_calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("🎉 Temperature calibration completed successfully!")
    logger.info(f"Optimal temperature: T = {optimal_temp:.4f}")
    logger.info(f"Fusion config saved to: {fusion_config_path}")
    logger.info(f"Detailed results saved to: {results_path}")
    logger.info(f"✅ Ready for Phase 4: Define raw signals (Energy + Cosine)")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()
