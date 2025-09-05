#!/home/cerlab/miniconda3/envs/pointosr/bin/python
"""
OSR (Open Set Recognition) Scorer for Online Classification

This module provides the OSRScorer class that implements energy and cosine scoring
for online out-of-distribution detection in the ROS classification pipeline.

Features:
- Cached loading of OSR configurations (fusion config, normalization stats, prototypes)
- Energy score computation with temperature scaling
- Cosine score computation with class-conditional prototypes
- Score normalization using pre-computed percentile mappings
- Fused score computation and threshold-based OOD detection
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class OSRScorer:
    """
    Online OSR scorer with caching for real-time OOD detection.
    
    This class loads pre-computed OSR configurations and provides efficient
    scoring for the online classification pipeline.
    """
    
    def __init__(self, 
                 fusion_config_path='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration/fusion_config.json',
                 stats_path='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration/stats.json',
                 prototypes_path='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
                 device='cuda'):
        """
        Initialize OSR scorer with cached configurations.
        
        Args:
            fusion_config_path: Path to fusion configuration JSON
            stats_path: Path to score normalization statistics JSON  
            prototypes_path: Path to prototypes directory
            device: Device to use for torch operations
        """
        self.device = device
        self.fusion_config_path = fusion_config_path
        self.stats_path = stats_path
        self.prototypes_path = prototypes_path
        
        # Cache for configurations
        self._fusion_config = None
        self._normalization_stats = None
        self._human_prototypes = None
        self._fp_prototypes = None
        
        # Load and cache all configurations
        self._load_configurations()
        
        logger.info("OSR Scorer initialized with cached configurations")
        logger.info(f"Temperature: {self.temperature:.4f}")
        logger.info(f"Fusion weights: {self.fusion_weights}")
        logger.info(f"OOD threshold: {self.ood_threshold:.6f}")
    
    def _load_configurations(self):
        """Load and cache all OSR configurations."""
        try:
            # Load fusion config
            with open(self.fusion_config_path, 'r') as f:
                self._fusion_config = json.load(f)
            
            # Load normalization stats
            with open(self.stats_path, 'r') as f:
                self._normalization_stats = json.load(f)
            
            # Load prototypes
            human_prototypes_path = os.path.join(self.prototypes_path, 'human_k6', 'prototypes.npy')
            fp_prototypes_path = os.path.join(self.prototypes_path, 'fp_k4', 'prototypes.npy')
            
            self._human_prototypes = np.load(human_prototypes_path)
            self._fp_prototypes = np.load(fp_prototypes_path)
            
            logger.info(f"Loaded OSR configurations:")
            logger.info(f"  Human prototypes: {self._human_prototypes.shape}")
            logger.info(f"  FP prototypes: {self._fp_prototypes.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load OSR configurations: {e}")
            raise
    
    @property
    def temperature(self):
        """Get calibrated temperature."""
        return self._fusion_config['T']
    
    @property
    def fusion_weights(self):
        """Get optimized fusion weights [energy_weight, cosine_weight]."""
        return self._fusion_config['fused_weights']
    
    @property
    def ood_threshold(self):
        """Get optimized OOD threshold."""
        return self._fusion_config['fused_threshold']
    
    @property
    def target_tpr(self):
        """Get target True Positive Rate."""
        return self._fusion_config['target_tpr']
    
    def compute_energy_score(self, logits):
        """
        Compute temperature-scaled energy score.
        
        Args:
            logits: (N, num_classes) tensor of logits
            
        Returns:
            energy_scores: (N,) tensor of energy scores
        """
        # Apply temperature scaling and compute energy
        scaled_logits = logits / self.temperature
        energy_scores = self.temperature * torch.logsumexp(scaled_logits, dim=1)
        return energy_scores
    
    def compute_cosine_score(self, embeddings, predicted_classes):
        """
        Compute class-conditional cosine similarity scores.
        
        Args:
            embeddings: (N, embed_dim) tensor of embeddings
            predicted_classes: (N,) tensor of predicted class indices
            
        Returns:
            cosine_scores: (N,) tensor of cosine scores
        """
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
        predicted_classes_np = predicted_classes.cpu().numpy() if torch.is_tensor(predicted_classes) else predicted_classes
        
        cosine_scores = np.zeros(len(embeddings_np))
        
        for i, (embedding, pred_class) in enumerate(zip(embeddings_np, predicted_classes_np)):
            embedding = embedding.reshape(1, -1)
            
            if pred_class == 0:  # Human class
                # Compute cosine similarity to all human prototypes
                similarities = cosine_similarity(embedding, self._human_prototypes)[0]
                cosine_scores[i] = np.max(similarities)
            else:  # FP class
                # Compute cosine similarity to all FP prototypes
                similarities = cosine_similarity(embedding, self._fp_prototypes)[0]
                cosine_scores[i] = np.max(similarities)
        
        return torch.tensor(cosine_scores, dtype=torch.float32)
    
    def normalize_scores(self, energy_scores, cosine_scores, predicted_classes):
        """
        Apply per-class percentile normalization to scores.
        
        Args:
            energy_scores: (N,) tensor of raw energy scores
            cosine_scores: (N,) tensor of raw cosine scores  
            predicted_classes: (N,) tensor of predicted class indices
            
        Returns:
            normalized_energy: (N,) array of normalized energy scores [0,1]
            normalized_cosine: (N,) array of normalized cosine scores [0,1]
        """
        try:
            energy_np = energy_scores.cpu().numpy() if torch.is_tensor(energy_scores) else energy_scores
            cosine_np = cosine_scores.cpu().numpy() if torch.is_tensor(cosine_scores) else cosine_scores
            predicted_np = predicted_classes.cpu().numpy() if torch.is_tensor(predicted_classes) else predicted_classes
            
            logger.debug(f"Normalize scores - energy shape: {energy_np.shape}, cosine shape: {cosine_np.shape}, predictions shape: {predicted_np.shape}")
            logger.debug(f"Available keys in normalization stats: {list(self._normalization_stats.keys())}")
            
            normalized_energy = np.zeros_like(energy_np)
            normalized_cosine = np.zeros_like(cosine_np)
            
            for class_idx in [0, 1]:  # Human, FP
                class_mask = predicted_np == class_idx
                if not np.any(class_mask):
                    logger.debug(f"No samples for class {class_idx}")
                    continue
                
                logger.debug(f"Processing class {class_idx} with {np.sum(class_mask)} samples")
                
                # Normalize Energy scores
                energy_key = f"energy_class_{class_idx}"
                if energy_key in self._normalization_stats:
                    logger.debug(f"Normalizing energy for class {class_idx}")
                    energy_normalizer = self._normalization_stats[energy_key]
                    values = np.array(energy_normalizer['values'])
                    percentiles = np.array(energy_normalizer['percentiles'])
                    
                    class_energy = energy_np[class_mask]
                    normalized_percentiles = np.interp(class_energy, values, percentiles)
                    normalized_energy[class_mask] = np.clip(normalized_percentiles / 100.0, 0.0, 1.0)
                else:
                    logger.warning(f"No energy normalization stats for class {class_idx} (key: {energy_key})")
                
                # Normalize Cosine scores
                cosine_key = f"cosine_class_{class_idx}"
                if cosine_key in self._normalization_stats:
                    logger.debug(f"Normalizing cosine for class {class_idx}")
                    cosine_normalizer = self._normalization_stats[cosine_key]
                    values = np.array(cosine_normalizer['values'])
                    percentiles = np.array(cosine_normalizer['percentiles'])
                    
                    class_cosine = cosine_np[class_mask]
                    normalized_percentiles = np.interp(class_cosine, values, percentiles)
                    normalized_cosine[class_mask] = np.clip(normalized_percentiles / 100.0, 0.0, 1.0)
                else:
                    logger.warning(f"No cosine normalization stats for class {class_idx} (key: {cosine_key})")
            
            return normalized_energy, normalized_cosine
        except Exception as e:
            logger.error(f"Error in normalize_scores: {e}")
            logger.error(f"Energy scores type: {type(energy_scores)}, shape: {energy_scores.shape if hasattr(energy_scores, 'shape') else 'N/A'}")
            logger.error(f"Cosine scores type: {type(cosine_scores)}, shape: {cosine_scores.shape if hasattr(cosine_scores, 'shape') else 'N/A'}")
            logger.error(f"Predicted classes type: {type(predicted_classes)}, shape: {predicted_classes.shape if hasattr(predicted_classes, 'shape') else 'N/A'}")
            raise
    
    def compute_fused_scores(self, normalized_energy, normalized_cosine):
        """
        Compute fused scores using optimized weights.
        
        Args:
            normalized_energy: (N,) array of normalized energy scores
            normalized_cosine: (N,) array of normalized cosine scores
            
        Returns:
            fused_scores: (N,) array of fused scores
        """
        w1, w2 = self.fusion_weights
        fused_scores = w1 * normalized_energy + w2 * normalized_cosine
        return fused_scores
    
    def detect_ood(self, fused_scores):
        """
        Detect OOD samples using the optimized threshold.
        
        Args:
            fused_scores: (N,) array of fused scores
            
        Returns:
            is_ood: (N,) boolean array indicating OOD samples
            ood_confidences: (N,) array of OOD confidence scores
        """
        # Samples with fused_score < threshold are considered OOD
        is_ood = fused_scores < self.ood_threshold
        
        # OOD confidence: how far below threshold (normalized to [0,1])
        # Higher confidence means more certain it's OOD
        distance_from_threshold = self.ood_threshold - fused_scores
        ood_confidences = np.clip(distance_from_threshold / self.ood_threshold, 0.0, 1.0)
        
        return is_ood, ood_confidences
    
    def score_batch(self, logits, embeddings):
        """
        Complete OSR scoring pipeline for a batch of samples.
        
        Args:
            logits: (N, num_classes) tensor of model logits
            embeddings: (N, embed_dim) tensor of embeddings
            
        Returns:
            osr_results: dict containing all OSR scores and decisions
        """
        try:
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            logger.debug(f"Predictions shape: {predictions.shape}")
            
            # Compute raw scores
            logger.debug("Computing energy scores...")
            energy_scores = self.compute_energy_score(logits)
            logger.debug(f"Energy scores shape: {energy_scores.shape}")
            
            logger.debug("Computing cosine scores...")
            cosine_scores = self.compute_cosine_score(embeddings, predictions)
            logger.debug(f"Cosine scores shape: {cosine_scores.shape}")
            
            # Normalize scores
            logger.debug("Normalizing scores...")
            energy_norm, cosine_norm = self.normalize_scores(
                energy_scores, cosine_scores, predictions
            )
            logger.debug(f"Normalized energy shape: {energy_norm.shape}, cosine shape: {cosine_norm.shape}")
            
            # Compute fused scores
            logger.debug("Computing fused scores...")
            fused_scores = self.compute_fused_scores(energy_norm, cosine_norm)
            logger.debug(f"Fused scores shape: {fused_scores.shape}")
            
            # Detect OOD samples
            logger.debug("Detecting OOD samples...")
            is_ood, ood_confidences = self.detect_ood(fused_scores)
            logger.debug(f"OOD detection complete: {is_ood.shape}")
            
            return {
                'predictions': predictions.cpu().numpy(),
                'energy_scores_raw': energy_scores.cpu().numpy(),
                'cosine_scores_raw': cosine_scores.cpu().numpy(),
                'energy_scores_normalized': energy_norm,
                'cosine_scores_normalized': cosine_norm,
                'fused_scores': fused_scores,
                'is_ood': is_ood,
                'ood_confidences': ood_confidences
            }
        except Exception as e:
            logger.error(f"Error in score_batch: {e}")
            logger.error(f"Logits shape: {logits.shape}, embeddings shape: {embeddings.shape}")
            raise
    
    def get_class_name_with_ood(self, predicted_class, is_ood, class_names=['human', 'fp']):
        """
        Get class name including OOD detection.
        
        Args:
            predicted_class: int, original predicted class index
            is_ood: bool, whether sample is detected as OOD
            class_names: list of class names
            
        Returns:
            class_name: str, final class name ('ood' if OOD, otherwise original class)
        """
        if is_ood:
            return 'ood'
        else:
            return class_names[predicted_class]
    
    def reload_configurations(self):
        """Reload OSR configurations from disk (useful for dynamic updates)."""
        logger.info("Reloading OSR configurations...")
        self._load_configurations()
        logger.info("OSR configurations reloaded successfully")
