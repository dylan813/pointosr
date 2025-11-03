#!/usr/bin/env python3
"""
Online Calibration Manager for PointOSR

This module provides the OnlineCalibrationManager class that performs
OSR calibration during node initialization, eliminating the need for
offline preprocessing steps.

Features:
- Automatic prototype generation from training data
- Online temperature calibration
- Score normalization and fusion optimization
- Cached calibration results with automatic refresh
- Fallback to standard classification on failure
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import time

# Add the pointosr package to Python path
pointosr_path = '/home/cerlab/Documents/dynablox_ws/src/pointosr/pointosr'
sys.path.insert(0, pointosr_path)

from pointnext.utils import EasyConfig
from pointnext.dataset.human.human import HumanDataset
from pointnext.model import build_model_from_cfg

# Import existing calibration modules
from osr.generate_prototypes import EmbeddingExtractor, generate_class_prototypes, save_prototypes
from osr.unified_calibration import (
    TemperatureScaling, EnergyScorer, CosineScorer, PercentileNormalizer, 
    FusionOptimizer, load_model, load_calibration_data, extract_logits_and_embeddings,
    load_prototypes, compute_calibration_metrics, save_results
)

logger = logging.getLogger(__name__)

class OnlineCalibrationManager:
    """
    Manager for online OSR calibration during node initialization.
    
    This class orchestrates the calibration process using existing modules:
    1. Prototype generation from training data (using generate_prototypes.py)
    2. Temperature calibration (using unified_calibration.py)
    3. Score normalization and fusion optimization (using unified_calibration.py)
    """
    
    def __init__(self, 
                 model_path,
                 config_path, 
                 data_dir,
                 calibration_cache_dir='src/pointosr/calib_cache',
                 k_human=8,
                 k_false=2,
                 target_tpr=0.95,
                 device='cuda'):
        """
        Initialize the online calibration manager.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration file
            data_dir: Path to dataset directory containing splits and clusters
            calibration_cache_dir: Directory to cache calibration results
            k_human: Number of human prototypes
            k_false: Number of false-positive prototypes  
            target_tpr: Target True Positive Rate for ID samples
            device: Device to use for torch operations
        """
        self.model_path = model_path
        self.config_path = config_path
        self.data_dir = data_dir
        self.calibration_cache_dir = calibration_cache_dir
        self.k_human = k_human
        self.k_false = k_false
        self.target_tpr = target_tpr
        self.device = device
        
        # Resolve cache directory relative to pointosr repository root
        if not os.path.isabs(self.calibration_cache_dir):
            # Find the pointosr repository root by traversing up from this file's location
            current_dir = os.path.dirname(os.path.abspath(__file__))  # This file is in osr/
            pointosr_root = None
            
            # Traverse up the directory tree looking for a directory containing 'pointosr' subdirectory
            search_dir = current_dir
            max_depth = 10  # Prevent infinite loops
            depth = 0
            
            while search_dir != '/' and depth < max_depth:
                # Check if this directory contains a 'pointosr' subdirectory
                if os.path.exists(os.path.join(search_dir, 'pointosr')):
                    pointosr_root = search_dir
                    break
                search_dir = os.path.dirname(search_dir)
                depth += 1
            
            if pointosr_root:
                # Handle relative paths that start with 'src/pointosr/'
                if self.calibration_cache_dir.startswith('src/pointosr/'):
                    # Remove 'src/pointosr/' prefix and append to pointosr root
                    cache_subpath = self.calibration_cache_dir[13:]  # Remove 'src/pointosr/' prefix
                    self.calibration_cache_dir = os.path.join(pointosr_root, cache_subpath)
                else:
                    # Use the relative path as-is from pointosr root
                    self.calibration_cache_dir = os.path.join(pointosr_root, self.calibration_cache_dir)
                logger.info(f"Resolved cache directory to: {self.calibration_cache_dir}")
            else:
                # Fallback: treat as absolute path or relative to current working directory
                if not os.path.isabs(self.calibration_cache_dir):
                    self.calibration_cache_dir = os.path.abspath(self.calibration_cache_dir)
                logger.warning(f"Could not find pointosr repository root, using absolute path: {self.calibration_cache_dir}")
        
        # Create cache directory
        os.makedirs(self.calibration_cache_dir, exist_ok=True)
        
        # Cache paths
        self.cache_config_path = os.path.join(self.calibration_cache_dir, 'fusion_config.json')
        self.cache_stats_path = os.path.join(self.calibration_cache_dir, 'stats.json')
        self.cache_prototypes_dir = os.path.join(self.calibration_cache_dir, 'prototypes')
        
        # Initialize calibration modules
        self.embedding_extractor = None
        self.temp_scaler = TemperatureScaling()
        self.energy_scorer = None
        self.cosine_scorer = None
        self.normalizer = PercentileNormalizer()
        self.fusion_optimizer = FusionOptimizer(target_tpr=self.target_tpr)
        
        logger.info("Online Calibration Manager initialized")
        logger.info(f"Model: {model_path}")
        logger.info(f"Data: {data_dir}")
        logger.info(f"Cache: {calibration_cache_dir}")
    
    # def should_recalibrate(self):
    #     """
    #     Check if calibration needs to be redone based on cache validity.
    #     DISABLED: Always run fresh calibration instead of using cache.
    #     
    #     Returns:
    #         bool: True if recalibration is needed
    #     """
    #     # Check if all cache files exist
    #     required_files = [
    #         self.cache_config_path,
    #         self.cache_stats_path,
    #         os.path.join(self.cache_prototypes_dir, f'human_k{self.k_human}', 'prototypes.npy'),
    #         os.path.join(self.cache_prototypes_dir, f'fp_k{self.k_false}', 'prototypes.npy')
    #     ]
    #     
    #     for file_path in required_files:
    #         if not os.path.exists(file_path):
    #             logger.info(f"Missing cache file: {file_path}")
    #             return True
    #     
    #     # Check if model file is newer than cache
    #     model_mtime = os.path.getmtime(self.model_path)
    #     for file_path in required_files:
    #         cache_mtime = os.path.getmtime(file_path)
    #         if model_mtime > cache_mtime:
    #             logger.info(f"Model is newer than cache: {file_path}")
    #             return True
    #     
    #     logger.info("Using cached calibration results")
    #     return False
    
    def load_model_and_data(self):
        """Load model and prepare datasets using existing modules."""
        logger.info("Loading model and datasets...")
        
        # Load model using existing function
        self.model, self.cfg = load_model(self.model_path, self.config_path, self.device)
        
        # Load datasets
        self.train_dataset = HumanDataset(
            data_dir=self.data_dir,
            split='train',
            num_points=2048,
            transform=None,
            uniform_sample=False
        )
        
        self.calib_dataset = HumanDataset(
            data_dir=self.data_dir,
            split='calib',
            num_points=2048,
            transform=None,
            uniform_sample=False
        )
        
        logger.info(f"Model loaded: {self.model_path}")
        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Calib dataset: {len(self.calib_dataset)} samples")
    
    def generate_prototypes(self):
        """Generate prototypes using existing generate_prototypes module."""
        logger.info("Generating prototypes using existing module...")
        
        # Create embedding extractor
        self.embedding_extractor = EmbeddingExtractor(
            self.model_path, self.config_path, self.device
        )
        
        # Extract embeddings from training dataset
        embeddings, labels, sample_ids = self.embedding_extractor.extract_embeddings(
            self.train_dataset, batch_size=32
        )
        
        # Generate human prototypes (class 0)
        human_prototypes, human_sample_ids, _ = generate_class_prototypes(
            embeddings, labels, sample_ids, 
            class_idx=0, k_prototypes=self.k_human, class_name="human"
        )
        
        # Generate false-positive prototypes (class 1)
        fp_prototypes, fp_sample_ids, _ = generate_class_prototypes(
            embeddings, labels, sample_ids,
            class_idx=1, k_prototypes=self.k_false, class_name="fp"
        )
        
        # Save prototypes (use K-specific directories to match consumers)
        human_dir = os.path.join(self.cache_prototypes_dir, f'human_k{self.k_human}')
        fp_dir = os.path.join(self.cache_prototypes_dir, f'fp_k{self.k_false}')
        
        # Create metadata
        human_metadata = {
            'class_name': 'human',
            'class_idx': 0,
            'num_prototypes': len(human_prototypes),
            'embedding_dim': human_prototypes.shape[1],
            'prototype_sample_ids': human_sample_ids,
            'total_samples_used': int(np.sum(labels == 0))
        }
        
        fp_metadata = {
            'class_name': 'false_positive',
            'class_idx': 1,
            'num_prototypes': len(fp_prototypes),
            'embedding_dim': fp_prototypes.shape[1],
            'prototype_sample_ids': fp_sample_ids,
            'total_samples_used': int(np.sum(labels == 1))
        }
        
        # Save prototypes using existing function
        save_prototypes(human_prototypes, human_sample_ids, human_metadata, human_dir)
        save_prototypes(fp_prototypes, fp_sample_ids, fp_metadata, fp_dir)
        
        logger.info(f"Prototypes saved:")
        logger.info(f"  Human: {human_prototypes.shape} -> {human_dir}")
        logger.info(f"  FP: {fp_prototypes.shape} -> {fp_dir}")
        
        return human_prototypes, fp_prototypes
    
    def run_calibration_pipeline(self):
        """Run the complete calibration pipeline using existing modules."""
        logger.info("Running calibration pipeline using existing modules...")
        
        # Extract features for calibration using existing function
        logits, embeddings, labels = extract_logits_and_embeddings(
            self.model, self.calib_dataset, batch_size=32, device=self.device
        )
        
        # Temperature calibration using existing module
        optimal_temp = self.temp_scaler.set_temperature(logits, labels)
        
        # Load prototypes
        human_prototypes, fp_prototypes = self._load_prototypes()
        
        # Initialize scorers
        self.energy_scorer = EnergyScorer(temperature=optimal_temp)
        self.cosine_scorer = CosineScorer(human_prototypes, fp_prototypes)
        
        # Get predictions for score computation
        scaled_logits = logits / optimal_temp
        probs = F.softmax(scaled_logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        # Compute scores using existing modules
        energy_scores = self.energy_scorer.compute_energy(logits)
        cosine_scores = self.cosine_scorer.compute_cosine(embeddings, predictions)
        
        # Normalize scores using existing module
        self.normalizer.fit(energy_scores.numpy(), predictions.numpy(), 'energy')
        self.normalizer.fit(cosine_scores.numpy(), predictions.numpy(), 'cosine')
        
        # Normalize scores. Flip energy so higher means better (ID) before fusion
        energy_norm = self.normalizer.transform(energy_scores.numpy(), predictions.numpy(), 'energy')
        energy_norm = 1.0 - energy_norm
        cosine_norm = self.normalizer.transform(cosine_scores.numpy(), predictions.numpy(), 'cosine')
        
        # Optimize fusion using existing module
        best_result, _ = self.fusion_optimizer.optimize(energy_norm, cosine_norm, labels.numpy())
        
        # Save results using existing function
        fusion_config = {
            "T": optimal_temp,
            "K_H": self.k_human,
            "K_F": self.k_false,
            "fused_weights": [best_result['w1'], best_result['w2']],
            "fused_threshold": best_result['threshold'],
            "target_tpr": self.target_tpr,
            "achieved_tpr": best_result['tpr'],
            "achieved_fpr": best_result['fpr'],
            "achieved_auroc": best_result['auroc'],
            "score_directions": {
                "energy": "higher_is_better",
                "cosine": "higher_is_better"
            },
            "calibration_data": {
                "model_path": self.model_path,
                "config_path": self.config_path,
                "data_dir": self.data_dir,
                "calibration_samples": len(self.calib_dataset)
            }
        }
        
        # Save fusion config
        with open(self.cache_config_path, 'w') as f:
            json.dump(fusion_config, f, indent=2)
        
        # Save normalization stats
        self.normalizer.save(self.cache_stats_path)
        
        logger.info(f"Calibration completed successfully!")
        logger.info(f"Temperature: {optimal_temp:.4f}")
        logger.info(f"Fusion weights: [{best_result['w1']:.2f}, {best_result['w2']:.2f}]")
        logger.info(f"Threshold: {best_result['threshold']:.6f}")
        logger.info(f"TPR: {best_result['tpr']:.3f}, FPR: {best_result['fpr']:.3f}")
        
        return True
    
    def _load_prototypes(self):
        """Load prototypes from cache directory."""
        human_prototypes_path = os.path.join(self.cache_prototypes_dir, f'human_k{self.k_human}', 'prototypes.npy')
        fp_prototypes_path = os.path.join(self.cache_prototypes_dir, f'fp_k{self.k_false}', 'prototypes.npy')
        
        human_prototypes = np.load(human_prototypes_path)
        fp_prototypes = np.load(fp_prototypes_path)
        
        logger.info(f"Loaded prototypes:")
        logger.info(f"  Human: {human_prototypes.shape}")
        logger.info(f"  FP: {fp_prototypes.shape}")
        
        return human_prototypes, fp_prototypes
    
    def run_calibration(self):
        """
        Run the complete online calibration pipeline using existing modules.
        Always runs fresh calibration (no caching).
        
        Returns:
            tuple: (success, fusion_config_path, stats_path, prototypes_path)
        """
        start_time = time.time()
        logger.info("🚀 Starting online calibration pipeline (fresh calibration)...")
        
        lock_path = f"{self.calibration_cache_dir}.lock"
        lock_created = False
        try:
            lock_parent = os.path.dirname(lock_path) or '.'
            try:
                os.makedirs(lock_parent, exist_ok=True)
                with open(lock_path, 'w') as lock_file:
                    lock_file.write(json.dumps({'started_at': time.time()}))
                lock_created = True
                logger.info(f"🔒 Created calibration lock at {lock_path}")
            except Exception as lock_err:
                logger.warning(f"⚠️ Could not create calibration lock file: {lock_err}")

            # Always clear existing cache to ensure fresh calibration
            if os.path.exists(self.calibration_cache_dir):
                import shutil
                logger.info("🧹 Clearing existing calibration cache for fresh calibration...")
                shutil.rmtree(self.calibration_cache_dir)
            
            # Recreate cache directory
            os.makedirs(self.calibration_cache_dir, exist_ok=True)
            logger.info("📁 Created fresh cache directory")
            
            # Load model and data
            self.load_model_and_data()
            
            # Step 1: Generate prototypes using existing module
            logger.info("\n" + "="*50)
            logger.info("STEP 1: PROTOTYPE GENERATION")
            logger.info("="*50)
            
            self.generate_prototypes()
            
            # Step 2: Run calibration pipeline using existing modules
            logger.info("\n" + "="*50)
            logger.info("STEP 2: CALIBRATION PIPELINE")
            logger.info("="*50)
            
            self.run_calibration_pipeline()
            
            # Write a completion marker to avoid race conditions
            try:
                completion_marker = os.path.join(self.calibration_cache_dir, 'calibration_complete.stamp')
                with open(completion_marker, 'w') as f:
                    f.write(json.dumps({
                        'completed_at': time.time(),
                        'K_H': self.k_human,
                        'K_F': self.k_false
                    }))
            except Exception as e_marker:
                logger.warning(f"Failed to write calibration completion marker: {e_marker}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n🎉 Online calibration completed successfully in {elapsed_time:.2f} seconds!")
            
            return True, self.cache_config_path, self.cache_stats_path, self.cache_prototypes_dir
            
        except Exception as e:
            logger.error(f"❌ Online calibration failed: {e}")
            return False, None, None, None
        finally:
            if lock_created:
                try:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                        logger.info(f"🔓 Removed calibration lock at {lock_path}")
                except Exception as unlock_err:
                    logger.warning(f"⚠️ Could not remove calibration lock file: {unlock_err}")