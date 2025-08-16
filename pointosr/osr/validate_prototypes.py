#!/usr/bin/env python3
"""
Validate that the generated prototypes are correctly saved and can be loaded.
"""

import os
import sys
import numpy as np
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_prototypes(prototype_dir):
    """Validate a prototype directory."""
    logger.info(f"Validating prototypes in: {prototype_dir}")
    
    # Check if directory exists
    if not os.path.exists(prototype_dir):
        logger.error(f"Prototype directory does not exist: {prototype_dir}")
        return False
    
    # Check required files
    prototypes_file = os.path.join(prototype_dir, 'prototypes.npy')
    metadata_file = os.path.join(prototype_dir, 'metadata.json')
    
    if not os.path.exists(prototypes_file):
        logger.error(f"Prototypes file missing: {prototypes_file}")
        return False
        
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file missing: {metadata_file}")
        return False
    
    # Load and validate prototypes
    try:
        prototypes = np.load(prototypes_file)
        logger.info(f"✅ Loaded prototypes shape: {prototypes.shape}")
    except Exception as e:
        logger.error(f"Failed to load prototypes: {e}")
        return False
    
    # Load and validate metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✅ Loaded metadata for class: {metadata['class_name']}")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return False
    
    # Validate consistency
    expected_prototypes = metadata['num_prototypes']
    expected_dim = metadata['embedding_dim']
    
    if prototypes.shape[0] != expected_prototypes:
        logger.error(f"Prototype count mismatch: got {prototypes.shape[0]}, expected {expected_prototypes}")
        return False
        
    if prototypes.shape[1] != expected_dim:
        logger.error(f"Embedding dimension mismatch: got {prototypes.shape[1]}, expected {expected_dim}")
        return False
    
    # Check that prototypes are not all zeros
    if np.allclose(prototypes, 0):
        logger.error("All prototypes are zero - this suggests an issue with extraction")
        return False
    
    # Print statistics
    logger.info(f"   Class: {metadata['class_name']} (idx: {metadata['class_idx']})")
    logger.info(f"   Prototypes: {prototypes.shape[0]}")
    logger.info(f"   Embedding dim: {prototypes.shape[1]}")
    logger.info(f"   Training samples used: {metadata['total_samples_used']}")
    logger.info(f"   Prototype sample IDs: {metadata['prototype_sample_ids']}")
    logger.info(f"   Prototype stats - mean: {prototypes.mean():.4f}, std: {prototypes.std():.4f}")
    logger.info(f"   Prototype norms: {np.linalg.norm(prototypes, axis=1)}")
    
    return True

def main():
    """Main validation function."""
    base_dir = '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes'
    
    logger.info("🧪 Validating OSR Prototypes")
    logger.info("=" * 50)
    
    # Validate human prototypes
    human_dir = os.path.join(base_dir, 'human_k6')
    human_valid = validate_prototypes(human_dir)
    
    logger.info("")
    
    # Validate false-positive prototypes
    fp_dir = os.path.join(base_dir, 'fp_k4')
    fp_valid = validate_prototypes(fp_dir)
    
    logger.info("")
    logger.info("=" * 50)
    
    if human_valid and fp_valid:
        logger.info("🎉 All prototypes validated successfully!")
        logger.info("   ✅ Human prototypes (K_H=6): Valid")
        logger.info("   ✅ False-positive prototypes (K_F=4): Valid")
        logger.info("   ✅ Ready for Phase 3: Temperature calibration")
        
        # Check summary file
        summary_file = os.path.join(base_dir, 'prototype_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            logger.info(f"\n📊 Summary:")
            logger.info(f"   Total training samples: {summary['total_training_samples']}")
            logger.info(f"   Human prototypes: {summary['human_prototypes']['count']}")
            logger.info(f"   FP prototypes: {summary['fp_prototypes']['count']}")
            logger.info(f"   Embedding dimension: {summary['human_prototypes']['embedding_dim']}")
        
        return True
    else:
        logger.error("❌ Prototype validation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
