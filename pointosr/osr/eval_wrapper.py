#!/usr/bin/env python3
"""
Wrapper script for OSR evaluation that handles command line arguments
and sets up the configuration properly.
"""

import os
import sys
import argparse
import json
import tempfile
import yaml

# Add the pointosr modules to the path
sys.path.insert(0, "/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr")
sys.path.insert(0, "/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/pointnext")

from pointnext.utils import EasyConfig, setup_logger_dist
from pointnext.classification.osr_eval import main as osr_eval_main

def main():
    parser = argparse.ArgumentParser('OSR Evaluation Wrapper')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True, help='Path to model config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--calibration_metadata', type=str, required=True, help='Path to calibration metadata JSON')
    parser.add_argument('--fusion_config_path', type=str, required=True, help='Path to fusion config JSON')
    parser.add_argument('--stats_path', type=str, required=True, help='Path to stats JSON')
    parser.add_argument('--prototypes_path', type=str, required=True, help='Path to prototypes directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load base config
    cfg = EasyConfig()
    cfg.load(args.config_path, recursive=True)
    
    # Load model config from pointnext-s.yaml
    model_config_path = "/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/pointnext/cfgs/pointnext-s.yaml"
    model_cfg = EasyConfig()
    model_cfg.load(model_config_path, recursive=True)
    cfg.model = model_cfg.model
    
    # Override config with command line arguments
    cfg.pretrained_path = args.model_path
    cfg.dataset.common.data_dir = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.device = args.device
    
    # Set OSR evaluation config
    cfg.osr_eval = EasyConfig()
    cfg.osr_eval.results_dir = args.output_dir
    cfg.osr_eval.fusion_config_path = args.fusion_config_path
    cfg.osr_eval.stats_path = args.stats_path
    # For ablation study, we need to pass the specific configuration directory
    # Extract config name from output directory
    config_name = os.path.basename(args.output_dir)
    cfg.osr_eval.prototypes_path = os.path.join(args.prototypes_path, config_name)
    cfg.osr_eval.calibration_metadata = args.calibration_metadata
    cfg.osr_eval.eval_dataset = "test"  # Use test split for evaluation
    
    # Set required fields for logging
    cfg.rank = 0
    cfg.world_size = 1
    cfg.distributed = False
    cfg.mp = False
    cfg.log_dir = args.output_dir
    cfg.run_dir = args.output_dir
    cfg.run_name = "osr_eval"
    cfg.log_path = os.path.join(args.output_dir, "eval.log")
    
    # Set up logging
    setup_logger_dist(output=args.output_dir, distributed_rank=0, name="osr_eval")
    
    # Run evaluation
    osr_eval_main(0, cfg)

if __name__ == "__main__":
    main()
