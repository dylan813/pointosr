#!/usr/bin/env python3
"""
Run calibration for all ablation study configurations.

This script runs the unified calibration pipeline for each prototype configuration
in the ablation study and organizes the results.
"""

import os
import sys
import subprocess
import json
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ablation_config(config_path):
    """Load ablation study configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_calibration_for_config(config_name, k_human, k_fp, args):
    """Run calibration for a specific configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running calibration for {config_name}")
    logger.info(f"Human prototypes: K={k_human}, False-positive prototypes: K={k_fp}")
    logger.info(f"{'='*60}")
    
    # Create configuration-specific output directory
    config_output_dir = os.path.join(args.output_base_dir, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Update prototypes path for this configuration
    prototypes_path = os.path.join(args.prototypes_base_dir, config_name)
    
    # Build calibration command
    cmd = [
        'python3', args.unified_calibration_script,
        '--model_path', args.model_path,
        '--config_path', args.config_path,
        '--data_dir', args.data_dir,
        '--calibration_metadata', args.calibration_metadata,
        '--prototypes_path', prototypes_path,
        '--output_dir', config_output_dir,
        '--batch_size', str(args.batch_size),
        '--device', args.device,
        '--target_tpr', str(args.target_tpr),
        '--weight_step', str(args.weight_step)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run calibration
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=args.working_dir)
        
        if result.returncode == 0:
            logger.info(f"✅ Calibration completed successfully for {config_name}")
            return True, result.stdout
        else:
            logger.error(f"❌ Calibration failed for {config_name}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        logger.error(f"❌ Exception during calibration for {config_name}: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Run ablation study calibration')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/cerlab/Documents/data/pointosr/dataset',
                       help='Path to data directory')
    parser.add_argument('--calibration_metadata', type=str,
                       default='/home/cerlab/Documents/data/pointosr/dataset/calibration_selection.json',
                       help='Path to calibration metadata JSON')
    parser.add_argument('--prototypes_base_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
                       help='Base directory containing prototype configurations')
    parser.add_argument('--output_base_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
                       help='Base output directory for calibration results')
    parser.add_argument('--ablation_config', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/ablation_config.json',
                       help='Path to ablation study configuration')
    parser.add_argument('--unified_calibration_script', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/unified_calibration.py',
                       help='Path to unified calibration script')
    parser.add_argument('--working_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr',
                       help='Working directory for running scripts')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for calibration')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--target_tpr', type=float, default=0.99,
                       help='Target True Positive Rate for ID samples')
    parser.add_argument('--weight_step', type=float, default=0.05,
                       help='Grid search step size for weights')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configurations to run (default: all)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Ablation Study Calibration")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Output base: {args.output_base_dir}")
    
    # Load ablation configuration
    ablation_config = load_ablation_config(args.ablation_config)
    configurations = ablation_config['ablation_study']['configurations']
    
    # Filter configurations if specified
    if args.configs:
        configurations = [c for c in configurations if c['name'] in args.configs]
        logger.info(f"Running specific configurations: {args.configs}")
    else:
        logger.info(f"Running all {len(configurations)} configurations")
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Track results
    results = {
        'ablation_calibration': {
            'start_time': datetime.now().isoformat(),
            'model_path': args.model_path,
            'config_path': args.config_path,
            'data_dir': args.data_dir,
            'configurations': {}
        }
    }
    
    successful_configs = []
    failed_configs = []
    
    # Run calibration for each configuration
    for config in configurations:
        config_name = config['name']
        k_human = config['human_prototypes']
        k_fp = config['fp_prototypes']
        
        success, output = run_calibration_for_config(config_name, k_human, k_fp, args)
        
        results['ablation_calibration']['configurations'][config_name] = {
            'k_human': k_human,
            'k_fp': k_fp,
            'success': success,
            'output_dir': os.path.join(args.output_base_dir, config_name),
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            successful_configs.append(config_name)
        else:
            failed_configs.append(config_name)
            results['ablation_calibration']['configurations'][config_name]['error'] = output
    
    # Save results summary
    results['ablation_calibration']['end_time'] = datetime.now().isoformat()
    results['ablation_calibration']['summary'] = {
        'total_configurations': len(configurations),
        'successful': len(successful_configs),
        'failed': len(failed_configs),
        'successful_configs': successful_configs,
        'failed_configs': failed_configs
    }
    
    results_path = os.path.join(args.output_base_dir, 'ablation_calibration_summary.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("🎉 Ablation Study Calibration Completed!")
    logger.info(f"Total configurations: {len(configurations)}")
    logger.info(f"Successful: {len(successful_configs)}")
    logger.info(f"Failed: {len(failed_configs)}")
    
    if successful_configs:
        logger.info("✅ Successful configurations:")
        for config in successful_configs:
            logger.info(f"  - {config}")
    
    if failed_configs:
        logger.info("❌ Failed configurations:")
        for config in failed_configs:
            logger.info(f"  - {config}")
    
    logger.info(f"Results summary saved to: {results_path}")
    logger.info(f"{'='*60}")
    
    # Return appropriate exit code
    return 0 if len(failed_configs) == 0 else 1

if __name__ == "__main__":
    exit(main())
