#!/usr/bin/env python3
"""
Master script to run the complete ablation study pipeline.

This script orchestrates the entire ablation study:
1. Generate prototypes for all configurations
2. Run calibration for all configurations  
3. Run evaluation for all configurations
4. Generate comparison reports

Usage:
    python run_ablation_study.py --model_path /path/to/model.pth --config_path /path/to/config.yaml
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(step_name, script_path, args, additional_args=None):
    """Run a single step of the ablation study."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 Starting Step: {step_name}")
    logger.info(f"{'='*80}")
    
    # Build command
    cmd = ['python3', script_path]
    
    # Add common arguments
    cmd.extend([
        '--model_path', args.model_path,
        '--config_path', args.config_path,
        '--data_dir', args.data_dir,
        '--device', args.device,
        '--batch_size', str(args.batch_size)
    ])
    
    # Add step-specific arguments
    if additional_args:
        cmd.extend(additional_args)
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=args.working_dir)
        
        if result.returncode == 0:
            logger.info(f"✅ {step_name} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"❌ {step_name} failed")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        logger.error(f"❌ Exception during {step_name}: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Run complete ablation study pipeline')
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
    parser.add_argument('--working_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr',
                       help='Working directory for running scripts')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--target_tpr', type=float, default=0.99,
                       help='Target True Positive Rate for ID samples')
    parser.add_argument('--weight_step', type=float, default=0.05,
                       help='Grid search step size for weights')
    parser.add_argument('--skip_prototypes', action='store_true',
                       help='Skip prototype generation step')
    parser.add_argument('--skip_calibration', action='store_true',
                       help='Skip calibration step')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configurations to run (default: all)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Complete Ablation Study Pipeline")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Working directory: {args.working_dir}")
    
    # Track overall results
    pipeline_results = {
        'ablation_study_pipeline': {
            'start_time': datetime.now().isoformat(),
            'model_path': args.model_path,
            'config_path': args.config_path,
            'data_dir': args.data_dir,
            'steps': {}
        }
    }
    
    # Step 1: Generate Prototypes
    if not args.skip_prototypes:
        step_args = [
            '--output_dir', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes'
        ]
        
        success, output = run_step(
            "Prototype Generation",
            '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/generate_prototypes_ablation.py',
            args,
            step_args
        )
        
        pipeline_results['ablation_study_pipeline']['steps']['prototype_generation'] = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'output': output if success else output
        }
        
        if not success:
            logger.error("❌ Prototype generation failed. Stopping pipeline.")
            return 1
    else:
        logger.info("⏭️  Skipping prototype generation step")
        pipeline_results['ablation_study_pipeline']['steps']['prototype_generation'] = {
            'skipped': True,
            'timestamp': datetime.now().isoformat()
        }
    
    # Step 2: Run Calibration
    if not args.skip_calibration:
        step_args = [
            '--calibration_metadata', args.calibration_metadata,
            '--prototypes_base_dir', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
            '--output_base_dir', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
            '--ablation_config', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/ablation_config.json',
            '--unified_calibration_script', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/unified_calibration.py',
            '--target_tpr', str(args.target_tpr),
            '--weight_step', str(args.weight_step)
        ]
        
        if args.configs:
            step_args.extend(['--configs'] + args.configs)
        
        success, output = run_step(
            "Calibration",
            '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/run_ablation_calibration.py',
            args,
            step_args
        )
        
        pipeline_results['ablation_study_pipeline']['steps']['calibration'] = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'output': output if success else output
        }
        
        if not success:
            logger.error("❌ Calibration failed. Stopping pipeline.")
            return 1
    else:
        logger.info("⏭️  Skipping calibration step")
        pipeline_results['ablation_study_pipeline']['steps']['calibration'] = {
            'skipped': True,
            'timestamp': datetime.now().isoformat()
        }
    
    # Step 3: Run Evaluation
    if not args.skip_evaluation:
        step_args = [
            '--calibration_metadata', args.calibration_metadata,
            '--calibration_base_dir', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
            '--evaluation_base_dir', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/evaluation',
            '--prototypes_base_dir', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
            '--ablation_config', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/ablation_config.json',
            '--osr_eval_script', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/pointnext/classification/osr_eval.py'
        ]
        
        if args.configs:
            step_args.extend(['--configs'] + args.configs)
        
        success, output = run_step(
            "Evaluation",
            '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/run_ablation_evaluation.py',
            args,
            step_args
        )
        
        pipeline_results['ablation_study_pipeline']['steps']['evaluation'] = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'output': output if success else output
        }
        
        if not success:
            logger.error("❌ Evaluation failed.")
    else:
        logger.info("⏭️  Skipping evaluation step")
        pipeline_results['ablation_study_pipeline']['steps']['evaluation'] = {
            'skipped': True,
            'timestamp': datetime.now().isoformat()
        }
    
    # Save pipeline results
    pipeline_results['ablation_study_pipeline']['end_time'] = datetime.now().isoformat()
    
    results_path = '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/ablation_study_pipeline_results.json'
    with open(results_path, 'w') as f:
        import json
        json.dump(pipeline_results, f, indent=2)
    
    # Print final summary
    logger.info(f"\n{'='*80}")
    logger.info("🎉 Ablation Study Pipeline Completed!")
    logger.info(f"{'='*80}")
    
    for step_name, step_result in pipeline_results['ablation_study_pipeline']['steps'].items():
        if step_result.get('skipped'):
            logger.info(f"⏭️  {step_name}: Skipped")
        elif step_result.get('success'):
            logger.info(f"✅ {step_name}: Success")
        else:
            logger.info(f"❌ {step_name}: Failed")
    
    logger.info(f"Pipeline results saved to: {results_path}")
    logger.info(f"{'='*80}")
    
    # Check if any step failed
    failed_steps = [name for name, result in pipeline_results['ablation_study_pipeline']['steps'].items() 
                   if not result.get('skipped') and not result.get('success')]
    
    if failed_steps:
        logger.error(f"❌ Pipeline completed with failures in: {failed_steps}")
        return 1
    else:
        logger.info("✅ Pipeline completed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())
