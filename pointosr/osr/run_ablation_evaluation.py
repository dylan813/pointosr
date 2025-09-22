#!/usr/bin/env python3
"""
Run evaluation for all ablation study configurations.

This script runs the OSR evaluation for each calibrated configuration
and compares the results across different prototype counts.
"""

import os
import sys
import subprocess
import json
import argparse
import logging
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ablation_config(config_path):
    """Load ablation study configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_evaluation_for_config(config_name, args):
    """Run evaluation for a specific configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running evaluation for {config_name}")
    logger.info(f"{'='*60}")
    
    # Configuration-specific paths
    config_calibration_dir = os.path.join(args.calibration_base_dir, config_name)
    config_output_dir = os.path.join(args.evaluation_base_dir, config_name)
    
    # Check if calibration results exist
    fusion_config_path = os.path.join(config_calibration_dir, 'fusion_config.json')
    stats_path = os.path.join(config_calibration_dir, 'stats.json')
    
    if not os.path.exists(fusion_config_path) or not os.path.exists(stats_path):
        logger.error(f"❌ Calibration files not found for {config_name}")
        logger.error(f"  Expected: {fusion_config_path}")
        logger.error(f"  Expected: {stats_path}")
        return False, "Calibration files not found"
    
    # Create output directory
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Build evaluation command
    cmd = [
        'python3', '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/eval_wrapper.py',
        '--model_path', args.model_path,
        '--config_path', args.config_path,
        '--data_dir', args.data_dir,
        '--calibration_metadata', args.calibration_metadata,
        '--fusion_config_path', fusion_config_path,
        '--stats_path', stats_path,
        '--prototypes_path', args.prototypes_base_dir,
        '--output_dir', config_output_dir,
        '--batch_size', str(args.batch_size),
        '--device', args.device
    ]
    
    # Set PYTHONPATH to include the pointosr modules
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr:/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/pointnext:' + env.get('PYTHONPATH', '')
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=args.working_dir, env=env)
        
        if result.returncode == 0:
            logger.info(f"✅ Evaluation completed successfully for {config_name}")
            return True, result.stdout
        else:
            logger.error(f"❌ Evaluation failed for {config_name}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        logger.error(f"❌ Exception during evaluation for {config_name}: {e}")
        return False, str(e)

def collect_evaluation_results(evaluation_base_dir, configurations):
    """Collect and compare evaluation results across configurations."""
    logger.info("\n📊 Collecting evaluation results...")
    
    all_results = {}
    comparison_data = []
    
    for config in configurations:
        config_name = config['name']
        config_dir = os.path.join(evaluation_base_dir, config_name)
        
        # Look for evaluation results
        eval_results_path = os.path.join(config_dir, 'osr_evaluation_results.json')
        eval_summary_path = os.path.join(config_dir, 'evaluation_summary.json')
        
        if os.path.exists(eval_results_path):
            with open(eval_results_path, 'r') as f:
                results = json.load(f)
                all_results[config_name] = results
                
                # Extract key metrics for comparison
                if 'metrics' in results:
                    metrics = results['metrics']
                    comparison_data.append({
                        'configuration': config_name,
                        'k_human': config['human_prototypes'],
                        'k_fp': config['fp_prototypes'],
                        'auc_roc': metrics.get('auc_roc', 0.0),
                        'auc_pr': metrics.get('auc_pr', 0.0),
                        'f1_score': metrics.get('f1_score', 0.0),
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'accuracy': metrics.get('accuracy', 0.0),
                        'tpr_at_fpr_01': metrics.get('tpr_at_fpr_01', 0.0),
                        'tpr_at_fpr_05': metrics.get('tpr_at_fpr_05', 0.0)
                    })
        else:
            logger.warning(f"No evaluation results found for {config_name}")
    
    return all_results, comparison_data

def create_comparison_report(comparison_data, output_path):
    """Create a comparison report of all configurations."""
    if not comparison_data:
        logger.warning("No comparison data available")
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame(comparison_data)
    
    # Sort by AUC-ROC (primary metric)
    df_sorted = df.sort_values('auc_roc', ascending=False)
    
    # Create summary statistics
    summary_stats = {
        'best_configuration': df_sorted.iloc[0]['configuration'],
        'best_auc_roc': float(df_sorted.iloc[0]['auc_roc']),
        'best_auc_pr': float(df_sorted.iloc[0]['auc_pr']),
        'best_f1_score': float(df_sorted.iloc[0]['f1_score']),
        'metrics_summary': {
            'auc_roc': {
                'mean': float(df['auc_roc'].mean()),
                'std': float(df['auc_roc'].std()),
                'min': float(df['auc_roc'].min()),
                'max': float(df['auc_roc'].max())
            },
            'auc_pr': {
                'mean': float(df['auc_pr'].mean()),
                'std': float(df['auc_pr'].std()),
                'min': float(df['auc_pr'].min()),
                'max': float(df['auc_pr'].max())
            },
            'f1_score': {
                'mean': float(df['f1_score'].mean()),
                'std': float(df['f1_score'].std()),
                'min': float(df['f1_score'].min()),
                'max': float(df['f1_score'].max())
            }
        }
    }
    
    # Save detailed comparison
    comparison_report = {
        'ablation_evaluation_comparison': {
            'timestamp': datetime.now().isoformat(),
            'total_configurations': len(comparison_data),
            'summary_statistics': summary_stats,
            'ranked_configurations': df_sorted.to_dict('records'),
            'all_configurations': df.to_dict('records')
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    # Save CSV for easy analysis
    csv_path = output_path.replace('.json', '.csv')
    df_sorted.to_csv(csv_path, index=False)
    
    logger.info(f"📊 Comparison report saved to: {output_path}")
    logger.info(f"📊 CSV data saved to: {csv_path}")
    
    # Print top configurations
    logger.info("\n🏆 Top 3 configurations by AUC-ROC:")
    for i, (_, row) in enumerate(df_sorted.head(3).iterrows()):
        logger.info(f"  {i+1}. {row['configuration']}: AUC-ROC={row['auc_roc']:.4f}, AUC-PR={row['auc_pr']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run ablation study evaluation')
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
    parser.add_argument('--calibration_base_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration',
                       help='Base directory containing calibration results')
    parser.add_argument('--evaluation_base_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/evaluation',
                       help='Base output directory for evaluation results')
    parser.add_argument('--prototypes_base_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/prototypes',
                       help='Base directory containing prototype configurations')
    parser.add_argument('--ablation_config', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/ablation_config.json',
                       help='Path to ablation study configuration')
    parser.add_argument('--osr_eval_script', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/pointnext/classification/osr_eval.py',
                       help='Path to OSR evaluation script')
    parser.add_argument('--working_dir', type=str,
                       default='/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr',
                       help='Working directory for running scripts')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configurations to run (default: all)')
    parser.add_argument('--collect_only', action='store_true',
                       help='Only collect and compare existing results (skip running evaluations)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Ablation Study Evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Evaluation base: {args.evaluation_base_dir}")
    
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
    os.makedirs(args.evaluation_base_dir, exist_ok=True)
    
    # Track results
    results = {
        'ablation_evaluation': {
            'start_time': datetime.now().isoformat(),
            'model_path': args.model_path,
            'config_path': args.config_path,
            'data_dir': args.data_dir,
            'configurations': {}
        }
    }
    
    successful_configs = []
    failed_configs = []
    
    if not args.collect_only:
        # Run evaluation for each configuration
        for config in configurations:
            config_name = config['name']
            
            success, output = run_evaluation_for_config(config_name, args)
            
            results['ablation_evaluation']['configurations'][config_name] = {
                'k_human': config['human_prototypes'],
                'k_fp': config['fp_prototypes'],
                'success': success,
                'output_dir': os.path.join(args.evaluation_base_dir, config_name),
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                successful_configs.append(config_name)
            else:
                failed_configs.append(config_name)
                results['ablation_evaluation']['configurations'][config_name]['error'] = output
    
    # Collect and compare results
    all_results, comparison_data = collect_evaluation_results(args.evaluation_base_dir, configurations)
    
    # Create comparison report
    comparison_path = os.path.join(args.evaluation_base_dir, 'ablation_evaluation_comparison.json')
    create_comparison_report(comparison_data, comparison_path)
    
    # Save results summary
    results['ablation_evaluation']['end_time'] = datetime.now().isoformat()
    results['ablation_evaluation']['summary'] = {
        'total_configurations': len(configurations),
        'successful': len(successful_configs),
        'failed': len(failed_configs),
        'successful_configs': successful_configs,
        'failed_configs': failed_configs,
        'comparison_data_available': len(comparison_data)
    }
    
    results_path = os.path.join(args.evaluation_base_dir, 'ablation_evaluation_summary.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("🎉 Ablation Study Evaluation Completed!")
    logger.info(f"Total configurations: {len(configurations)}")
    if not args.collect_only:
        logger.info(f"Successful: {len(successful_configs)}")
        logger.info(f"Failed: {len(failed_configs)}")
    logger.info(f"Results with comparison data: {len(comparison_data)}")
    
    if successful_configs:
        logger.info("✅ Successful configurations:")
        for config in successful_configs:
            logger.info(f"  - {config}")
    
    if failed_configs:
        logger.info("❌ Failed configurations:")
        for config in failed_configs:
            logger.info(f"  - {config}")
    
    logger.info(f"Results summary saved to: {results_path}")
    logger.info(f"Comparison report saved to: {comparison_path}")
    logger.info(f"{'='*60}")
    
    # Return appropriate exit code
    return 0 if len(failed_configs) == 0 else 1

if __name__ == "__main__":
    exit(main())
