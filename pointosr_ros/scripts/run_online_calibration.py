#!/home/cerlab/miniconda3/envs/pointosr/bin/python
"""
Online Calibration Runner for PointOSR

This script runs the online calibration process before starting the classification node.
It can be used as a separate initialization step or integrated into launch files.

Usage:
    python run_online_calibration.py --model_path /path/to/model.pth --config_path /path/to/config.yaml
"""

import argparse
import logging
import sys
import os

# Add the pointosr package to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
pointosr_path = os.path.join(os.path.dirname(script_dir), '..', 'pointosr')
sys.path.insert(0, pointosr_path)

from osr.online_calibration_manager import OnlineCalibrationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run online OSR calibration')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/cerlab/Documents/data/pointosr/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--calibration_cache_dir', type=str,
                       default='src/pointosr/calib_cache',
                       help='Directory to cache calibration results')
    parser.add_argument('--k_human', type=int, default=8,
                       help='Number of human prototypes')
    parser.add_argument('--k_false', type=int, default=2,
                       help='Number of false-positive prototypes')
    parser.add_argument('--target_tpr', type=float, default=0.99,
                       help='Target True Positive Rate for ID samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # Parse known args to handle ROS-specific arguments like __name:= and __log:=
    args, unknown = parser.parse_known_args()
    
    logger.info("🚀 Starting PointOSR Online Calibration")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Cache: {args.calibration_cache_dir}")
    
    # Create calibration manager
    calibration_manager = OnlineCalibrationManager(
        model_path=args.model_path,
        config_path=args.config_path,
        data_dir=args.data_dir,
        calibration_cache_dir=args.calibration_cache_dir,
        k_human=args.k_human,
        k_false=args.k_false,
        target_tpr=args.target_tpr,
        device=args.device
    )
    
    # Run calibration
    success, fusion_config_path, stats_path, prototypes_path = calibration_manager.run_calibration()
    
    if success:
        logger.info("✅ Online calibration completed successfully!")
        logger.info(f"📁 Results cached in: {args.calibration_cache_dir}")
        logger.info(f"🔧 Fusion config: {fusion_config_path}")
        logger.info(f"📊 Stats: {stats_path}")
        logger.info(f"🎯 Prototypes: {prototypes_path}")
        logger.info("\n🎉 Ready to start classification node with OSR!")
        sys.exit(0)
    else:
        logger.error("❌ Online calibration failed!")
        logger.error("🔄 Falling back to standard classification (no OSR)")
        sys.exit(1)

if __name__ == "__main__":
    main()
