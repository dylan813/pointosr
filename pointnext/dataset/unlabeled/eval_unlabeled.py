import os
import torch
import numpy as np
import argparse
import yaml
import logging
from tqdm import tqdm
import sys
import csv

sys.path.append('.')
from pointnext.utils import EasyConfig, setup_logger_dist
from pointnext.model import build_model_from_cfg
from pointnext.dataset.unlabeled.unlabeled import UnlabeledTestDataset
from pointnext.dataset.build import build_dataloader_from_cfg
from pointnext.transforms import build_transforms_from_cfg

def parse_args():
    parser = argparse.ArgumentParser('Model Evaluation')
    parser.add_argument('--cfg', type=str, default='pointnext/cfgs/pointnext-s.yaml', help='config file')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--data_path', type=str, default='data/cluster_data', help='path to test data')
    parser.add_argument('--output', type=str, default='log/unlabeled_predictions/predictions.csv', help='output file')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger("inference")
    
    model = build_model_from_cfg(cfg.model)
    model.cuda()
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    model.eval()

    test_dataset = UnlabeledTestDataset(
        data_root=args.data_path,
        num_points=cfg.dataset.get('num_points', 2048),
        uniform_sample=cfg.dataset.get('uniform_sample', True)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )
    
    class_names = cfg.dataset.get('classes', None)
    if class_names is None:
        logger.info("No class names found in config, using human/misc as class labels")
        class_names = ['human', 'misc']
    
    predictions = []
    with torch.no_grad():
        for data_dict in tqdm(test_loader, desc="Inference"):
            pos = data_dict['pos'].cuda()
            features = data_dict['x'].cuda()
            filenames = data_dict['filename']
            
            features_transposed = features.transpose(1, 2).contiguous()
            
            input_dict = {
                'pos': pos,
                'x': features_transposed
            }
            
            logits = model(input_dict)
            
            preds = torch.argmax(logits, dim=1)
            
            for i, filename in enumerate(filenames):
                pred_idx = preds[i].item()
                if pred_idx < len(class_names):
                    pred_class = class_names[pred_idx]
                else:
                    logger.warning(f"Predicted class index {pred_idx} is out of range. Using 'unknown' as class name.")
                    pred_class = "unknown"
                predictions.append((filename, pred_class))
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Predicted Class'])
        writer.writerows(predictions)
    
    logger.info(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main() 