import os
import re
import argparse
import numpy as np
from collections import defaultdict
import random

def create_splits(id_data_dirs, ood_data_dir, output_dir, seed=42):
    """
    ID data: 70/10/10/10 = Train/Val/Calib/Test
    OOD data: 40/60 = Calib/Test
    """
    random.seed(seed)
    
    print(f"ID data directories: {id_data_dirs}")
    print(f"OOD data directory: {ood_data_dir}")
    print(f"Output directory: {output_dir}")
    
    id_class_files = defaultdict(list)
    
    for id_data_dir in id_data_dirs:
        for root, _, files in os.walk(id_data_dir):
            for file in files:
                if file.endswith('.bin'):
                    rel_path = os.path.relpath(root, id_data_dir)
                    class_name = rel_path.split(os.sep)[0] if rel_path != '.' else 'unknown'
                    id_class_files[class_name].append(os.path.join(root, file))
    
    id_train_files = []
    id_val_files = []
    id_calib_files = []
    id_test_files = []
    
    for class_name, filepaths in id_class_files.items():
        random.shuffle(filepaths)
        
        total_files = len(filepaths)
        train_size = int(total_files * 0.70)
        val_size = int(total_files * 0.10)
        calib_size = int(total_files * 0.10)
        test_size = total_files - train_size - val_size - calib_size
        
        if test_size < 0:
            calib_size += test_size
            test_size = 0
        
        class_train = filepaths[:train_size]
        class_val = filepaths[train_size:train_size + val_size]
        class_calib = filepaths[train_size + val_size:train_size + val_size + calib_size]
        class_test = filepaths[train_size + val_size + calib_size:]
        
        id_train_files.extend([os.path.basename(f) for f in class_train])
        id_val_files.extend([os.path.basename(f) for f in class_val])
        id_calib_files.extend([os.path.basename(f) for f in class_calib])
        id_test_files.extend([os.path.basename(f) for f in class_test])
        
        print(f"  Class '{class_name}' split: {len(class_train)} train, {len(class_val)} val, {len(class_calib)} calib, {len(class_test)} test")
    
    ood_filepaths = []
    
    for root, _, files in os.walk(ood_data_dir):
        for file in files:
            if file.endswith('.bin'):
                ood_filepaths.append(os.path.join(root, file))
    
    random.shuffle(ood_filepaths)
    
    total_ood = len(ood_filepaths)
    ood_calib_size = int(total_ood * 0.40)
    ood_test_size = total_ood - ood_calib_size
    
    ood_calib_filepaths = ood_filepaths[:ood_calib_size]
    ood_test_filepaths = ood_filepaths[ood_calib_size:]
    
    ood_calib_files = [os.path.basename(f) for f in ood_calib_filepaths]
    ood_test_files = [os.path.basename(f) for f in ood_test_filepaths]
    
    print(f"OOD split: {len(ood_calib_files)} calib, {len(ood_test_files)} test")
    
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    if id_train_files:
        train_file = os.path.join(output_dir, 'id_train_split.txt')
        with open(train_file, 'w') as f:
            for file in sorted(id_train_files):
                f.write(f"{file}\n")
        created_files.append(train_file)
    
    if id_val_files:
        val_file = os.path.join(output_dir, 'id_val_split.txt')
        with open(val_file, 'w') as f:
            for file in sorted(id_val_files):
                f.write(f"{file}\n")
        created_files.append(val_file)
    
    if id_calib_files:
        id_calib_file = os.path.join(output_dir, 'id_calib_split.txt')
        with open(id_calib_file, 'w') as f:
            for file in sorted(id_calib_files):
                f.write(f"{file}\n")
        created_files.append(id_calib_file)
    
    if id_test_files:
        id_test_file = os.path.join(output_dir, 'id_test_split.txt')
        with open(id_test_file, 'w') as f:
            for file in sorted(id_test_files):
                f.write(f"{file}\n")
        created_files.append(id_test_file)
    
    if ood_calib_files:
        ood_calib_file = os.path.join(output_dir, 'ood_calib_split.txt')
        with open(ood_calib_file, 'w') as f:
            for file in sorted(ood_calib_files):
                f.write(f"{file}\n")
        created_files.append(ood_calib_file)
    
    if ood_test_files:
        ood_test_file = os.path.join(output_dir, 'ood_test_split.txt')
        with open(ood_test_file, 'w') as f:
            for file in sorted(ood_test_files):
                f.write(f"{file}\n")
        created_files.append(ood_test_file)
    
    print(f"\nSplit files created:")
    for file_path in created_files:
        print(f"  - {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Create splits for ID and OOD data')
    parser.add_argument('--id_data', type=str, nargs='+', required=True)
    parser.add_argument('--ood_data', type=str, required=True)
    parser.add_argument('--output', type=str, default='splits')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    create_splits(args.id_data, args.ood_data, args.output, args.seed)

if __name__ == "__main__":
    main() 