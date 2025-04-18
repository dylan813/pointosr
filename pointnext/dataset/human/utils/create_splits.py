import os
import re
import argparse
import numpy as np
from collections import defaultdict
import random

def create_splits(data_dirs, output_dir, train_ratio=0.65, val_ratio=0.175, test_ratio=0.175, seed=42):
    random.seed(seed) 
    
    all_train_files = []
    all_val_files = []
    all_test_files = []
    
    total_bin_files_processed = 0

    print(f"Processing data directories: {data_dirs}")
    for data_dir in data_dirs:
        print(f"\n--- Processing directory: {data_dir} ---")
        
        local_bin_filepaths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.bin'):
                    local_bin_filepaths.append(os.path.join(root, file))
        
        local_file_count = len(local_bin_filepaths)
        
        if local_file_count == 0:
            print(f"Warning: No .bin files found in {data_dir}. Skipping.")
            continue

        print(f"Found {local_file_count} .bin files in {data_dir}.")
        total_bin_files_processed += local_file_count

        random.shuffle(local_bin_filepaths)
        
        local_train_size = int(local_file_count * train_ratio)
        local_val_size = int(local_file_count * val_ratio)
        local_test_size = local_file_count - local_train_size - local_val_size
        
        dir_train_filepaths = local_bin_filepaths[:local_train_size]
        dir_val_filepaths = local_bin_filepaths[local_train_size : local_train_size + local_val_size]
        dir_test_filepaths = local_bin_filepaths[local_train_size + local_val_size:]
        
        dir_train_files = [os.path.basename(f) for f in dir_train_filepaths]
        dir_val_files = [os.path.basename(f) for f in dir_val_filepaths]
        dir_test_files = [os.path.basename(f) for f in dir_test_filepaths]

        all_train_files.extend(dir_train_files)
        all_val_files.extend(dir_val_files)
        all_test_files.extend(dir_test_files)
        
        print(f"Directory Split: {len(dir_train_files)} train, {len(dir_val_files)} val, {len(dir_test_files)} test files")

    print(f"\n--- Aggregated Results ---")
    
    if total_bin_files_processed == 0:
        print("Error: No .bin files found in any specified directory. No split files created.")
        return
        
    all_train_files.sort()
    all_val_files.sort()
    all_test_files.sort()

    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train_split.txt')
    val_file = os.path.join(output_dir, 'val_split.txt')
    test_file = os.path.join(output_dir, 'test_split.txt')
    
    with open(train_file, 'w') as f:
        for file in all_train_files:
            f.write(f"{file}\n")
            
    with open(val_file, 'w') as f:
        for file in all_val_files:
            f.write(f"{file}\n")
            
    with open(test_file, 'w') as f:
        for file in all_test_files:
             f.write(f"{file}\n")

    total_files_in_splits = len(all_train_files) + len(all_val_files) + len(all_test_files)
    
    print(f"Total .bin files processed across all directories: {total_bin_files_processed}")
    print(f"Total files written to split lists: {total_files_in_splits}")

    if total_files_in_splits != total_bin_files_processed:
         print(f"Warning: Mismatch between total files processed ({total_bin_files_processed}) and files written to splits ({total_files_in_splits}).")

    print(f"\n--- Final File Split Statistics ---")
    if total_files_in_splits > 0:
        print(f"Train files: {len(all_train_files)} ({len(all_train_files)/total_files_in_splits:.2%})")
        print(f"Val files: {len(all_val_files)} ({len(all_val_files)/total_files_in_splits:.2%})")
        print(f"Test files: {len(all_test_files)} ({len(all_test_files)/total_files_in_splits:.2%})")
    else:
        print("No files were assigned to splits.")

    print(f"\nSplit files created at:")
    print(f"- {train_file}")
    print(f"- {val_file}")
    print(f"- {test_file}")

def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits by processing .bin files within each directory independently.')
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True, help='Paths to the directories containing .bin files (specify one or more)')
    parser.add_argument('--output_dir', type=str, default='splits', help='Path to save the split files (default: splits)')
    parser.add_argument('--train_ratio', type=float, default=0.65, help='Ratio of files for training per directory (default: 0.65)')
    parser.add_argument('--val_ratio', type=float, default=0.175, help='Ratio of files for validation per directory (default: 0.175)')
    parser.add_argument('--test_ratio', type=float, default=0.175, help='Ratio of files for testing per directory (default: 0.175)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        print(f"Warning: The sum of ratios ({total_ratio:.4f}) is not 1.0. Normalizing ratios...")
        norm_factor = total_ratio
        args.train_ratio /= norm_factor
        args.val_ratio /= norm_factor
        args.test_ratio /= norm_factor
        print(f"Normalized ratios: Train={args.train_ratio:.4f}, Val={args.val_ratio:.4f}, Test={args.test_ratio:.4f}")
    
    create_splits(
        args.data_dirs,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

if __name__ == "__main__":
    main() 