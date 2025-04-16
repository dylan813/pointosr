import os
import re
import argparse
import numpy as np
from collections import defaultdict
import random

def extract_frame_number(filename):
    match = re.search(r'cluster_frame_(\d+)_cluster_\d+\.bin', filename)
    if match:
        return int(match.group(1))
    return 0

def group_by_frame(files):
    frames = defaultdict(list)
    for file_path in files:
        filename = os.path.basename(file_path)
        frame_num = extract_frame_number(filename)
        frames[frame_num].append(file_path)
    
    return frames

def create_splits(data_dir, output_dir, train_ratio=0.65, val_ratio=0.175, test_ratio=0.175, seed=42):
    random.seed(seed)
    
    bin_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    
    frames_dict = group_by_frame(bin_files)
    
    frame_numbers = sorted(frames_dict.keys())
    total_frames = len(frame_numbers)
    
    random.shuffle(frame_numbers)
    
    train_size = int(total_frames * train_ratio)
    val_size = int(total_frames * val_ratio)
    test_size = total_frames - train_size - val_size
    
    train_frames = frame_numbers[:train_size]
    val_frames = frame_numbers[train_size:train_size + val_size]
    test_frames = frame_numbers[train_size + val_size:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train_split.txt')
    val_file = os.path.join(output_dir, 'val_split.txt')
    test_file = os.path.join(output_dir, 'test_split.txt')
    
    with open(train_file, 'w') as f:
        for frame in sorted(train_frames):
            files = [os.path.basename(file) for file in frames_dict[frame]]
            for file in sorted(files):
                f.write(f"{file}\n")
    
    with open(val_file, 'w') as f:
        for frame in sorted(val_frames):
            files = [os.path.basename(file) for file in frames_dict[frame]]
            for file in sorted(files):
                f.write(f"{file}\n")
    
    with open(test_file, 'w') as f:
        for frame in sorted(test_frames):
            files = [os.path.basename(file) for file in frames_dict[frame]]
            for file in sorted(files):
                f.write(f"{file}\n")
    
    print(f"Total frames: {total_frames}")
    print(f"Train frames: {len(train_frames)} ({len(train_frames)/total_frames:.2%})")
    print(f"Val frames: {len(val_frames)} ({len(val_frames)/total_frames:.2%})")
    print(f"Test frames: {len(test_frames)} ({len(test_frames)/total_frames:.2%})")
    
    train_files = sum(len(frames_dict[frame]) for frame in train_frames)
    val_files = sum(len(frames_dict[frame]) for frame in val_frames)
    test_files = sum(len(frames_dict[frame]) for frame in test_frames)
    total_files = train_files + val_files + test_files
    
    print(f"Total cluster files: {total_files}")
    print(f"Train files: {train_files} ({train_files/total_files:.2%})")
    print(f"Val files: {val_files} ({val_files/total_files:.2%})")
    print(f"Test files: {test_files} ({test_files/total_files:.2%})")
    
    print(f"\nSplit files created at:")
    print(f"- {train_file}")
    print(f"- {val_file}")
    print(f"- {test_file}")

def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits for human point cloud data')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing human cluster bin files')
    parser.add_argument('--output_dir', type=str, default='splits', help='Path to save the split files')
    parser.add_argument('--train_ratio', type=float, default=0.65, help='Ratio of data for training (default: 0.65)')
    parser.add_argument('--val_ratio', type=float, default=0.175, help='Ratio of data for validation (default: 0.175)')
    parser.add_argument('--test_ratio', type=float, default=0.175, help='Ratio of data for testing (default: 0.175)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        print(f"Warning: The sum of ratios ({total_ratio}) is not 1.0. Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    create_splits(
        args.data_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

if __name__ == "__main__":
    main() 