import os
import json
import numpy as np
import glob
from pathlib import Path

def label_human_clusters(input_dir, output_file):
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    
    labels = {}
    for bin_file in bin_files:
        file_name = os.path.basename(bin_file)
        labels[file_name] = "human"
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=4)
    
    print(f"Labeled {len(bin_files)} files as 'human' and saved to {output_file}")
    return labels

if __name__ == "__main__":
    input_directory = "data/human/human_clusters"
    output_file_path = "data/human/labels.json"
    
    labels = label_human_clusters(input_directory, output_file_path)
    
    print(f"Total files labeled: {len(labels)}")
