#!/bin/bash

# Base path to the eval_dynamic_batch directory
BASE_PATH='/home/cerlab/Documents/data/pointosr/eval_dynamic_batch'

# Path to the YAML configuration file
CONFIG_FILE='pointnext/cfgs/default.yaml'

# Iterate over each directory in the base path
for folder in $BASE_PATH/*; do
    if [ -d "$folder" ]; then
        # Count the number of files in the directory
        num_files=$(find "$folder" -type f | wc -l)

        # Update the val_batch_size in the YAML configuration file
        yq eval ".val_batch_size = $num_files" -i $CONFIG_FILE

        # Run the inference command
        CUDA_VISIBLE_DEVICES=0 python pointnext/classification/main.py --cfg pointnext/cfgs/pointnext-s.yaml mode=test --pretrained_path log/cfgs/.../...ckpt_best.pth
    fi
done 