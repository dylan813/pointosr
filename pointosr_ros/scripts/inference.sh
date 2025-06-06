#!/bin/bash

# Path to the project's root directory
POINTOSR_DIR="/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr"
LOG_DIR="/home/cerlab/Documents/data/pointosr"

# Path to the YAML configuration file for the model
# This file will be dynamically updated with the number of clusters.
CONFIG_FILE='pointnext/cfgs/default.yaml'

# Get the directory of the script itself to locate update_val_batch_size.py
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Change to the project directory to ensure relative paths are correct
cd "$POINTOSR_DIR" || { echo "Failed to change directory to $POINTOSR_DIR. Exiting."; exit 1; }

while true; do
    echo "--- New Frame ---"
    
    # Update val_batch_size in the config file by counting active /cluster_* topics.
    echo "Attempting to update batch size based on active clusters..."
    python3 "$SCRIPT_DIR/update_val_batch_size.py" "$CONFIG_FILE"
    UPDATE_STATUS=$?

    # Check the exit code of the python script
    if [ $UPDATE_STATUS -eq 2 ]; then
        echo "No clusters found. Skipping inference for this frame."
        sleep 5 # Wait before trying again
        continue
    elif [ $UPDATE_STATUS -ne 0 ]; then
        echo "An error occurred while updating the configuration. Exiting."
        exit 1
    fi

    echo "Clusters found. Running inference..."
    # Run the inference command using the updated config
    CUDA_VISIBLE_DEVICES=0 python3 $POINTOSR_DIR/pointnext/classification/main.py --cfg $POINTOSR_DIR/pointnext/cfgs/pointnext-s.yaml mode=test --pretrained_path $LOG_DIR/pointosr_log/cfgs/bs32_vbs4/checkpoint/cfgs-train-pointnext-s-ngpus1-seed6126-20250418-012725-cJ8XQUbkLJotEiCEfg37Pm_ckpt_best.pth

    echo "Inference for this frame complete. Waiting before next frame."
    sleep 1 # Adjust as needed
done