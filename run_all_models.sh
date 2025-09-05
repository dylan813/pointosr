#!/bin/bash

# PointNext Model Architecture Runner
# This script runs all the different PointNext model configurations identified from the screenshots

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration directory
CFG_DIR="pointosr/pointosr/pointnext/cfgs"
MAIN_SCRIPT="pointosr/pointosr/pointnext/classification/main.py"

# Create logs directory
mkdir -p logs

echo -e "${BLUE}=== PointNext Model Architecture Runner ===${NC}"
echo "This script will run all identified PointNext model configurations"
echo ""

# Function to create configuration file
create_config() {
    local name=$1
    local blocks=$2
    local strides=$3
    local width=$4
    local sa_layers=$5
    
    local config_file="pointosr/pointosr/pointnext/cfgs/pointnext-${name}.yaml"
    
    cat > "$config_file" << EOF
# Auto-generated configuration for ${name}
# This config inherits from default.yaml automatically

model:
  NAME: BaseCls
  encoder_args:
    NAME: PointNextEncoder
    blocks: [${blocks}]
    strides: [${strides}]
    width: ${width}
    in_channels: 9
    sa_layers: ${sa_layers}
    sa_use_res: True
    radius: 0.15
    radius_scaling: 1.5
    nsample: 32
    expansion: 4
    aggr_args:
      feature_type: 'dp_fj'
      reduction: 'max'
    group_args:
      NAME: 'ballquery'
      normalize_dp: True
    conv_args:
      order: conv-norm-act
    act_args:
      act: 'relu'
    norm_args:
      norm: 'bn'
  cls_args: 
    NAME: ClsHead
    num_classes: 2
    mlps: [512, 256]
    norm_args: 
      norm: 'bn1d'
EOF
    
    echo "$config_file"
}

# Function to run training
run_training() {
    local config_file=$1
    local model_name=$2
    local log_file="logs/${model_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${YELLOW}Running ${model_name}...${NC}"
    echo "Config: $config_file"
    echo "Log: $log_file"
    
    # Run the training (matching your actual command structure)
    CUDA_VISIBLE_DEVICES=0 python "$MAIN_SCRIPT" --cfg "$config_file" 2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ ${model_name} completed successfully${NC}"
    else
        echo -e "${RED}✗ ${model_name} failed${NC}"
        return 1
    fi
    
    echo ""
}

# Main execution
main() {
    echo -e "${BLUE}Creating model configurations...${NC}"
    
    # Model 1: 6 blocks, 6 strides, width 32, sa_layers 2 (Original)
    config1=$(create_config "s-6blocks-width32" "1, 1, 1, 1, 1, 1" "1, 2, 2, 2, 2, 1" "32" "2")
    
    # Model 2: 5 blocks, 5 strides, width 32, sa_layers 2
    config2=$(create_config "s-5blocks-width32" "1, 1, 1, 1, 1" "1, 2, 2, 2, 1" "32" "2")
    
    # Model 3: 4 blocks, 4 strides, width 32, sa_layers 2
    config3=$(create_config "s-4blocks-width32" "1, 1, 1, 1" "1, 2, 2, 1" "32" "2")

    # Model 4: 6 blocks, 6 strides, width 32, sa_layers 1
    config4=$(create_config "s-6blocks-width32-sa1" "1, 1, 1, 1, 1, 1" "1, 2, 2, 2, 2, 1" "32" "1")
    
    # Model 5: 5 blocks, 5 strides, width 32, sa_layers 1
    config5=$(create_config "s-5blocks-width32-sa1" "1, 1, 1, 1, 1" "1, 2, 2, 2, 1" "32" "1")
    
    # Model 6: 4 blocks, 4 strides, width 32, sa_layers 1
    config6=$(create_config "s-4blocks-width32-sa1" "1, 1, 1, 1" "1, 2, 2, 1" "32" "1")
    
    # Model 7: 6 blocks, 6 strides, width 24, sa_layers 2
    config7=$(create_config "s-6blocks-width24" "1, 1, 1, 1, 1, 1" "1, 2, 2, 2, 2, 1" "24" "2")
    
    # Model 8: 5 blocks, 5 strides, width 24, sa_layers 2
    config8=$(create_config "s-5blocks-width24" "1, 1, 1, 1, 1" "1, 2, 2, 2, 1" "24" "2")
    
    # Model 9: 4 blocks, 4 strides, width 24, sa_layers 2
    config9=$(create_config "s-4blocks-width24" "1, 1, 1, 1" "1, 2, 2, 1" "24" "2")

    # Model 10: 6 blocks, 6 strides, width 24, sa_layers 1
    config10=$(create_config "s-6blocks-width24-sa1" "1, 1, 1, 1, 1, 1" "1, 2, 2, 2, 2, 1" "24" "1")
    
    # Model 11: 5 blocks, 5 strides, width 24, sa_layers 1
    config11=$(create_config "s-5blocks-width24-sa1" "1, 1, 1, 1, 1" "1, 2, 2, 2, 1" "24" "1")
    
    # Model 12: 4 blocks, 4 strides, width 24, sa_layers 1
    config12=$(create_config "s-4blocks-width24-sa1" "1, 1, 1, 1" "1, 2, 2, 1" "24" "1")
    
    # Model 13: 6 blocks, 6 strides, width 16, sa_layers 2
    config13=$(create_config "s-6blocks-width16" "1, 1, 1, 1, 1, 1" "1, 2, 2, 2, 2, 1" "16" "2")
    
    # Model 14: 5 blocks, 5 strides, width 16, sa_layers 2
    config14=$(create_config "s-5blocks-width16" "1, 1, 1, 1, 1" "1, 2, 2, 2, 1" "16" "2")
    
    # Model 15: 4 blocks, 4 strides, width 16, sa_layers 2
    config15=$(create_config "s-4blocks-width16" "1, 1, 1, 1" "1, 2, 2, 1" "16" "2")
    
    # Model 16: 6 blocks, 6 strides, width 16, sa_layers 1
    config16=$(create_config "s-6blocks-width16-sa1" "1, 1, 1, 1, 1, 1" "1, 2, 2, 2, 2, 1" "16" "1")
    
    # Model 17: 5 blocks, 5 strides, width 16, sa_layers 1
    config17=$(create_config "s-5blocks-width16-sa1" "1, 1, 1, 1, 1" "1, 2, 2, 2, 1" "16" "1")
    
    # Model 18: 4 blocks, 4 strides, width 16, sa_layers 1
    config18=$(create_config "s-4blocks-width16-sa1" "1, 1, 1, 1" "1, 2, 2, 1" "16" "1")
    
    echo -e "${GREEN}Created ${#configs[@]} model configurations${NC}"
    echo ""
    
    # Array of all configurations
    configs=("$config1" "$config2" "$config3" "$config4" "$config5" "$config6" "$config7" "$config8" "$config9" "$config10" "$config11" "$config12" "$config13" "$config14" "$config15" "$config16" "$config17" "$config18")
    model_names=("s-6blocks-width32" "s-5blocks-width32" "s-4blocks-width32" "s-6blocks-width32-sa1" "s-5blocks-width32-sa1" "s-4blocks-width32-sa1" "s-6blocks-width24" "s-5blocks-width24" "s-4blocks-width24" "s-6blocks-width24-sa1" "s-5blocks-width24-sa1" "s-4blocks-width24-sa1" "s-6blocks-width16" "s-5blocks-width16" "s-4blocks-width16" "s-6blocks-width16-sa1" "s-5blocks-width16-sa1" "s-4blocks-width16-sa1")
    
    echo -e "${BLUE}Starting training for all models...${NC}"
    echo "Total models to run: ${#configs[@]}"
    echo ""
    
    # Run all models
    for i in "${!configs[@]}"; do
        echo -e "${BLUE}=== Model $((i+1))/${#configs[@]} ===${NC}"
        run_training "${configs[$i]}" "${model_names[$i]}"
        
        # Optional: Add delay between runs to avoid resource conflicts
        if [ $i -lt $((${#configs[@]}-1)) ]; then
            echo -e "${YELLOW}Waiting 10 seconds before next model...${NC}"
            sleep 10
        fi
    done
    
    echo -e "${GREEN}=== All models completed! ===${NC}"
    echo "Check the 'logs/' directory for individual training logs"
    echo "Check the 'pointosr/pointosr/pointnext/cfgs/' directory for generated configuration files"
}

# Check if main script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${RED}Error: Main training script not found at $MAIN_SCRIPT${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "$CFG_DIR" ]; then
    echo -e "${RED}Error: Configuration directory not found at $CFG_DIR${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Run the main function
main "$@"
