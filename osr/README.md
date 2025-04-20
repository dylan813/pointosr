# Prototype-Based Open Set Recognition with PointNeXt

This module implements a prototype-based Open Set Recognition (OSR) system using PointNeXt feature embeddings. It's specifically configured for a 'human' (label 0) vs 'false' (label 1) classification task with open-set rejection.

## Overview

The system follows these steps:
1. Train a PointNeXt model for feature extraction (outside this module)
2. Extract normalized features from point clouds in the training set (human and false classes)
3. Compute class prototypes (mean embeddings per class: one for human, one for false)
4. Load test samples and classify them by comparing features to prototypes using cosine similarity.
5. Classify samples using per-class thresholds:
   - If similarity to human prototype exceeds `threshold_human` AND is greater than similarity to false prototype -> classify as 'human'.
   - If similarity to false prototype exceeds `threshold_false` AND is greater than similarity to human prototype -> classify as 'false'.
   - Otherwise -> classify as 'unknown'.

## Pipeline

### Step 1: Extract Features

First, extract features from the training data (containing only known classes, e.g., human=0, false=1). This now requires separate configuration files for the model architecture and the dataset setup.

```bash
python osr/extract_features.py \
    --cfg configs/pointnext-s.yaml \
    --data_cfg configs/default.yaml \
    --pretrained path/to/pretrained_model.pth \
    --save_dir data/features \
    --subset train
```

**Explanation of Arguments:**
- `--cfg`: Path to the model architecture configuration file (e.g., `pointnext-s.yaml`).
- `--data_cfg`: Path to the data/dataloader configuration file (e.g., `default.yaml`).
- `--pretrained`: Path to the pretrained model weights.
- `--save_dir`: Directory to save the output features file.
- `--subset`: Which data split to process ('train', 'val', 'test', or 'all').

This will create a `data/features/train_features.pkl` file containing feature vectors and their corresponding labels (0 or 1).

### Step 2: Build Prototypes

Next, build the 'human' and 'false' class prototypes from the extracted training features:

```bash
python osr/build_prototypes.py --features_path data/features/train_features.pkl --save_path data/prototypes.pkl
```

This will create a `data/prototypes.pkl` file containing the prototype vectors for class 0 and class 1.

### Step 3: Classification and Evaluation with OSR

You can now classify test samples and evaluate the dual-threshold OSR performance using the updated `osr_classifier.py` script. This script now handles both classification and generates key evaluation metrics and plots.

**Option A: Using Pre-extracted Features**

If you have already extracted features using `extract_features.py` (e.g., into `data/features/val_features.pkl`), you can run:

```bash
python osr/osr_classifier.py \
    --prototypes data/prototypes.pkl \
    --threshold_human 0.8 \
    --threshold_false 0.7 \
    --test_features data/features/val_features.pkl \
    --plot \
    --output_dir results_dual_thresh 
```

**Option B: Extracting Features On-the-Fly**

Alternatively, the script can extract features directly using the model. This requires providing model and data configurations:

```bash
python osr/osr_classifier.py \
    --cfg configs/pointnext-s.yaml \
    --data_cfg configs/default.yaml \
    --pretrained path/to/pretrained_model.pth \
    --prototypes data/prototypes.pkl \
    --threshold_human 0.8 \
    --threshold_false 0.7 \
    --eval_split val \
    --plot \
    --output_dir results_dual_thresh 
```

**Explanation of Arguments:**
- `--cfg`: Path to the model architecture config file (e.g., `pointnext-s.yaml`). **Required only if extracting features on the fly.**
- `--data_cfg`: Path to the data/dataloader config file (e.g., `default.yaml`). **Required only if extracting features on the fly.**
- `--pretrained`: Path to the pretrained model weights. **Required only if extracting features on the fly.**
- `--prototypes`: Path to the `prototypes.pkl` file created in Step 2. (Required)
- `--threshold_human`: The cosine similarity threshold specific to the human class (0). (Required)
- `--threshold_false`: The cosine similarity threshold specific to the false class (1). (Required)
- `--test_features`: Path to a pickle file containing pre-extracted test/validation features and labels. If provided, on-the-fly extraction arguments (`--cfg`, `--data_cfg`, `--pretrained`, `--eval_split`) are ignored.
- `--eval_split`: Which data split ('val' or 'test') to use from `--data_cfg` when extracting features on the fly. Defaults to 'val'.
- `--plot`: Flag to generate evaluation plots (similarity distributions).
- `--output_dir`: Directory to save metrics and plots.

This script will output metrics like accuracy for human samples, accuracy for false samples, and the rejection rate for true unknown samples (if present in the evaluation set). It will also save plots visualizing the similarity distributions against each prototype.

### Step 4: (Optional) Threshold Tuning

The optimal values for `--threshold_human` and `--threshold_false` should be determined using a separate validation dataset (distinct from the training and test sets). This typically involves:
1. Extracting features for the validation set (containing only human and false samples).
2. Calculating similarities of validation features to the human and false prototypes.
3. Choosing thresholds based on desired performance (e.g., retaining 95% of true positives for each class).
(A script like `tune_thresholds.py` could be created for this purpose).

### Step 5: (Removed) Comprehensive Evaluation

The functionality previously in `evaluate_osr.py` (like confusion matrix, detailed plots) has been partially integrated into `osr_classifier.py`. If more detailed, specific evaluation plots (like t-SNE or ROC curves tailored to the dual-threshold setup) are needed, the `evaluate_osr.py` script could be adapted, or new evaluation functions added.

## Adding New Classes

This specific implementation is tailored for the binary (human/false) known-set case with dual thresholds. Adding more *known* classes would require significant modification to the `OSRClassifier` logic and the evaluation metrics. The concept of adding *unknown* classes to the test set is implicitly handled by the rejection mechanism.

## Files

- `models/pointnext_wrapper.py`: Wrapper for PointNeXt model to extract features.
- `extract_features.py`: Script to extract and save features for known classes.
- `build_prototypes.py`: Script to build 'human' and 'false' prototypes from features.
- `osr_classifier.py`: Implementation of the dual-threshold OSR classifier and evaluation script.
- `evaluate_osr.py`: (Potentially outdated/redundant) Previous comprehensive evaluation script. Needs adaptation if specific plots beyond those in `osr_classifier.py` are required for the dual-threshold method.

## Requirements

- PyTorch
- NumPy
- scikit-learn
- matplotlib
- tqdm
- (Optional: seaborn, if adapting `evaluate_osr.py` for advanced plots)

## Reference

This implementation is based on the prototype approach to open set recognition, which leverages normalized feature embeddings and cosine similarity for classification. This version uses a dual-threshold mechanism specific to a binary known set. 