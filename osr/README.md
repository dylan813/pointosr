# Prototype-Based Open Set Recognition with PointNeXt

This module implements a prototype-based Open Set Recognition (OSR) system using PointNeXt feature embeddings.

## Overview

The system follows these steps:
1. Train a PointNeXt model for feature extraction
2. Extract normalized features from point clouds in the training set
3. Compute class prototypes (mean embeddings per class)
4. Classify test samples by comparing to prototypes using cosine similarity
5. Reject samples as "unknown" if similarity falls below a threshold

## Pipeline

### Step 1: Extract Features

First, extract features from the training data:

```bash
python osr/extract_features.py --cfg configs/your_config.py --pretrained path/to/pretrained_model.pth --save_dir data/features --subset train
```

This will create a `data/features/train_features.pkl` file.

### Step 2: Build Prototypes

Next, build class prototypes from the extracted features:

```bash
python osr/build_prototypes.py --features_path data/features/train_features.pkl --save_path data/prototypes.pkl
```

This will create a `data/prototypes.pkl` file containing the class prototypes.

### Step 3: Classification with OSR

You can now classify test samples using the OSR classifier:

```bash
python osr/osr_classifier.py --cfg configs/your_config.py --prototypes data/prototypes.pkl --threshold 0.75 --test_features data/features/test_features.pkl --plot --output_dir results
```

This will classify the test samples and generate evaluation metrics.

### Step 4: Comprehensive Evaluation

For a more comprehensive evaluation, including visualizations and analysis:

```bash
python osr/evaluate_osr.py --train_features data/features/train_features.pkl --test_features data/features/test_features.pkl --prototypes data/prototypes.pkl --known_classes 0 1 2 3 4 --threshold 0.75 --output_dir results
```

This will generate various visualizations and metrics to help evaluate the OSR system.

## Adding New Classes

To add new classes to the system:

1. Extract features for the new classes:
   ```bash
   python osr/extract_features.py --cfg configs/your_config.py --pretrained path/to/pretrained_model.pth --save_dir data/features_new --subset train
   ```

2. Merge new features with existing ones (or just rebuild prototypes from all available data)

3. Rebuild the prototypes:
   ```bash
   python osr/build_prototypes.py --features_path data/merged_features.pkl --save_path data/new_prototypes.pkl
   ```

4. Use the updated prototypes for classification.

## Files

- `models/pointnext_wrapper.py`: Wrapper for PointNeXt model to extract features
- `extract_features.py`: Script to extract and save features
- `build_prototypes.py`: Script to build class prototypes
- `osr_classifier.py`: Implementation of the OSR classifier
- `evaluate_osr.py`: Comprehensive evaluation script

## Requirements

- PyTorch
- NumPy
- scikit-learn
- matplotlib
- tqdm
- seaborn

## Reference

This implementation is based on the prototype approach to open set recognition, which leverages normalized feature embeddings and cosine similarity for classification. 