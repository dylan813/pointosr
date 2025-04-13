# Point Cloud Open Set Recognition

This repository implements open set recognition methods for 3D point cloud data.

## Installation

```bash
git clone https://github.com/dylan813/point_osr.git
cd point_osr
```

```bash
conda create -n pointosr -y python=3.9 numpy=1.24 numba
conda activate pointosr
```

```bash
conda install -y pytorch=2.1.0 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install -r requirements.txt
```

```bash
cd cpp/pointnet2_batch
python setup.py install
```

## Prototype-Based Open Set Recognition

This project includes a prototype-based open set recognition system using PointNeXt features.

The system computes class prototypes (mean feature vectors) from a trained PointNeXt model and classifies test samples by comparing their features to these prototypes using cosine similarity. Samples with low similarity to all known prototypes are rejected as "unknown".

### Usage

See the [OSR module README](osr/README.md) for detailed usage instructions.

### Key Features

- Extract normalized features from point clouds
- Build class prototypes from training data 
- Classify test samples with cosine similarity
- Reject unknown samples using a threshold
- Easy to add new classes (just compute new prototypes)
- Comprehensive evaluation tools and visualizations

### Pipeline

1. Extract features from training samples
2. Compute class prototypes
3. Classify test samples with similarity threshold
4. Evaluate performance with various metrics

### Getting Started

```bash
# Extract features
python osr/extract_features.py --cfg configs/your_config.py --pretrained path/to/model.pth --save_dir data/features

# Build prototypes
python osr/build_prototypes.py --features_path data/features/train_features.pkl --save_path data/prototypes.pkl

# Classify and evaluate
python osr/evaluate_osr.py --train_features data/features/train_features.pkl --test_features data/features/test_features.pkl --prototypes data/prototypes.pkl --known_classes 0 1 2 3 4 --threshold 0.75
```