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

## Train Model
```bash
cd point_osr
CUDA_VISIBLE_DEVICES=0 python pointnext/classification/main.py --cfg pointnext/cfgs/pointnext-s.yaml
```