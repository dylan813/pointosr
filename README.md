# Point Cloud Open Set Recognition

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
add these later to requirements.txt:
- pip install rospkg
- pip install catkin_pkg pyyaml
- also need ros-numpy
- pip install h5py

```bash
conda install -c -y conda-forge gcc_linux-64=12 gxx_linux-64=12

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export CFLAGS="-fPIC"
export NVCC_FLAGS="--compiler-options '-fPIC' -allow-unsupported-compiler"
```

```bash
cd pointnext/cpp/pointnet2_batch
python setup.py install
```

## Train Model
```bash
cd point_osr
CUDA_VISIBLE_DEVICES=0 python pointnext/classification/main.py --cfg pointnext/cfgs/pointnext-s.yaml
```

## Evaluate Model
```bash
cd point_osr
CUDA_VISIBLE_DEVICES=0 python pointnext/classification/main.py --cfg pointnext/cfgs/pointnext-s.yaml mode=test --pretrained_path log/cfgs/.../...ckpt_best.pth
```

## Test Computational Efficiency
```bash
cd point_osr
CUDA_VISIBLE_DEVICES=0 python pointnext/eval/profile.py --cfg pointnext/cfgs/pointnext-s.yaml batch_size=128 num_points=1024 timing=True flops=True
```
