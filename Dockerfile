FROM ubuntu:20.04
LABEL maintainer="PointOSR Docker"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies including CUDA in one layer
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    vim \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    software-properties-common \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-1 \
    && rm cuda-keyring_1.0-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

# Install Python and ros-numpy from source
RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install catkin_pkg \
    && pip3 install git+https://github.com/eric-wieser/ros_numpy.git \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64

# Install Miniconda and configure conda in one layer
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh \
    && ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && conda config --set channel_priority strict \
    && conda config --set auto_update_conda false \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda init bash

# Create conda environment and install packages in one layer
COPY requirements.txt /tmp/requirements.txt
RUN conda create -n pointosr -y python=3.9 numpy=1.24 numba \
    && echo "conda activate pointosr" >> ~/.bashrc \
    && echo "conda activate pointosr" >> ~/.zshrc \
    && conda run -n pointosr conda install -y pytorch=2.1.0 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia \
    && conda run -n pointosr pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    && conda run -n pointosr conda install -c conda-forge -y gcc_linux-64=12 gxx_linux-64=12

# Set compiler environment variables
ENV CC=$CONDA_DIR/envs/pointosr/bin/x86_64-conda-linux-gnu-gcc
ENV CXX=$CONDA_DIR/envs/pointosr/bin/x86_64-conda-linux-gnu-g++
ENV CUDAHOSTCXX=$CXX
ENV CFLAGS="-fPIC"
ENV NVCC_FLAGS="--compiler-options '-fPIC' -allow-unsupported-compiler"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Install Python requirements and ROS packages
RUN conda run -n pointosr pip install -r /tmp/requirements.txt \
    && conda run -n pointosr pip install rospkg catkin_pkg pyyaml

# Set working directory and copy source code
WORKDIR /workspace/pointosr_ws
COPY . /workspace/pointosr_ws/src/pointosr/

# Build the C++ extension
RUN conda run -n pointosr bash -c "cd /workspace/pointosr_ws/src/pointosr/pointosr/pointnext/cpp/pointnet2_batch && export CUDA_HOME=/usr/local/cuda-12.1 && export PATH=\$CUDA_HOME/bin:\$PATH && export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH && python setup.py install"

# Set default command to activate conda environment and keep container running
CMD ["/bin/bash"]
