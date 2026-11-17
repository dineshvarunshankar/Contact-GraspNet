ARG BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    --index-url ${PYTORCH_INDEX}

RUN pip install --no-cache-dir \
    "numpy>=1.24,<2" \
    "scipy>=1.10" \
    "trimesh>=3.20" \
    "pyyaml>=6.0" \
    "tqdm>=4.64" \
    "pillow>=9.0" \
    "matplotlib>=3.7" \
    "open3d>=0.17.0" \
    "plotly>=5.18.0" \
    "transforms3d>=0.4.1" \
    "opencv-python>=4.8.0,<4.12" \
    "h5py>=3.8.0" \
    "pyrender>=0.1.45"

WORKDIR /workspace
COPY . /workspace

COPY contact_graspnet_pytorch /opt/cgn
# Compatibility with PyTorch 2.6+ and batched direction vectors.
RUN sed -i 's/torch.load(filename)/torch.load(filename, map_location="cpu", weights_only=False)/g' \
       /opt/cgn/contact_graspnet_pytorch/checkpoints.py \
    && sed -i 's/torch\.cross(/torch.linalg.cross(/g' \
       /opt/cgn/contact_graspnet_pytorch/contact_graspnet.py \
    && pip install --no-cache-dir -e /opt/cgn
ENV PYTHONPATH="/opt/cgn/Pointnet_Pointnet2_pytorch:/opt/cgn/contact_graspnet_pytorch"

ENV PYOPENGL_PLATFORM=egl
ENV MPLCONFIGDIR=/tmp

CMD ["/bin/bash"]
