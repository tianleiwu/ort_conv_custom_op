#!/bin/bash
set -e

ORT_DIR="vendor/onnxruntime-linux-x64-gpu-1.23.0"
if [ ! -d "$ORT_DIR" ]; then
    echo "ONNX Runtime not found. Downloading and extracting..."
    mkdir -p vendor
    wget -P vendor https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-gpu-1.23.0.tgz
    tar -zxvf vendor/onnxruntime-linux-x64-gpu-1.23.0.tgz -C vendor/
    rm vendor/onnxruntime-linux-x64-gpu-1.23.0.tgz
else
    echo "ONNX Runtime already exists. Skipping download."
fi

pip install triton==3.4.0
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8

export CUDNN_HOME=/home/tlwu/cudnn9.14_cuda12
if [ ! -f "$CUDNN_HOME/include/cudnn.h" ]; then
    echo "cuDNN not found. Downloading and extracting..."
    mkdir -p "$CUDNN_HOME"
    wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz
    tar -xvf cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz -C "$CUDNN_HOME" --strip-components=1
    rm cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz
else
    echo "cuDNN already exists. Skipping download."
fi

python run_onnx_custom_op.py