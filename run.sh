# mkdir -p vendor
# cd vendor
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-gpu-1.23.0.tgz
# tar -zxvf onnxruntime-linux-x64-gpu-1.23.0.tgz
# cd ..
# pip install triton==3.3.0
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8

# mkdir -p /home/tlwu/cudnn9.14_cuda12
# wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz
# tar -xvf cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz -C /home/tlwu/cudnn9.14_cuda12
export CUDNN_HOME=/home/tlwu/cudnn9.14_cuda12
python run_onnx_custom_op.py