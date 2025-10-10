mkdir -p vendor
cd vendor
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-gpu-1.23.0.tgz
tar -zxvf onnxruntime-linux-x64-gpu-1.23.0.tgz
cd ..
pip install triton==3.3.0
export PATH=/home/tlwu/cuda12.8/bin:$PATH
export CUDNN_PATH=
python run_onnx_custom_op.py