import torch
import numpy as np
import onnx
import onnxruntime as ort
import subprocess
import os
import platform
import sys
import triton
import re
import glob
from collections import OrderedDict

# --- Define Project Root ---
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Define path to your downloaded ONNX Runtime ---
ORT_CUSTOM_PATH = os.path.join(PROJ_ROOT, "vendor", "onnxruntime-linux-x64-gpu-1.23.0")
# -------------------------

def aot_compile_and_prepare_kernel():
    """
    Compiles the Triton kernel, finds the hashed output files, renames them,
    and replaces the hashed function names inside the files with clean names.
    """
    print("\n[Step 1] Compiling Triton kernel (AOT)...")
    kernel_file = os.path.join(PROJ_ROOT, "conv_triton.py")
    if not os.path.exists(kernel_file):
        print(f"❌ Error: Kernel file '{kernel_file}' not found.")
        return False

    signature = OrderedDict([
        ("input_ptr", "*fp16"), ("weight_ptr", "*fp16"), ("bias_ptr", "*fp16"),
        ("output_ptr", "*fp16"), ("H_in", "i32"), ("W_in", "i32"), ("C_in", "i32"),
        ("H_out", "i32"), ("C_out", "i32"), ("KH", "i32"), ("KW", "i32"),
        ("stride_in_c", "i64"), ("stride_in_h", "i64"), ("stride_in_w", "i64"),
        ("stride_w_cout", "i64"), ("stride_out_cout", "i64"), ("stride_out_h", "i64"),
        ("pad_h", "i32"), ("groups", "i32"),
    ])
    constants = {"BLOCK_SIZE": 256}
    sig_str = ", ".join(signature.values()) + ", " + str(constants["BLOCK_SIZE"])

    build_dir = os.path.join(PROJ_ROOT, "build")
    temp_output_name = "triton_conv_kernel"
    final_output_name = "triton_conv_kernel"
    os.makedirs(build_dir, exist_ok=True)

    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    try:
        subprocess.run(
            [
                sys.executable, compiler_path, kernel_file,
                "--kernel-name", "conv_kernel_triton",
                "--signature", sig_str,
                "--out-name", temp_output_name,
                "--num-warps", "4",
                "--grid", "C_out, H_out, 1"
            ],
            check=True, capture_output=True, text=True, cwd=build_dir
        )
    except subprocess.CalledProcessError as e:
        print("❌ Triton AOT compilation failed.")
        print("--- STDERR ---")
        print(e.stderr)
        return False

    try:
        print("   - Finding, renaming, and cleaning generated files...")
        search_pattern = os.path.join(build_dir, f"{temp_output_name}.*.c")
        hashed_c_files = glob.glob(search_pattern)
        if not hashed_c_files:
            raise FileNotFoundError(f"Could not find compiled C file with pattern: {search_pattern}")

        hashed_c_file_path = hashed_c_files[0]
        base_path, _ = os.path.splitext(hashed_c_file_path)
        hashed_h_file_path = base_path + ".h"
        hashed_function_name = os.path.splitext(os.path.basename(hashed_c_file_path))[0]
        
        print(f"   - Detected hashed function name: {hashed_function_name}")
        print(f"   - Renaming to: {final_output_name}")
        
        final_h_path = os.path.join(build_dir, f"{final_output_name}.h")
        final_c_path = os.path.join(build_dir, f"{final_output_name}.cpp")

        for path in [hashed_h_file_path, hashed_c_file_path]:
            with open(path, 'r') as f:
                content = f.read()
            # The replace logic needs to handle potential '.' in the hash
            content = content.replace(hashed_function_name.replace(".", "_"), final_output_name)
            new_path = final_h_path if path.endswith(".h") else final_c_path
            with open(new_path, 'w') as f:
                f.write(content)
            os.remove(path)

        print(f"✅ Triton kernel AOT compiled and cleaned successfully.")
        return True

    except Exception as e:
        print(f"❌ Error processing generated Triton files: {e}")
        return False


def build_custom_op():
    """Compiles the C++ custom operator using CMake."""
    print("\n[Step 2] Building C++ custom operator...")
    build_dir = os.path.join(PROJ_ROOT, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    ort_root = ORT_CUSTOM_PATH
    print(f"   - Using ONNX Runtime from: {ort_root}")

    cmake_args = ["cmake", "-S", PROJ_ROOT, "-B", build_dir, f"-DONNXRUNTIME_ROOT_DIR={ort_root}"]
    try:
        print(f"   - Running CMake command: {' '.join(cmake_args)}")
        subprocess.run(cmake_args, check=True, capture_output=True, text=True)
        
        build_command = ["cmake", "--build", build_dir]
        print(f"   - Running Build command: {' '.join(build_command)}")
        subprocess.run(build_command, check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed.")
        print("--- CMAKE STDERR ---")
        print(e.stderr)
        return False
    print("✅ Custom operator built successfully!")
    return True


def create_onnx_model():
    """Creates an ONNX model that uses the TritonConv custom op."""
    print("\n[Step 3] Creating ONNX model...")
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT16, [1, 198, None, 768])
    W = onnx.helper.make_tensor_value_info('W', onnx.TensorProto.FLOAT16, [178200, 1, 5, 768])
    B = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT16, [178200])
    Z = onnx.helper.make_tensor_value_info('Z', onnx.TensorProto.FLOAT16, [1, 178200, None])
    node = onnx.helper.make_node('TritonConv', ['Y', 'W', 'B'], ['Z'], domain='com.custom.ops', pad_h=4, groups=198)
    graph = onnx.helper.make_graph([node], 'triton-conv-graph', [Y, W, B], [Z])
    model = onnx.helper.make_model(graph, producer_name='triton-aot-example')
    model_path = os.path.join(PROJ_ROOT, "build", "triton_conv.onnx")
    onnx.save(model, model_path)
    print(f"✅ ONNX model saved to: {model_path}")
    return model_path

def run_inference(model_path, custom_op_lib_path):
    """Runs inference using the ONNX model and the custom operator."""
    print("\n[Step 4] Running inference...")
    so = ort.SessionOptions()
    so.register_custom_ops_library(custom_op_lib_path)
    providers = [('CUDAExecutionProvider', {'device_id': 0})]
    print(f"   - Loading session with custom op library: {custom_op_lib_path}")
    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    
    seq_len = 8
    C_in, KH, KW = 198, 5, 768
    C_out = 178200
    
    Y_torch = torch.randn((1, C_in, seq_len, KW), dtype=torch.float16, device='cuda')
    W_torch = torch.randn((C_out, 1, KH, KW), dtype=torch.float16, device='cuda')
    B_torch = torch.randn((C_out,), dtype=torch.float16, device='cuda')
    inputs = {
        'Y': Y_torch.cpu().numpy(),
        'W': W_torch.cpu().numpy(),
        'B': B_torch.cpu().numpy(),
    }
    print("   - Executing model...")
    result = session.run(['Z'], inputs)
    output_ort = result[0]
    print("✅ Inference successful!")
    print(f"   - Output shape: {output_ort.shape}")
    print(f"   - Output dtype: {output_ort.dtype}")

def main():
    if not aot_compile_and_prepare_kernel():
        return
    if not build_custom_op():
        return

    model_path = create_onnx_model()
    
    lib_name = "libtriton_conv_op.so"
    if platform.system() == "Windows": lib_name = "triton_conv_op.dll"
    elif platform.system() == "Darwin": lib_name = "libtriton_conv_op.dylib"
    
    custom_op_lib_path = os.path.join(PROJ_ROOT, "build", lib_name)
    if not os.path.exists(custom_op_lib_path):
       print(f"❌ Custom op library not found at {custom_op_lib_path}")
       return
          
    run_inference(model_path, custom_op_lib_path)

if __name__ == '__main__':
    main()

