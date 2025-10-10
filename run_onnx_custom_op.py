import torch
import onnx
import onnxruntime as ort
import subprocess
import os
import platform
import sys
import triton
import glob
import time
from collections import OrderedDict

# --- Define Project Root ---
# This makes the script runnable from anywhere
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Define path to your downloaded ONNX Runtime ---
# This should be the path to the extracted directory, e.g., 'onnxruntime-linux-x64-gpu-1.23.0'
ORT_CUSTOM_PATH = os.path.join(PROJ_ROOT, "vendor", "onnxruntime-linux-x64-gpu-1.23.0")


def aot_compile_and_prepare_kernel():
    """
    Compiles the Triton kernel, finds the hashed output files, renames them,
    and wraps both header and source in extern "C" to prevent C++ name mangling.
    """
    print("\n[Step 1] Compiling Triton kernel (AOT)...")

    kernel_file = os.path.join(PROJ_ROOT, "conv_triton.py")
    if not os.path.exists(kernel_file):
        print(f"❌ Error: Kernel file '{kernel_file}' not found.")
        return False

    # Note: The signature is hardcoded for FP16. For FP32, this and the kernel would need changes.
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
            check=True, capture_output=True, text=True,
            cwd=build_dir
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
        symbol_safe_hashed_name = hashed_function_name.replace('.', '_')
        
        print(f"   - Detected hashed function name: {hashed_function_name}")
        print(f"   - Renaming symbol '{symbol_safe_hashed_name}' to '{final_output_name}'")

        final_h_path = os.path.join(build_dir, f"{final_output_name}.h")
        final_c_path = os.path.join(build_dir, f"{final_output_name}.cpp")

        # C-style linkage guards for C++ compatibility
        c_guard_start = "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
        c_guard_end = "\n#ifdef __cplusplus\n}\n#endif\n"

        # Process Header File
        with open(hashed_h_file_path, 'r') as f:
            content = f.read()
        content = content.replace(symbol_safe_hashed_name, final_output_name)
        # Find where the declarations start (after the #endif of the include guard)
        include_guard_end = "#endif"
        pos = content.find(include_guard_end)
        if pos != -1:
            insertion_point = pos + len(include_guard_end)
            declarations = content[insertion_point:]
            declarations = declarations.lstrip('\n')
            # Reconstruct content with guards around the declarations
            content = content[:insertion_point] + c_guard_start + declarations + c_guard_end
        with open(final_h_path, 'w') as f:
            f.write(content)
        os.remove(hashed_h_file_path)

        # Process C Source File
        with open(hashed_c_file_path, 'r') as f:
            content = f.read()

        content = content.replace(symbol_safe_hashed_name, final_output_name)
        
        include_guard_end = "#include <cuda.h>"
        pos = content.find(include_guard_end)
        if pos != -1:
            insertion_point = pos + len(include_guard_end)
            declarations = content[insertion_point:]
            declarations = declarations.lstrip('\n')
            content = content[:insertion_point] + c_guard_start + declarations + c_guard_end

        content = content.replace("if(gX * gY * gZ > 0)", "")
        with open(final_c_path, 'w') as f:
            f.write(content)
        os.remove(hashed_c_file_path)

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

    cmake_args = [
        "cmake",
        "-S", PROJ_ROOT,
        "-B", build_dir,
        f"-DONNXRUNTIME_ROOT_DIR={ort_root}"
    ]
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

def create_onnx_model(precision: str):
    """Creates an ONNX model that uses the TritonConv custom op."""
    print(f"\n[Step 3] Creating ONNX model ({precision.upper()})...")

    if precision == 'fp16':
        onnx_dtype = onnx.TensorProto.FLOAT16
    elif precision == 'fp32':
        onnx_dtype = onnx.TensorProto.FLOAT
    else:
        raise ValueError("Unsupported precision, use 'fp16' or 'fp32'")

    # Define tensor shapes. Using dynamic shape for sequence length.
    Y = onnx.helper.make_tensor_value_info('Y', onnx_dtype, [1, 198, None, 768])
    W = onnx.helper.make_tensor_value_info('W', onnx_dtype, [178200, 1, 5, 768])
    B = onnx.helper.make_tensor_value_info('B', onnx_dtype, [178200])
    Z = onnx.helper.make_tensor_value_info('Z', onnx_dtype, [1, 178200, None])
    
    # Create the custom node. pad_h and groups are attributes passed to the op.
    node = onnx.helper.make_node('TritonConv', ['Y', 'W', 'B'], ['Z'], domain='com.custom.ops', pad_h=4, groups=198)
    graph = onnx.helper.make_graph([node], 'triton-conv-graph', [Y, W, B], [Z])
    model = onnx.helper.make_model(graph, producer_name='triton-aot-example')
    
    model_name = f"triton_conv_{precision}.onnx"
    model_path = os.path.join(PROJ_ROOT, "build", model_name)
    onnx.save(model, model_path)
    
    print(f"✅ ONNX model saved to: {model_path}")
    return model_path

def run_benchmark(model_path, custom_op_lib_path, seq_len: int, precision: str, warmup: int = 20, runs: int = 100):
    """Runs inference, correctness check, and benchmark for the ONNX custom op."""
    print(f"\n🚀 [ONNX Custom Op - {precision.upper()}] seq_len={seq_len}")
    print("-" * 50)
    
    # --- Setup Session and Tensors ---
    so = ort.SessionOptions()
    so.register_custom_ops_library(custom_op_lib_path)
    providers = [('CUDAExecutionProvider', {'device_id': 0})]
    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    
    if precision == 'fp16':
        torch_dtype = torch.float16
    elif precision == 'fp32':
        torch_dtype = torch.float32
    else:
        raise ValueError("Unsupported precision, use 'fp16' or 'fp32'")

    device = torch.device("cuda:0")

    # Define tensor dimensions and convolution parameters
    C_in, KH, KW = 198, 5, 768
    C_out = 178200
    pad_h, groups = 4, 198
    
    Y_torch = torch.randn((1, C_in, seq_len, KW), dtype=torch_dtype, device=device)
    W_torch = torch.randn((C_out, 1, KH, KW), dtype=torch_dtype, device=device)
    B_torch = torch.randn((C_out,), dtype=torch_dtype, device=device)
    
    inputs = {
        'Y': Y_torch.cpu().numpy(),
        'W': W_torch.cpu().numpy(),
        'B': B_torch.cpu().numpy(),
    }

    # --- Correctness Check ---
    print("Verifying correctness against torch.nn.functional.conv2d...")
    torch_result = torch.nn.functional.conv2d(
        Y_torch, W_torch, B_torch, stride=(1, 1), padding=(pad_h, 0), groups=groups
    )
    
    # Run ONNX model once for the check
    result_ort = session.run(['Z'], inputs)
    output_ort_torch = torch.from_numpy(result_ort[0]).to(device)

    # Compare results. Squeeze the trailing dimension of 1 from PyTorch's output.
    torch_result_squeezed = torch_result.squeeze(-1)
    diff = (torch_result_squeezed - output_ort_torch).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max abs diff: {max_diff:.6f}, Mean abs diff: {mean_diff:.6f}")
    
    atol = 0.5 if precision == 'fp16' else 0.1
    rtol = 0.5 if precision == 'fp16' else 0.01

    is_correct = torch.allclose(torch_result_squeezed, output_ort_torch, atol=atol, rtol=rtol)
    if is_correct:
        print("✅ Correctness check PASSED!")
    else:
        print("❌ Correctness check FAILED!")
        print(f"   - PyTorch result shape (squeezed): {torch_result_squeezed.shape}")
        print(f"   - ONNX result shape: {output_ort_torch.shape}")
        
    # --- Benchmark ---
    print("\nBenchmarking ONNX custom op...")
    
    # Warmup runs
    for _ in range(warmup):
        session.run(['Z'], inputs)

    # Use torch.cuda.synchronize to ensure accurate timing of host-side calls
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(runs):
        session.run(['Z'], inputs)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time_ms = (end_time - start_time) * 1000
    avg_latency = total_time_ms / runs
    throughput = runs / (total_time_ms / 1000)

    print(f"Avg Latency: {avg_latency:.3f} ms (includes H2D/D2H copy)")
    print(f"Throughput: {throughput:.2f} runs/s")
    print("-" * 50 + "\n")

def main():
    if not aot_compile_and_prepare_kernel():
        return
    if not build_custom_op():
        return

    # Find the compiled custom op library
    lib_name = "libtriton_conv_op.so"
    if platform.system() == "Windows": lib_name = "triton_conv_op.dll"
    elif platform.system() == "Darwin": lib_name = "libtriton_conv_op.dylib"
    
    custom_op_lib_path = os.path.join(PROJ_ROOT, "build", lib_name)
    if not os.path.exists(custom_op_lib_path):
       print(f"❌ Custom op library not found at {custom_op_lib_path}")
       return
          
    # The AOT compilation is hardcoded for FP16, so we only run the FP16 benchmark.
    # To run for FP32, the Triton kernel and AOT signature would need to be modified.
    model_path = create_onnx_model(precision='fp16')
    run_benchmark(model_path, custom_op_lib_path, seq_len=8, precision='fp16')

if __name__ == '__main__':
    main()
