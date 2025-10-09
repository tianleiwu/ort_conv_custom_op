import torch
import triton
import os
import sys
import subprocess
from collections import OrderedDict

# --- Your kernel file must exist (e.g., conv_triton.py) ---
KERNEL_FILE = "conv_triton.py"

def main():
    if not os.path.exists(KERNEL_FILE):
        print(f"Error: Kernel file '{KERNEL_FILE}' not found.")
        return

    print("🚀 Starting Triton AOT compilation via command-line...")

    # --- Define kernel argument types ---
    signature = OrderedDict({
        "input_ptr": "*fp16",
        "weight_ptr": "*fp16",
        "bias_ptr": "*fp16",
        "output_ptr": "*fp16",
        "H_in": "i32",
        "W_in": "i32",
        "C_in": "i32",
        "H_out": "i32",
        "C_out": "i32",
        "KH": "i32",
        "KW": "i32",
        "stride_in_c": "i64",
        "stride_in_h": "i64",
        "stride_in_w": "i64",
        "stride_w_cout": "i64",
        "stride_out_cout": "i64",
        "stride_out_h": "i64",
        "pad_h": "i32",
        "groups": "i32",
    })

    # --- Define constexpr values ---
    constants = {
        "BLOCK_SIZE": 256
    }

    sig_str = ", ".join(signature.values())
    for val in constants.values():
        sig_str += f", {val}"

    build_dir = "build"
    output_name = "triton_conv_kernel"
    os.makedirs(build_dir, exist_ok=True)

    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    try:
        # Use the absolute path for the kernel file in case CWD changes
        kernel_abs_path = os.path.abspath(KERNEL_FILE)

        subprocess.run(
            [
                sys.executable,
                compiler_path,
                kernel_abs_path, # Use absolute path here
                "--kernel-name", "conv_kernel_triton",
                "--signature", sig_str,
                "--out-name", output_name,
                # --- CHANGE 1: Set --out-path to the desired filename ---
                "--out-path", output_name,
                "--num-warps", "4",
                "--grid", "C_out, H_out, 1"
            ],
            check=True,
            capture_output=True,
            text=True,
            # --- CHANGE 2: Run the command from the build directory ---
            cwd=build_dir
        )

        header_path = os.path.join(build_dir, f"{output_name}.h")
        source_path = os.path.join(build_dir, f"{output_name}.cpp")

        print(f"✅ AOT compilation successful!")
        print(f"   - Header: {header_path}")
        print(f"   - Source: {source_path}")

    except subprocess.CalledProcessError as e:
        print("❌ AOT compilation failed.")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)

if __name__ == "__main__":
    main()