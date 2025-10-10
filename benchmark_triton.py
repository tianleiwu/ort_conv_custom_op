# File: benchmark_triton.py

import torch
from conv_triton_autotune import conv_kernel_triton

def benchmark(seq_len: int, precision: str, warmup: int = 20, runs: int = 100):
    """
    Runs the benchmark using the Triton kernel and checks against PyTorch.
    """
    print(f"🚀 [Triton Kernel - {precision.upper()}] seq_len={seq_len}")
    print("-" * 50)

    # Set dtype based on precision
    if precision == 'fp32':
        torch_dtype = torch.float32
    elif precision == 'fp16':
        torch_dtype = torch.float16
    else:
        raise ValueError("Unsupported precision, use 'fp32' or 'fp16'")
        
    device = torch.device("cuda:0")

    # Define tensor dimensions and convolution parameters
    C_in, KH, KW = 198, 5, 768
    C_out = 178200
    pad_h, groups = 4, 198
    
    h_out = seq_len - KH + 1 + (2 * pad_h)
    
    # Allocate tensors on GPU
    # Input tensor shape for the operation: (N, C, H, W)
    Y_gpu = torch.randn((1, C_in, seq_len, KW), dtype=torch_dtype, device=device)
    W_gpu = torch.randn((C_out, 1, KH, KW), dtype=torch_dtype, device=device)
    B_gpu = torch.randn((C_out,), dtype=torch_dtype, device=device)
    # Output tensor for Triton is 3D (N is implicit)
    Z_gpu = torch.empty((1, C_out, h_out), dtype=torch_dtype, device=device).squeeze(0)

    # Grid for launching the kernel: one program per output pixel
    grid = (C_out, h_out)

    # --- Correctness Check ---
    print("Verifying correctness against torch.nn.functional.conv2d...")
    torch_result = torch.nn.functional.conv2d(
        Y_gpu, W_gpu, B_gpu, stride=(1, 1), padding=(pad_h, 0), groups=groups
    )
    
    # Run Triton kernel once for the check
    conv_kernel_triton[grid](
        Y_gpu, W_gpu, B_gpu, Z_gpu,
        seq_len, KW, C_in, h_out, C_out, KH, KW,
        Y_gpu.stride(1), Y_gpu.stride(2), Y_gpu.stride(3),
        W_gpu.stride(0),
        Z_gpu.stride(0), Z_gpu.stride(1),
        pad_h, groups,
    )
    
    # Compare results
    diff = (torch_result.squeeze() - Z_gpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max abs diff: {max_diff:.6f}, Mean abs diff: {mean_diff:.6f}")
    
    atol = 0.5 if precision == 'fp16' else 0.1
    rtol = 0.5 if precision == 'fp16' else 0.01

    is_correct = torch.allclose(torch_result.squeeze(), Z_gpu, atol=atol, rtol=rtol)
    if is_correct:
        print("✅ Correctness check PASSED!")
    else:
        print("❌ Correctness check FAILED!")
    
    # --- Benchmark ---
    
    # Warmup runs
    for _ in range(warmup):
        conv_kernel_triton[grid](
            Y_gpu, W_gpu, B_gpu, Z_gpu,
            seq_len, KW, C_in, h_out, C_out, KH, KW,
            Y_gpu.stride(1), Y_gpu.stride(2), Y_gpu.stride(3),
            W_gpu.stride(0),
            Z_gpu.stride(0), Z_gpu.stride(1),
            pad_h, groups,
        )
    torch.cuda.synchronize()

    # Timed benchmark runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(runs):
        conv_kernel_triton[grid](
            Y_gpu, W_gpu, B_gpu, Z_gpu,
            seq_len, KW, C_in, h_out, C_out, KH, KW,
            Y_gpu.stride(1), Y_gpu.stride(2), Y_gpu.stride(3),
            W_gpu.stride(0),
            Z_gpu.stride(0), Z_gpu.stride(1),
            pad_h, groups,
        )
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency = total_time_ms / runs
    throughput = runs / (total_time_ms / 1000)

    print(f"\nAvg Latency: {avg_latency:.3f} ms")
    print(f"Throughput: {throughput:.2f} runs/s")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    # Ensure Triton is installed: pip install triton
    benchmark(seq_len=8, precision='fp32')
    benchmark(seq_len=8, precision='fp16')
