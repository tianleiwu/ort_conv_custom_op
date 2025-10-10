# File: conv_triton.py

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['KH', 'KW'],
)
@triton.jit
def conv_kernel_triton(
    # Pointers to Tensors
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Dimensions
    H_in, W_in, C_in,
    H_out, C_out,
    KH, KW,
    # Strides for Tensor memory layout (in elements)
    stride_in_c, stride_in_h, stride_in_w,
    stride_w_cout,
    stride_out_cout, stride_out_h,
    # Convolution Parameters
    pad_h, groups,
    # Meta-parameters (tuned by Triton)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for grouped convolution:
    Each (pid_k, pid_h) computes one output element for output channel k and output row h_out.
    """
    # --- Program IDs ---
    pid_k = tl.program_id(0)   # output channel index
    pid_h = tl.program_id(1)   # output row index (h_out)

    # --- Accumulator (use FP32) ---
    acc = tl.zeros((), dtype=tl.float32)

    # --- Compute group and input-channel base index correctly ---
    # number of output channels per group
    cout_per_group = C_out // groups        # scalar
    cin_per_group = C_in // groups         # scalar

    # group index that this output channel belongs to
    group_idx = pid_k // cout_per_group

    # input channel base for that group (we want the first input channel index of the group)
    input_channel_base = group_idx * cin_per_group

    # Input and weight base pointers (element-indexed)
    input_base_ptr = input_ptr + input_channel_base * stride_in_c
    # weight layout: we assume weights are laid out such that pid_k indexes output channel
    weight_base_ptr = weight_ptr + pid_k * stride_w_cout

    # --- Main reduction loop over KH * KW in BLOCK_SIZE chunks ---
    total_k = KH * KW

    # Use a typed arange for block offsets (int32)
    offs_start = 0
    for offs in range(0, total_k, BLOCK_SIZE):
        # vector of lane offsets (0..BLOCK_SIZE-1)
        #block_offs = offs + tl.arange(0, BLOCK_SIZE, dtype=tl.int32)
        block_offs = offs + tl.arange(0, BLOCK_SIZE)

        # compute mask: valid lanes are those with block_offs < total_k
        valid = block_offs < total_k

        # compute kernel coordinates kh and kw for each lane
        kh = block_offs // KW
        kw = block_offs % KW

        # compute input h coordinate (per lane)
        h_in = pid_h - pad_h + kh   # may be negative; filter via mask

        # mask lanes where h_in is inside [0, H_in)
        mask = valid & (h_in >= 0) & (h_in < H_in)

        # --- Build pointers for weight and input loads (element indices) ---
        w_ptrs = weight_base_ptr + block_offs                # weight elements for pid_k
        in_ptrs = input_base_ptr + h_in * stride_in_h + kw * stride_in_w

        # Load fp16 data (use other=0.0 to avoid undef lanes)
        weight_block_fp16 = tl.load(w_ptrs, mask=mask, other=0.0)
        input_block_fp16  = tl.load(in_ptrs, mask=mask, other=0.0)

        # cast to fp32 for accumulation
        weight_block = weight_block_fp16.to(tl.float32)
        input_block  = input_block_fp16.to(tl.float32)

        # fused multiply-add per lane, then reduce to scalar
        temp = tl.math.fma(input_block, weight_block, 0.0)  # vector
        acc += tl.sum(temp)                                 # scalar

    # Add bias (assume bias is per-output-channel)
    bias_val = tl.load(bias_ptr + pid_k).to(tl.float32)
    # bias_val = tl.load(bias_ptr + pid_k, other=0.0).to(tl.float32)
    acc += bias_val

    # Cast to output dtype (fp16) and write back
    out_val = acc.to(tl.float16)
    output_final_ptr = output_ptr + pid_k * stride_out_cout + pid_h * stride_out_h
    tl.store(output_final_ptr, out_val)
