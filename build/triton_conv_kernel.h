#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif
#ifdef __cplusplus
extern "C" {
#endif
void unload_triton_conv_kernel(void);
void load_triton_conv_kernel(void);
// tt-linker: triton_conv_kernel:CUdeviceptr input_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr output_ptr, int32_t H_in, int32_t W_in, int32_t C_in, int32_t H_out, int32_t C_out, int32_t KH, int32_t KW, int64_t stride_in_c, int64_t stride_in_h, int64_t stride_in_w, int64_t stride_w_cout, int64_t stride_out_cout, int64_t stride_out_h, int32_t pad_h, int32_t groups:256_warps4xstages3
CUresult triton_conv_kernel(CUstream stream, CUdeviceptr input_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr output_ptr, int32_t H_in, int32_t W_in, int32_t C_in, int32_t H_out, int32_t C_out, int32_t KH, int32_t KW, int64_t stride_in_c, int64_t stride_in_h, int64_t stride_in_w, int64_t stride_w_cout, int64_t stride_out_cout, int64_t stride_out_h, int32_t pad_h, int32_t groups);

#ifdef __cplusplus
}
#endif
