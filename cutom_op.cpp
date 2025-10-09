#include <cuda_fp16.h>
#include <onnxruntime_cxx_api.h>
#include <vector>

// --- THE FIX: Wrap the Triton header in extern "C" ---
// This tells the C++ compiler to use C-style linkage for the Triton
// functions, preventing name mangling and resolving the "undefined symbol"
// error.
extern "C" {
#include "triton_conv_kernel.h"
}

// Define a struct to hold kernel parameters and logic
struct TritonConvKernel {
  TritonConvKernel(const OrtKernelInfo *info) {
    // Use ConstKernelInfo for safe, read-only access to attributes
    Ort::ConstKernelInfo kernel_info(info);
    pad_h = kernel_info.GetAttribute<int64_t>("pad_h");
    groups = kernel_info.GetAttribute<int64_t>("groups");
  }

  void Compute(OrtKernelContext *context) {
    Ort::KernelContext ctx(context);

    // Get Input Tensors using the ConstValue wrapper
    Ort::ConstValue Y_ort = ctx.GetInput(0);
    Ort::ConstValue W_ort = ctx.GetInput(1);
    Ort::ConstValue B_ort = ctx.GetInput(2);

    // Get Input Shapes and Data Pointers
    auto Y_shape_info = Y_ort.GetTensorTypeAndShapeInfo();
    auto W_shape_info = W_ort.GetTensorTypeAndShapeInfo();
    auto Y_shape = Y_shape_info.GetShape();
    auto W_shape = W_shape_info.GetShape();

    const half *Y_ptr = Y_ort.GetTensorData<half>();
    const half *W_ptr = W_ort.GetTensorData<half>();
    const half *B_ptr = B_ort.GetTensorData<half>();

    // Prepare Output Tensor
    const int64_t N = Y_shape[0];
    const int64_t C_in = Y_shape[1];
    const int64_t H_in = Y_shape[2];
    const int64_t W_in = Y_shape[3];
    const int64_t C_out = W_shape[0];
    const int64_t KH = W_shape[2];
    const int64_t KW = W_shape[3];
    const int64_t H_out = H_in - KH + 1 + (2 * pad_h);

    std::vector<int64_t> Z_shape = {N, C_out, H_out};

    Ort::UnownedValue Z_ort_unowned = ctx.GetOutput(0, Z_shape);
    half *Z_ptr = Z_ort_unowned.GetTensorMutableData<half>();

    // Calculate Strides
    int64_t stride_in_c = H_in * W_in;
    int64_t stride_in_h = W_in;
    int64_t stride_in_w = 1;
    int64_t stride_w_cout = (C_in / groups) * KH * KW;
    int64_t stride_out_cout = H_out;
    int64_t stride_out_h = 1;

    // Launch the CUDA Kernel
    CUstream stream = (CUstream)ctx.GetGPUComputeStream();
    triton_conv_kernel(stream, (CUdeviceptr)Y_ptr, (CUdeviceptr)W_ptr,
                       (CUdeviceptr)B_ptr, (CUdeviceptr)Z_ptr, (int32_t)H_in,
                       (int32_t)W_in, (int32_t)C_in, (int32_t)H_out,
                       (int32_t)C_out, (int32_t)KH, (int32_t)KW, stride_in_c,
                       stride_in_h, stride_in_w, stride_w_cout, stride_out_cout,
                       stride_out_h, (int32_t)pad_h, (int32_t)groups);
  }

private:
  int64_t pad_h;
  int64_t groups;
};

// Custom Operator Definition
struct TritonConvOp : Ort::CustomOpBase<TritonConvOp, TritonConvKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return new TritonConvKernel(info);
  };
  const char *GetName() const { return "TritonConv"; };
  const char *GetExecutionProviderType() const {
    return "CUDAExecutionProvider";
  };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  };
};

// Registration Logic
static Ort::CustomOpDomain custom_op_domain("com.custom.ops");
static TritonConvOp triton_conv_op;

extern "C" OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  custom_op_domain.Add(&triton_conv_op);
  return Ort::GetApi().AddCustomOpDomain(options, custom_op_domain);
}
