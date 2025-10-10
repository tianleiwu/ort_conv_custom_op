// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include "core/providers/cuda/cuda_context.h"
#include "onnxruntime_lite_custom_op.h"

#include <cuda_fp16.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

// Include the Triton kernel header within an extern "C" block to prevent C++
// name mangling.
extern "C" {
#include "triton_conv_kernel.h"
}

// Use the namespaces from the lite custom op header for cleaner code.
using namespace Ort::Custom;

// Global flag to ensure the CUDA module is loaded only once across all threads
// and instances.
static std::once_flag triton_module_loaded_flag;

// The TritonConvKernel struct now acts as a "functor" for the custom op.
// Its `Compute` method contains the kernel logic.
struct TritonConvKernel {
  TritonConvKernel(const OrtApi *&, const OrtKernelInfo *info) {
    // The constructor's only job is to read attributes from the ONNX node.
    Ort::ConstKernelInfo kernel_info(info);
    pad_h_ = kernel_info.GetAttribute<int64_t>("pad_h");
    groups_ = kernel_info.GetAttribute<int64_t>("groups");
  }

  // The signature of the Compute method is automatically parsed by the lite op
  // framework to determine the operator's inputs and outputs. This is much
  // safer than manual indexing.
  void Compute(const CudaContext &cuda_ctx,
               const Tensor<Ort::Float16_t> &Y, // 1st Input
               const Tensor<Ort::Float16_t> &W, // 2nd Input
               const Tensor<Ort::Float16_t> &B, // 3rd Input
               Tensor<Ort::Float16_t> &Z) {     // 1st Output (mutable)

    std::call_once(triton_module_loaded_flag, []() {
      std::cout
          << "[INFO] TritonConvKernel: First execution, loading CUDA module."
          << std::endl;
      load_triton_conv_kernel();
    });

    // Get shapes and data pointers from the strongly-typed Tensor objects.
    const auto &Y_shape = Y.Shape();
    const auto &W_shape = W.Shape();

    // --- FIX: Cast the data pointers to the type the kernel expects ---
    const half *Y_ptr = reinterpret_cast<const half *>(Y.Data());
    const half *W_ptr = reinterpret_cast<const half *>(W.Data());
    const half *B_ptr = reinterpret_cast<const half *>(B.Data());

    // Calculate output shape.
    const int64_t N = Y_shape[0];
    const int64_t C_in = Y_shape[1];
    const int64_t H_in = Y_shape[2];
    const int64_t W_in = Y_shape[3];
    const int64_t C_out = W_shape[0];
    const int64_t KH = W_shape[2];
    const int64_t KW = W_shape[3];
    const int64_t H_out = H_in - KH + 1 + (2 * pad_h_);

    std::vector<int64_t> Z_shape = {N, C_out, H_out};

    // Allocate the output tensor's memory and get a mutable pointer to it.
    half *Z_ptr = reinterpret_cast<half *>(Z.Allocate(Z_shape));
    
    // Calculate strides.
    int64_t stride_in_c = H_in * W_in;
    int64_t stride_in_h = W_in;
    int64_t stride_in_w = 1;
    int64_t stride_w_cout = (C_in / groups_) * KH * KW;
    int64_t stride_out_cout = H_out;
    int64_t stride_out_h = 1;

    // Get the CUDA stream from the context object provided by the framework.
    CUstream stream = cuda_ctx.cuda_stream;

    // Launch the Triton kernel.
    triton_conv_kernel(stream, (CUdeviceptr)Y_ptr, (CUdeviceptr)W_ptr,
                       (CUdeviceptr)B_ptr, (CUdeviceptr)Z_ptr, (int32_t)H_in,
                       (int32_t)W_in, (int32_t)C_in, (int32_t)H_out,
                       (int32_t)C_out, (int32_t)KH, (int32_t)KW, stride_in_c,
                       stride_in_h, stride_in_w, stride_w_cout, stride_out_cout,
                       stride_out_h, (int32_t)pad_h_, (int32_t)groups_);
  }
  
private:
  int64_t pad_h_;
  int64_t groups_;
};

// --- New Registration Logic (following the provided example) ---

static const char *c_OpDomain = "com.custom.ops";

// This function registers all CUDA custom ops for a given domain.
void RegisterCudaOps(Ort::CustomOpDomain &domain) {
  // A static unique_ptr ensures the operator definition has a permanent
  // lifetime after it's created on the first call.
  static const auto triton_conv_op =
      Ort::Custom::CreateLiteCustomOp<TritonConvKernel>(
          "TritonConv", "CUDAExecutionProvider");
  domain.Add(triton_conv_op);
}

// This container and mutex are used to manage the lifetime of the
// CustomOpDomain object, preventing it from being destroyed while the ONNX
// Runtime session is still active.
static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain &&domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

// This is the main C-style entry point that ONNX Runtime will call to register
// the operators.
extern "C" OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  OrtStatus *result = nullptr;

  // The ORT_TRY/ORT_CATCH block provides exception safety.
  try {
    Ort::CustomOpDomain domain{c_OpDomain};
    RegisterCudaOps(domain);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);

    // Move the domain to the static container to ensure it stays alive.
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    printf("Exception during custom op registration: %s\n", e.what());
  }

  return result;
}
