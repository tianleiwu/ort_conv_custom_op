cmake_minimum_required(VERSION 3.18)
project(TritonCustomOp LANGUAGES CXX C) # Enable both C and CXX compilers

# --- Find Dependencies ---

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIRS}")

set(CUDNN_PATH /home/tlwu/cudnn9.12)

find_path(
    CUDNN_INCLUDE_DIR cudnn.h
    HINTS $ENV{CUDNN_PATH} ${CUDNN_PATH}
    PATH_SUFFIXES include
    REQUIRED
)

if(NOT DEFINED ONNXRUNTIME_ROOT_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT_DIR is not defined.")
endif()
message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT_DIR}")
link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)

# --- Define the Custom Operator Library ---

add_library(triton_conv_op SHARED)

# Add preprocessor definition needed for the InitApi() function in the C++ code
target_compile_definitions(triton_conv_op PRIVATE ORT_API_MANUAL_INIT)
target_compile_definitions(triton_conv_op PRIVATE ORT_CUDA_CTX)

# Add all necessary include directories.

target_include_directories(triton_conv_op PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

# List the source files.

target_sources(triton_conv_op PRIVATE
    cutom_op.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/triton_conv_kernel.cpp
)

# --- THE FIX ---
# Tell CMake to compile the Triton-generated file as C code, not C++.
# This prevents C++ name mangling on the Triton kernel functions.
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/triton_conv_kernel.cpp
                          PROPERTIES LANGUAGE C)

# Link against the required libraries.

target_link_libraries(triton_conv_op PRIVATE
    onnxruntime
    cuda # For the CUDA Driver API (cuModuleLoadData, etc.)
)

# Set C++ standard for C++ files in the target

set_target_properties(triton_conv_op PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Ensure the final shared library is placed in the build directory

set_target_properties(triton_conv_op PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)