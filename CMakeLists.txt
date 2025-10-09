# Set minimum required version and project details
cmake_minimum_required(VERSION 3.18)
project(TritonCustomOp LANGUAGES CXX CUDA)

# --- Find Dependencies ---

# Find CUDA using the standard modern CMake approach
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIRS}")

# Manually configure ONNX Runtime paths from the provided root directory.
# The Python script MUST pass -DONNXRUNTIME_ROOT_DIR=/path/to/ort
if(NOT DEFINED ONNXRUNTIME_ROOT_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT_DIR is not defined. Please pass it via the CMake command line.")
endif()
message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT_DIR}")

# Explicitly add the ONNX Runtime library directory to the search path for the linker
link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)


# --- Define the Custom Operator Library ---
add_library(triton_conv_op SHARED)

# Add preprocessor definition needed for the InitApi() function in the C++ code
target_compile_definitions(triton_conv_op PRIVATE ORT_API_MANUAL_INIT)

# Add all necessary include directories for the compiler to find headers.
target_include_directories(triton_conv_op PRIVATE
    # For the generated 'triton_conv_kernel.h' in the build directory
    ${CMAKE_CURRENT_BINARY_DIR}

    # For 'onnxruntime_cxx_api.h'
    ${ONNXRUNTIME_ROOT_DIR}/include

    # For 'cuda.h' and other CUDA headers
    ${CUDA_INCLUDE_DIRS}
)

# List the source files for the library, correcting the typo in "cutom_op.cpp"
target_sources(triton_conv_op PRIVATE
    cutom_op.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/triton_conv_kernel.cpp
)

# Link against the required libraries by name.
target_link_libraries(triton_conv_op PRIVATE
    onnxruntime      # Found via link_directories()
    cuda
    ${CUDA_LIBRARIES} # Variable defined by find_package(CUDA)
)

# Set C++ standard
set_target_properties(triton_conv_op PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Ensure the final shared library is placed in the build directory for the Python script
set_target_properties(triton_conv_op PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)