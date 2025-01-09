cmake_minimum_required(VERSION 3.18)
project(cuda_benchmark CUDA)

# Set CUDA architecture to match your GPU
# Change this according to your GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 75)  # For RTX 20xx series, adjust as needed

# Add executable
add_executable(benchmark benchmark.cu)

# Set C++ standard
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)
