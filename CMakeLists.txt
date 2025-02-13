cmake_minimum_required(VERSION 3.23.1)
project(vector_add_benchmark CUDA CXX)

# Find NVBench package
find_package(nvbench REQUIRED)

# Add the benchmark executable
add_executable(vector_add_bench 
    engine/benchmark/benchmark.cu
)

# Link against NVBench
target_link_libraries(vector_add_bench PRIVATE nvbench::main)

# Set C++17 standard
set_target_properties(vector_add_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
) 