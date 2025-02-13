cmake_minimum_required(VERSION 3.18)
project(cuda_solution CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Create CUDA library
add_library(cuda_kernel STATIC
    solution.cuh
)
set_target_properties(cuda_kernel PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create Python module
pybind11_add_module(cuda_solution bindings.cpp)
target_link_libraries(cuda_solution PRIVATE cuda_kernel ${CUDA_LIBRARIES})
target_include_directories(cuda_solution PRIVATE ${CUDA_INCLUDE_DIRS}) 