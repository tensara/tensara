#!/usr/bin/env python3
import json
import argparse
import os
import re

def camel_to_snake(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def generate_files(spec_file, output_dir):
    """Generate all necessary files for the problem."""
    # Read the spec file
    with open(spec_file, 'r') as f:
        spec = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate files
    generate_python_reference_hpp(spec, output_dir)
    generate_python_reference_cpp(spec, output_dir)
    generate_bindings_cpp(spec, output_dir)
    generate_reference_py(spec, output_dir)
    generate_solution_cu(spec, output_dir)
    generate_tests_hpp(spec, output_dir)
    
    print(f"Generated all files for {spec['name']} in {output_dir}")

def generate_python_reference_hpp(spec, output_dir):
    """Generate python_reference.hpp file."""
    content = """#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

// Function pointer type for Python reference solution callback
typedef void (*PythonReferenceCallback)(
    float* input_a, float* input_b, float* output_c,
    size_t m, size_t n, size_t k);

// Global function pointer that will hold the Python callback
extern PythonReferenceCallback g_python_reference_callback;

// Initialize function to register the Python callback
extern "C" void register_python_reference(PythonReferenceCallback callback);

// Reference solution that calls into Python
extern "C" void reference_solution(float* input_a, float* input_b, float* output_c, size_t m, size_t n, size_t k);
"""
    
    with open(os.path.join(output_dir, "python_reference.hpp"), 'w') as f:
        f.write(content)

def generate_python_reference_cpp(spec, output_dir):
    """Generate python_reference.cpp file."""
    content = """#include "python_reference.hpp"
#include <iostream>

// Initialize the global callback pointer to nullptr
PythonReferenceCallback g_python_reference_callback = nullptr;

// Function to register the Python callback
extern "C" void register_python_reference(PythonReferenceCallback callback) {
    g_python_reference_callback = callback;
    std::cout << "Python reference callback registered" << std::endl;
}

// Implementation of the reference_solution that calls the Python callback
extern "C" void reference_solution(float* input_a, float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    if (g_python_reference_callback == nullptr) {
        std::cerr << "Error: Python reference callback not registered" << std::endl;
        // Fill output with zeros to avoid undefined behavior
        for (size_t i = 0; i < m * n; i++) {
            output_c[i] = 0.0f;
        }
        return;
    }
    
    // Call the Python implementation
    g_python_reference_callback(input_a, input_b, output_c, m, n, k);
}
"""
    
    with open(os.path.join(output_dir, "python_reference.cpp"), 'w') as f:
        f.write(content)

def generate_bindings_cpp(spec, output_dir):
    """Generate bindings.cpp file."""
    content = """#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "python_reference.hpp"
#include "core.hpp"
#include <functional>

namespace py = pybind11;

// Wrapper function for Python to call
void run_checker() {
    // This will invoke the main function in checker.cu
    int result = system("./checker");
    if (result != 0) {
        throw std::runtime_error("Checker failed with exit code " + std::to_string(result));
    }
}

// Callback from Python to C++
void python_reference_callback_wrapper(
    py::function py_callback,
    float* input_a, float* input_b, float* output_c,
    size_t m, size_t n, size_t k) {
    
    // Create numpy array views without copying the data
    py::array_t<float> py_input_a({m, k}, input_a);
    py::array_t<float> py_input_b({k, n}, input_b);
    
    // Call the Python function
    py::object result = py_callback(py_input_a, py_input_b, m, n, k);
    
    // Get numpy array from result
    py::array_t<float> py_output = result.cast<py::array_t<float>>();
    auto r = py_output.unchecked<2>();
    
    // Copy results back to C++ array
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            output_c[i * n + j] = r(i, j);
        }
    }
}

// Register a Python function as the reference implementation
void register_python_reference_impl(py::function py_callback) {
    // Create a C++ callback that wraps the Python function
    auto cpp_callback = [py_callback](
        float* input_a, float* input_b, float* output_c,
        size_t m, size_t n, size_t k) {
        python_reference_callback_wrapper(
            py_callback, input_a, input_b, output_c, m, n, k);
    };
    
    // Register the C++ callback
    register_python_reference(cpp_callback);
}

PYBIND11_MODULE(checker_bindings, m) {
    m.doc() = "PyTorch to CUDA checker bindings";
    
    // Expose function to register Python reference implementation
    m.def("register_reference", &register_python_reference_impl, 
          "Register a PyTorch reference implementation");
    
    // Expose function to run checker
    m.def("run_checker", &run_checker, 
          "Run the checker to compare PyTorch reference with CUDA solution");
}
"""
    
    with open(os.path.join(output_dir, "bindings.cpp"), 'w') as f:
        f.write(content)

def generate_reference_py(spec, output_dir):
    """Generate reference.py file."""
    function_name = camel_to_snake(spec["name"]) + "_reference"
    
    # Use raw strings for triple quotes to avoid f-string issues
    content = f"""import torch
import numpy as np
import checker_bindings

def {function_name}(input_a, input_b, m, n, k):
    '''
    PyTorch implementation of {spec['name']}.
    
    Args:
        input_a: numpy array of shape (m, k)
        input_b: numpy array of shape (k, n)
        m, n, k: dimensions
        
    Returns:
        numpy array of shape (m, n) containing the result
    '''
    # Convert numpy arrays to PyTorch tensors
    tensor_input_a = torch.from_numpy(np.asarray(input_a)).float()
    tensor_input_b = torch.from_numpy(np.asarray(input_b)).float()
    
    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_input_a = tensor_input_a.to(device)
    tensor_input_b = tensor_input_b.to(device)
    
    # TODO: Implement your PyTorch solution here
    # For matrix multiplication, we can use torch.matmul or the @ operator
    result = torch.matmul(tensor_input_a, tensor_input_b)
    
    # Convert result back to numpy array on CPU
    return result.cpu().numpy()

def main():
    # Register our PyTorch reference implementation
    checker_bindings.register_reference({function_name})
    
    # Run the checker
    try:
        checker_bindings.run_checker()
        print("All tests passed!")
    except RuntimeError as e:
        print(f"Checker failed: {{e}}")

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(output_dir, "reference.py"), 'w') as f:
        f.write(content)

def generate_solution_cu(spec, output_dir):
    """Generate solution.cu file."""
    content = """#include <cuda_runtime.h>

__global__ void matrix_multiply_kernel(float* input_a, float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += input_a[row * k + i] * input_b[i * n + col];
        }
        output_c[row * n + col] = sum;
    }
}

extern "C" void solution(float* input_a, float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);

    matrix_multiply_kernel<<<gridDim, blockDim>>>(input_a, input_b, output_c, m, n, k);   
}
"""
    
    with open(os.path.join(output_dir, "solution.cu"), 'w') as f:
        f.write(content)

def generate_tests_hpp(spec, output_dir):
    """Generate tests.hpp file."""
    content = """#include "core.hpp"

template<typename T>
class MatrixMultiplyTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T*, T*, size_t, size_t, size_t);
    
    MatrixMultiplyTest(size_t m, size_t n, size_t k) {
        this->problem_size_ = m * n * k;
        this->name_ = std::to_string(m) + "x" + std::to_string(k) + " x " + std::to_string(k) + "x" + std::to_string(n);
        
        auto matrix_a_shape = std::vector<size_t>{m, k};
        auto matrix_b_shape = std::vector<size_t>{k, n}; 
        auto matrix_c_shape = std::vector<size_t>{m, n};

        this->inputs_ = {
            std::make_shared<Tensor<T>>(matrix_a_shape),
            std::make_shared<Tensor<T>>(matrix_b_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(matrix_c_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        const size_t n = this->inputs_[1]->shape()[1];
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < k; j++) {
                host_inputs[0][i * k + j] = static_cast<T>(i + j);
            }
        }
        
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < n; j++) {
                host_inputs[1][i * n + j] = static_cast<T>(i * j);
            }
        }
    }
    
    std::string get_name() const override {
        return this->name_;
    }

    size_t calculate_flops() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        const size_t n = this->inputs_[1]->shape()[1];
        return m * n * k * 2;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        const size_t n = this->inputs_[1]->shape()[1];
        return {m, n, k};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], outputs[0], sizes[0], sizes[1], sizes[2]);
    }
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    test_cases.push_back(std::make_unique<MatrixMultiplyTest<float>>(128, 128, 128));
    test_cases.push_back(std::make_unique<MatrixMultiplyTest<float>>(256, 256, 256));
    test_cases.push_back(std::make_unique<MatrixMultiplyTest<float>>(512, 512, 512));

    return test_cases;
}
"""
    
    with open(os.path.join(output_dir, "tests.hpp"), 'w') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Generate files for CUDA/PyTorch checker")
    parser.add_argument("--spec", required=True, help="Path to the problem specification JSON")
    parser.add_argument("--output", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    generate_files(args.spec, args.output)

if __name__ == "__main__":
    main() 