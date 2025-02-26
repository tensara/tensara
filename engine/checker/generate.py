#!/usr/bin/env python3
import json
import argparse
import os
import textwrap

def camel_to_snake(name):
    """Convert CamelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def generate_python_reference_hpp(spec):
    """Generate python_reference.hpp for the given problem specification."""
    
    # Create the function signature for the callback type and functions
    input_params = []
    for inp in spec["inputs"]:
        input_params.append(f"{inp['type']}* {inp['name']}")
    
    output_params = []
    for outp in spec["outputs"]:
        output_params.append(f"{outp['type']}* {outp['name']}")
    
    param_list = ", ".join(input_params + output_params + [f"size_t {p}" for p in spec["parameters"]])
    
    template = f"""#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

// Function pointer type for Python reference solution callback
typedef void (*PythonReferenceCallback)(
    {param_list}
);

// Global function pointer that will hold the Python callback
extern PythonReferenceCallback g_python_reference_callback;

// Initialize function to register the Python callback
extern "C" void register_python_reference(PythonReferenceCallback callback);

// Reference solution that calls into Python
extern "C" void reference_solution({param_list});
"""
    return template

def generate_python_reference_cpp(spec):
    """Generate python_reference.cpp for the given problem specification."""
    
    input_params = []
    input_names = []
    for inp in spec["inputs"]:
        input_params.append(f"{inp['type']}* {inp['name']}")
        input_names.append(inp['name'])
    
    output_params = []
    output_names = []
    for outp in spec["outputs"]:
        output_params.append(f"{outp['type']}* {outp['name']}")
        output_names.append(outp['name'])
    
    param_names = input_names + output_names + spec["parameters"]
    param_list = ", ".join(input_params + output_params + [f"size_t {p}" for p in spec["parameters"]])
    param_call = ", ".join(param_names)
    
    # For error handling, find the first output and its dimensions to zero it
    first_output = spec["outputs"][0]
    output_size = " * ".join(first_output["dims"])
    
    template = f"""#include "python_reference.hpp"
#include <iostream>

// Initialize the global callback pointer to nullptr
PythonReferenceCallback g_python_reference_callback = nullptr;

// Function to register the Python callback
extern "C" void register_python_reference(PythonReferenceCallback callback) {{
    g_python_reference_callback = callback;
    std::cout << "Python reference callback registered" << std::endl;
}}

// Implementation of the reference_solution that calls the Python callback
extern "C" void reference_solution({param_list}) {{
    if (g_python_reference_callback == nullptr) {{
        std::cerr << "Error: Python reference callback not registered" << std::endl;
        // Fill output with zeros to avoid undefined behavior
        for (size_t i = 0; i < {output_size}; i++) {{
            {first_output["name"]}[i] = 0.0f;
        }}
        return;
    }}
    
    // Call the Python implementation
    g_python_reference_callback({param_call});
}}
"""
    return template

def generate_bindings_cpp(spec):
    """Generate bindings.cpp for the given problem specification."""
    
    # Get parameter types and names
    input_params = []
    input_vars = []
    for inp in spec["inputs"]:
        input_params.append(f"{inp['type']}* {inp['name']}")
        input_vars.append(inp['name'])
    
    output_params = []
    output_vars = []
    for outp in spec["outputs"]:
        output_params.append(f"{outp['type']}* {outp['name']}")
        output_vars.append(outp['name'])
    
    param_list = ", ".join(input_params + output_params + [f"size_t {p}" for p in spec["parameters"]])
    param_vars = ", ".join(input_vars + output_vars + spec["parameters"])
    
    # Generate numpy array creation code for each input
    numpy_input_code = []
    for idx, inp in enumerate(spec["inputs"]):
        dims = ", ".join(inp["dims"])
        numpy_input_code.append(f"""    // Create numpy array view for {inp['name']}
    py::array_t<{inp['type']}> py_{inp['name']}({{{dims}}}, {inp['name']});""")
    
    numpy_inputs = "\n".join(numpy_input_code)
    
    # Generate output copy code for the first output
    first_output = spec["outputs"][0]
    dims = len(first_output["dims"])
    
    output_copy_code = []
    if dims == 1:
        dim = first_output["dims"][0]
        output_copy_code.append(f"""    // Copy results back to C++ array
    for (size_t i = 0; i < {dim}; i++) {{
        {first_output['name']}[i] = r(i);
    }}""")
    elif dims == 2:
        dim1, dim2 = first_output["dims"]
        output_copy_code.append(f"""    // Copy results back to C++ array
    for (size_t i = 0; i < {dim1}; i++) {{
        for (size_t j = 0; j < {dim2}; j++) {{
            {first_output['name']}[i * {dim2} + j] = r(i, j);
        }}
    }}""")
    elif dims == 3:
        dim1, dim2, dim3 = first_output["dims"]
        output_copy_code.append(f"""    // Copy results back to C++ array
    for (size_t i = 0; i < {dim1}; i++) {{
        for (size_t j = 0; j < {dim2}; j++) {{
            for (size_t k = 0; k < {dim3}; k++) {{
                {first_output['name']}[i * {dim2} * {dim3} + j * {dim3} + k] = r(i, j, k);
            }}
        }}
    }}""")
    
    output_copy = "\n".join(output_copy_code)
    
    # Parameters for the python callback
    py_params = ", ".join([f"py_{inp['name']}" for inp in spec["inputs"]] + spec["parameters"])
    
    template = f"""#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "python_reference.hpp"
#include "core.hpp"
#include <functional>

namespace py = pybind11;

// Wrapper function for Python to call
void run_checker() {{
    // This will invoke the main function in checker.cu
    int result = system("./checker");
    if (result != 0) {{
        throw std::runtime_error("Checker failed with exit code " + std::to_string(result));
    }}
}}

// Callback from Python to C++
void python_reference_callback_wrapper(
    py::function py_callback,
    {param_list}) {{
    
{numpy_inputs}
    
    // Call the Python function
    py::object result = py_callback({py_params});
    
    // Get numpy array from result
    py::array_t<{first_output['type']}> py_output = result.cast<py::array_t<{first_output['type']}>>();
    auto r = py_output.unchecked<{dims}>();
    
{output_copy}
}}

// Register a Python function as the reference implementation
void register_python_reference_impl(py::function py_callback) {{
    // Create a C++ callback that wraps the Python function
    auto cpp_callback = [py_callback](
        {param_list}) {{
        python_reference_callback_wrapper(
            py_callback, {param_vars});
    }};
    
    // Register the C++ callback
    register_python_reference(cpp_callback);
}}

PYBIND11_MODULE(checker_bindings, m) {{
    m.doc() = "PyTorch to CUDA checker bindings for {spec["name"]}";
    
    // Expose function to register Python reference implementation
    m.def("register_reference", &register_python_reference_impl, 
          "Register a PyTorch reference implementation");
    
    // Expose function to run checker
    m.def("run_checker", &run_checker, 
          "Run the checker to compare PyTorch reference with CUDA solution");
}}
"""
    return template

def generate_reference_py(spec):
    """Generate reference.py template for the given problem specification."""
    
    function_name = camel_to_snake(spec["name"]) + "_reference"
    
    # Create the function signature for the Python reference
    input_params = [inp['name'] for inp in spec["inputs"]]
    param_list = ", ".join(input_params + spec["parameters"])
    
    # Generate docstring with input descriptions
    doc_inputs = "\n    ".join([f"{inp['name']}: numpy array of shape ({', '.join(inp['dims'])})" 
                               for inp in spec["inputs"]])
    doc_params = "\n    ".join([f"{p}: size parameter" for p in spec["parameters"]])
    doc_output = f"numpy array of shape ({', '.join(spec['outputs'][0]['dims'])})"
    
    # Generate tensor conversion code
    tensor_conversions = ", ".join([f"tensor_{inp}" for inp in input_params])
    tensor_from_numpy = ", ".join([f"torch.from_numpy(np.asarray({inp})).float()" for inp in input_params])
    tensor_to_device = ", ".join([f"tensor_{inp} = tensor_{inp}.to(device)" for inp in input_params])
    
    # Generate result shape
    result_shape = ", ".join([dim for dim in spec["outputs"][0]["dims"]])
    
    # Create template by concatenating parts to avoid f-string issues with triple quotes
    template = [
        "import torch",
        "import numpy as np",
        "import checker_bindings",
        "",
        f"def {function_name}({param_list}):",
        '    """',
        f'    PyTorch implementation of {spec["name"]}.',
        '    ',
        '    Args:',
        f'        {doc_inputs}',
        f'        {doc_params}',
        '    ',
        '    Returns:',
        f'        {doc_output} containing the result',
        '    """',
        f'    # Convert numpy arrays to PyTorch tensors',
        f'    {tensor_conversions} = {tensor_from_numpy}',
        '    ',
        '    # Move tensors to GPU if available',
        '    device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')',
        f'    {tensor_to_device}',
        '    ',
        '    # TODO: Implement your PyTorch solution here',
        '    # For example:',
        '    # result = torch.some_operation(tensor_input_a, tensor_input_b)',
        '    ',
        '    # Placeholder for your implementation',
        '    # This is a dummy implementation that should be replaced',
        f'    result = torch.zeros({result_shape})',
        '    ',
        '    # Convert result back to numpy array on CPU',
        '    return result.cpu().numpy()',
        '',
        'def main():',
        '    # Register our PyTorch reference implementation',
        f'    checker_bindings.register_reference({function_name})',
        '    ',
        '    # Run the checker',
        '    try:',
        '        checker_bindings.run_checker()',
        '        print("All tests passed!")',
        '    except RuntimeError as e:',
        '        print(f"Checker failed: {e}")',
        '',
        'if __name__ == "__main__":',
        '    main()'
    ]
    
    return "\n".join(template)

def generate_solution_cu(spec):
    """Generate solution.cu template for the given problem specification."""
    
    # Create the function signature
    input_params = []
    for inp in spec["inputs"]:
        input_params.append(f"{inp['type']}* {inp['name']}")
    
    output_params = []
    for outp in spec["outputs"]:
        output_params.append(f"{outp['type']}* {outp['name']}")
    
    param_list = ", ".join(input_params + output_params + [f"size_t {p}" for p in spec["parameters"]])
    
    # Create a basic kernel template
    template = f"""#include <cuda_runtime.h>

__global__ void {camel_to_snake(spec["name"])}_kernel({param_list}) {{
    // TODO: Implement your CUDA kernel here
    
    // Example code (this needs to be replaced with your implementation):
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < /* max_size */) {{
        // Do computation
    }}
}}

extern "C" void solution({param_list}) {{
    // TODO: Set up your grid and block dimensions appropriately
    dim3 blockDim(256);
    dim3 gridDim(/* Calculate appropriate grid dimensions */);

    // Launch your kernel
    {camel_to_snake(spec["name"])}_kernel<<<gridDim, blockDim>>>({", ".join([inp["name"] for inp in spec["inputs"]] + [outp["name"] for outp in spec["outputs"]] + spec["parameters"])});
}}
"""
    return template

def generate_tests_hpp(spec):
    """Generate tests.hpp template for the given problem specification."""
    
    class_name = spec["name"] + "Test"
    
    # Create a function pointer type for the kernel
    input_params = []
    for inp in spec["inputs"]:
        input_params.append(f"{inp['type']}*")
    
    output_params = []
    for outp in spec["outputs"]:
        output_params.append(f"{outp['type']}*")
    
    func_ptr_params = ", ".join(input_params + output_params + ["size_t" for _ in spec["parameters"]])
    
    # Create constructor code that initializes tensors
    tensor_init = []
    for idx, inp in enumerate(spec["inputs"]):
        dims = ", ".join([f"sizes[{i}]" for i in range(len(inp["dims"]))])
        tensor_init.append(f"""        auto {inp['name']}_shape = std::vector<size_t>{{{dims}}};""")
    
    for idx, outp in enumerate(spec["outputs"]):
        dims = ", ".join([f"sizes[{i}]" for i in range(len(outp["dims"]))])
        tensor_init.append(f"""        auto {outp['name']}_shape = std::vector<size_t>{{{dims}}};""")
    
    tensor_init_code = "\n".join(tensor_init)
    
    # Create code to initialize inputs and outputs vectors
    inputs_init = []
    for inp in spec["inputs"]:
        inputs_init.append(f"""            std::make_shared<Tensor<T>>({inp['name']}_shape)""")
    
    outputs_init = []
    for outp in spec["outputs"]:
        outputs_init.append(f"""            std::make_shared<Tensor<T>>({outp['name']}_shape)""")
    
    inputs_init_code = ",\n".join(inputs_init)
    outputs_init_code = ",\n".join(outputs_init)
    
    # Prepare data section
    prepare_data = f"""        // TODO: Initialize your test data here
        // This is just an example - replace with appropriate initialization
        
        // Initialize inputs with example values
        for (size_t i = 0; i < this->inputs_[0]->size(); i++) {{
            host_inputs[0][i] = static_cast<T>(i % 10);  // Example data
        }}
        
        // More initialization code here as needed
"""
    
    # Launch kernel code
    kernel_params = ", ".join(["inputs[" + str(i) + "]" for i in range(len(spec["inputs"]))] + 
                              ["outputs[" + str(i) + "]" for i in range(len(spec["outputs"]))] + 
                              ["sizes[" + str(i) + "]" for i in range(len(spec["parameters"]))])
    
    template = f"""#include "core.hpp"

template<typename T>
class {class_name}: public TestCase<T> {{
public:
    using kernel_func_t = void (*)(
        {func_ptr_params}
    );
    
    {class_name}({", ".join([f"size_t {p}" for p in spec["parameters"]])}) {{
        // Calculate problem size (can be customized)
        this->problem_size_ = {" * ".join(spec["parameters"])};
        this->name_ = {" + " + '.join([f'std::to_string({p})' for p in spec["parameters"]])};
        
        // Define tensor shapes based on parameters
{tensor_init_code}

        // Initialize inputs and outputs tensors
        this->inputs_ = {{
{inputs_init_code}
        }};
        this->outputs_ = {{
{outputs_init_code}
        }};
    }}
    
    void prepare_data(T** host_inputs, T** host_outputs) override {{
{prepare_data}
    }}
    
    std::string get_name() const override {{
        return this->name_;
    }}

    size_t calculate_flops() const override {{
        // TODO: Calculate FLOPs for this operation
        // This is an example - replace with appropriate calculation
        return this->problem_size_ * 2;
    }}

    std::vector<size_t> get_sizes() const override {{
        // Return the sizes needed for the kernel
        return {{{", ".join(spec["parameters"])}}};
    }}

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {{
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func({kernel_params});
    }}
}};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {{
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    // TODO: Add appropriate test cases for your problem
    // These are example test cases - replace with appropriate ones
    test_cases.push_back(std::make_unique<{class_name}<float>>({", ".join(["64" for _ in spec["parameters"]])}));
    test_cases.push_back(std::make_unique<{class_name}<float>>({", ".join(["128" for _ in spec["parameters"]])}));
    test_cases.push_back(std::make_unique<{class_name}<float>>({", ".join(["256" for _ in spec["parameters"]])}));

    return test_cases;
}}
"""
    return template

def create_problem_files(spec, output_dir):
    """Create all problem-specific files based on the specification."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each file
    files = {
        "python_reference.hpp": generate_python_reference_hpp(spec),
        "python_reference.cpp": generate_python_reference_cpp(spec),
        "bindings.cpp": generate_bindings_cpp(spec),
        "reference.py": generate_reference_py(spec),
        "solution.cu": generate_solution_cu(spec),
        "tests.hpp": generate_tests_hpp(spec)
    }
    
    # Write files to output directory
    for filename, content in files.items():
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(content)
    
    print(f"Created problem files for {spec['name']} in {output_dir}")
    print("Generated files:")
    for filename in files.keys():
        print(f"  - {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate problem-specific files for CUDA checker")
    parser.add_argument("--spec", required=True, help="Path to problem specification JSON file")
    parser.add_argument("--output", default=".", help="Output directory for generated files")
    
    args = parser.parse_args()
    
    # Load the specification
    with open(args.spec, 'r') as f:
        spec = json.load(f)
    
    # Create the files
    create_problem_files(spec, args.output)

if __name__ == "__main__":
    main() 