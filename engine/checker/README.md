# CUDA Checker with PyTorch Reference

This system allows comparing a CUDA implementation against a PyTorch reference implementation for correctness and performance.

## Requirements

- CUDA Toolkit
- Python 3.6+
- PyTorch
- pybind11 (install with `pip install pybind11`)

## Installation

### Option 1: Build Manually

Build both the checker and Python bindings:

```bash
make all
```

### Option 2: Install as Python Package

Install the checker as a Python package:

```bash
pip install -e .
```

This will build the necessary bindings and make the module available in your Python environment.

## How it Works

The system consists of:

1. A CUDA solution implementation in `solution.cu`
2. A PyTorch reference implementation in `reference.py`
3. Python bindings to connect the PyTorch and CUDA code
4. A checker system that verifies the output of both implementations

## Usage

### Running with Manual Build

Run the checker with the PyTorch reference:

```bash
make python
```

Or just run:

```bash
python3 reference.py
```

### Running as Python Package

After installing as a Python package, you can import and use it in your code:

```python
import torch
from checker import checker_bindings

def my_reference_impl(input_a, input_b, m, n, k):
    # Your PyTorch implementation here
    tensor_a = torch.from_numpy(input_a)
    tensor_b = torch.from_numpy(input_b)
    result = torch.matmul(tensor_a, tensor_b)
    return result.numpy()

# Register your reference implementation
checker_bindings.register_reference(my_reference_impl)

# Run the checker
checker_bindings.run_checker()
```

## Customization

### Modifying the Reference Implementation

To modify the reference implementation, edit the `matrix_multiply_reference` function in `reference.py`. You can use any PyTorch operations inside this function.

### Modifying the Solution Implementation

To modify the CUDA solution, edit the `solution` function in `solution.cu`. This is the implementation that will be compared against the PyTorch reference.

### Adding New Test Cases

To add new test cases, modify the `create_test_cases` function in `tests.hpp`.

## How It Works

1. The Python script registers a reference implementation with the C++ bindings
2. When the checker runs, it:
   - Allocates memory for inputs and outputs
   - Generates input data
   - Runs the CUDA solution
   - Runs the PyTorch reference (via the registered callback)
   - Compares the outputs for correctness
   - Reports success or failure

This approach allows you to use all the power and flexibility of PyTorch for your reference implementation while still validating against a high-performance CUDA solution. 