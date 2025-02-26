# CUDA Checker Generator

This tool generates problem-specific files for the CUDA/PyTorch checker system, allowing you to compare CUDA implementations against PyTorch reference implementations for any computational problem.

## Quick Start

1. Create a problem specification JSON file (see example below)
2. Run the generator script
3. Fill in the implementation details in the generated files
4. Build and run

## Creating a Problem Specification

Create a JSON file that describes your problem. Here's an example for matrix multiplication:

```json
{
  "name": "MatrixMultiply",
  "inputs": [
    {"name": "input_a", "type": "float", "dims": ["m", "k"]},
    {"name": "input_b", "type": "float", "dims": ["k", "n"]}
  ],
  "outputs": [
    {"name": "output_c", "type": "float", "dims": ["m", "n"]}
  ],
  "parameters": ["m", "n", "k"]
}
```

The specification includes:
- `name`: The name of your problem (in CamelCase)
- `inputs`: Array of input tensors with name, type, and dimensions
- `outputs`: Array of output tensors with name, type, and dimensions
- `parameters`: Array of dimension parameters

## Generating the Files

Run the generator script with your specification:

```bash
python generate_simple.py --spec your_problem_spec.json --output output_directory
```

This will generate all the necessary files with the appropriate interfaces for your problem:

- `python_reference.hpp` - Header for Python callback interface
- `python_reference.cpp` - Implementation of Python interface
- `bindings.cpp` - PyBind11 bindings for Python/C++ interop
- `reference.py` - Template for your PyTorch implementation
- `solution.cu` - Template for your CUDA implementation
- `tests.hpp` - Test case definitions for your problem

## Customizing the Implementation

After generating the files, you'll need to:

1. Fill in the PyTorch implementation in `reference.py`
2. Fill in the CUDA implementation in `solution.cu`
3. Adjust test cases in `tests.hpp` if needed

## Building and Running

Build the checker and Python bindings:

```bash
make all
```

Run the checker with your PyTorch reference:

```bash
python reference.py
```

## Adding New Problem Types

You can create new problem specifications to generate checker files for different problems:

### Vector Addition Example

```json
{
  "name": "VectorAdd",
  "inputs": [
    {"name": "input_a", "type": "float", "dims": ["n"]},
    {"name": "input_b", "type": "float", "dims": ["n"]}
  ],
  "outputs": [
    {"name": "output_c", "type": "float", "dims": ["n"]}
  ],
  "parameters": ["n"]
}
```

### Convolution Example

```json
{
  "name": "Convolution2D",
  "inputs": [
    {"name": "input", "type": "float", "dims": ["batch", "in_channels", "height", "width"]},
    {"name": "weight", "type": "float", "dims": ["out_channels", "in_channels", "kernel_h", "kernel_w"]}
  ],
  "outputs": [
    {"name": "output", "type": "float", "dims": ["batch", "out_channels", "out_height", "out_width"]}
  ],
  "parameters": ["batch", "in_channels", "out_channels", "height", "width", "kernel_h", "kernel_w", "stride", "padding"]
}
```

## Generator Limitations

The current simplified generator works best for problems with:
- Float inputs and outputs
- 1D to 3D tensor shapes
- Standard math operations

For more complex problems, you may need to modify the generated templates. 