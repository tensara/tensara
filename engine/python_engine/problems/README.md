# Problems

This folder contains the problem definitions for the engine.

## Structure

Each problem should have:
2. A `[problem-name].py` file that contains the problem definition using the `Problem` pattern.

## Problem Structure

All problems must inherit from the base `Problem` class and implement its abstract methods:

- `reference_solution`: The PyTorch implementation that serves as the ground truth
- `generate_test_cases`: Creates test cases with varying sizes and parameters
- `verify_result`: Checks if the CUDA solution output matches the reference solution
- `get_function_signature`: Defines the C types for the CUDA function interface
- `get_flops`: Calculates the number of floating point operations for performance metrics
- `get_extra_params` (optional): Provides additional parameters to the CUDA solution

## Example Implementation

See `vector_addition.py` for a complete example of a problem implementation.

## Adding New Problems

To add a new problem:
Create a `[problem-name].py` file that:
   - Inherits from the `Problem` base class
   - Implements all required abstract methods
   - Provides appropriate test cases with varying sizes

Register the problem in the engine by adding it to the problem registry.

## Problem Difficulty Guidelines
Problems should be categorized by difficulty:
- Mangla will fill this....

## Additional Guidelines
- Mangla will fill this....