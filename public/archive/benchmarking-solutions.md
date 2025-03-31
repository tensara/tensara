---
title: "Benchmarking Solutions"
date: "2025-02-26"
---

To benchmark solutions, we use [GigaFLOPS](https://en.wikipedia.org/wiki/Floating_point_operations_per_second) as the primary performance metric. FLOPS is defined as:
$$
\text{FLOPS} = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Runtime in seconds}}
$$

## Calculating FLOPs

For each problem type, we define a `calculate_flops` function that computes the total number of floating point operations based on input dimensions. For example, the `calculate_flops` function for matrix multiplication of dimensions `m x k` and `k x n` is defined as:

```cpp
size_t calculate_flops() {
      const size_t m = this->inputs_[0]->shape()[0];
      const size_t k = this->inputs_[0]->shape()[1];
      const size_t n = this->inputs_[1]->shape()[1];
      return m * n * k * 2;
  }
```
Since we now have a generalized `calculate_flops` function, we can measure performance across different input sizes and average the FLOPS across test cases. This approach is similar to ones used in papers like [Flash Attention 3](https://arxiv.org/pdf/2407.08608) (see section 4.1).

## Starter Code

Our starter code accepts **device** pointers for the inputs and outputs, along with the problem size. For instance, the starter code for matrix multiplication is defined as:

```cpp
extern "C" void solution(float* input_a, float* input_b, float* output_c, 
                          size_t m, size_t n, size_t k) {
    
} 
```

This simple interface provides full flexibility - launch multiple kernels, control block/thread dimensions, and select algorithms as needed. Just compute the correct result using the provided device pointers.

## Runtime Calculation

First, we allocate GPU memory and transfer data from host to device **before** timing a solution. 


```cpp
// Allocate and transfer input data to GPU
for (size_t i = 0; i < test_case.input_shapes().size(); i++) {
    size_t size = test_case.input_shapes()[i]->size();
    cudaMalloc(&d_inputs[i], size * sizeof(T));
    cudaMemcpy(d_inputs[i], h_inputs[i], size * sizeof(T), cudaMemcpyHostToDevice);
}

// Allocate output memory on GPU
for (size_t i = 0; i < test_case.output_shapes().size(); i++) {
    size_t size = test_case.output_shapes()[i]->size();
    cudaMalloc(&d_outputs[i], size * sizeof(T));
}
```

`test_case.input_shapes()` and `test_case.output_shapes()` are defined in the problem specification (more details coming soon!).


Then, we use CUDA events to measure the solution's execution time with microsecond precision:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Timing code for each kernel execution
cudaEventRecord(start);
// Launch solution
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

For reliable benchmarking, we run each test case multiple times (10 runs by default) and take the average execution time. Before starting the timed runs, we perform a warm-up run to ensure the GPU is in a steady state.

