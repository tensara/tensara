---
title: "Benchmarking Solutions"
date: "2025-03-31"
---

To benchmark solutions, we use [GigaFLOPS](https://en.wikipedia.org/wiki/Floating_point_operations_per_second) as the primary performance metric. FLOPS is defined as:
$$
\text{FLOPS} = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Runtime in seconds}}
$$

## Problem Definition

We define any problem as a class that implements the following methods using the `Problem` abstract base class:

```python
class SomeProblem(Problem):
    def __init__(self):
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
    def verify_result(self, expected_output: torch.Tensor, actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
    def get_function_signature(self) -> Dict[str, Any]:
    def get_flops(self, test_case: Dict[str, Any]) -> int:
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
```


## Calculating FLOPs

For each problem type, we have a `get_flops` function that computes the total number of floating point operations based on input dimensions. For example, the `get_flops` function for matrix multiplication of dimensions `m x k` and `k x n` is defined as:

```python
def get_flops(self, test_case: Dict[str, Any]) -> int:
    # One multiply and one add for each cell in the result, done K times
    M, N, K = test_case["dims"]
    return 2 * M * N * K
```
Since we now have a generalized `get_flops` function, we can measure performance across different input sizes and average the FLOPS across test cases. This approach is similar to ones used in papers like [Flash Attention 3](https://arxiv.org/pdf/2407.08608) (see section 4.1).

## Starter Code

We dynamically generate the starter code for each problem type using a function signature that is defined in the problem specification. For example, the function signature for matrix multiplication is defined as:

```python
def get_function_signature(self) -> Dict[str, Any]:
    return {
        "argtypes": [
            ctypes.POINTER(ctypes.c_float),  # matrix_a
            ctypes.POINTER(ctypes.c_float),  # matrix_b
            ctypes.POINTER(ctypes.c_float),  # matrix_c (output)
            ctypes.c_size_t,                 # M (rows in A and C)
            ctypes.c_size_t,                 # N (columns in B and C)
            ctypes.c_size_t                  # K (columns in A, rows in B)
        ],
    "restype": None
    }   
```

We use this to generate our starter code which accepts **device** pointers for the inputs and outputs, along with the extra arguments. For instance, the starter code for matrix multiplication will be generated as:

```cpp
extern "C" void solution(float* input_a, float* input_b, float* output_c, 
                          size_t m, size_t n, size_t k) {
    
} 
```

This simple interface provides full flexibility - launch multiple kernels, control block/thread dimensions, and select algorithms as needed. Just compute the correct result using the provided device pointers.

## Runtime Calculation

First, we generate test cases for a particular problem.
```python
test_cases = problem.generate_test_cases(dtype)
```

Then for each test case, we allocate memory for the input:
```python
input_tensors = test_case["create_inputs"]()
```

The, we get the expected output from the reference solution and move it to the CPU:
```python
expected_output = problem.reference_solution(*input_tensors).cpu()
```

Then, we create a `torch.Tensor` of the same shape and device as the expected output:
```python
actual_output = torch.zeros_like(expected_output, device='cuda')
```

Before benchmarking, we prepare the GPU for benchmarking by running a warm-up computations. We run 10 matrix multiplications of a 5000x5000 tensor with itself:
```python
warmup_tensor = torch.rand(5000, 5000, device='cuda')
for _ in range(10):
    torch.matmul(warmup_tensor, warmup_tensor.t())
torch.cuda.synchronize()
del warmup_tensor
torch.cuda.empty_cache()
```

### Runtime Measurement

We use CUDA events for precise measurement of kernel execution time:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
if language == "cuda":
    solution_func(*(input_ptrs + [output_ptr] + extra_params))
elif language == "python":
    solution_func(*(list(input_tensors) + [actual_output] + list(extra_params)))
end_event.record()
torch.cuda.synchronize()

initial_runtime = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
```

CUDA events give timings by recording timestamps directly on the GPU. The `torch.cuda.synchronize()` call ensures that all CUDA operations have completed before we measure the elapsed time.

## Benchmarking Process

To ensure a fair benchmark, we use a convergence based benchmark for kernels.

### Convergence Based Benchmark

We use a coefficient of variation (CV) approach to determine when we've collected enough samples:

```python
    mean_gflops = statistics.mean(gflops_measurements)
    
    # Can only calculate stdev with more than 1 sample
    if len(gflops_measurements) > 1:
        stdev_gflops = statistics.stdev(gflops_measurements)
        cv = stdev_gflops / mean_gflops if mean_gflops > 0 else float('inf')
        
        if cv < target_cv:
            break
```
The coefficient of variation (CV) is the ratio of the standard deviation to the mean. It tells us the relative variability of a set of benchmark results and allows us to determine when our measurements have stabilized. When the CV falls below a target threshold, we consider our benchmark results to be sufficiently reliable and stop collecting additional samples.

### Results
After collecting all the measurements, we calculate the mean runtime and GFLOPS:

```python
if len(runtimes) > 1:
    mean_runtime = statistics.mean(runtimes)
else:
    mean_runtime = runtimes[0]
mean_gflops = statistics.mean(gflops_measurements)
```

The final benchmark result includes:
- The test case name and ID
- The mean GFLOPS achieved
- The mean runtime in milliseconds

```python
benchmark_result = {
    "name": test_case["name"],
    "test_id": test_id,
    "gflops": mean_gflops,
    "runtime_ms": mean_runtime * 1000
}
```

Thanks for reading! If you have any questions, please reach out to us on [Discord](https://discord.gg/xudryk).