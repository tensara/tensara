---
title: "Benchmarking Solutions"
date: "2025-03-31"
---

To benchmark solutions, we use [GFLOPS](https://en.wikipedia.org/wiki/Floating_point_operations_per_second) (GigaFLOPS) as the primary performance metric. GFLOPS is defined as:
$$
\text{GFLOPS} = \frac{\text{Total Floating Point Operations (FLOPs)}}{10^9 \times \text{Runtime in seconds}}
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

For each problem type, we have an analytical `get_flops` function that estimates the total number of floating point operations based on input dimensions and problem structure. **Important**: This is an analytical estimation, not a profiled measurement of actual operations executed by the GPU.

For example, matrix multiplication of dimensions `m × k` and `k × n`:

```python
def get_flops(self, test_case: Dict[str, Any]) -> int:
    M, N, K = test_case["dims"]
    return 2 * M * N * K  # One multiply and one add per output element
```

GFLOPS is calculated as: `(analytical_flops / runtime) / 1e9`. Since we average GFLOPS across test cases, this approach is similar to ones used in papers like [Flash Attention 3](https://arxiv.org/pdf/2407.08608) (see section 4.1).

**Note**: Not all problems support FLOPS calculation. Problems that don't implement `supports_flops()` will only report runtime metrics.

## Starter Code

We dynamically generate starter code using the problem's function signature. For matrix multiplication, the signature is:

```python
def get_function_signature(self) -> Dict[str, Any]:
    return {
        "argtypes": [
            ctypes.POINTER(ctypes.c_float),  # matrix_a
            ctypes.POINTER(ctypes.c_float),  # matrix_b
            ctypes.POINTER(ctypes.c_float),  # matrix_c (output)
            ctypes.c_size_t,                 # M
            ctypes.c_size_t,                 # N
            ctypes.c_size_t                  # K
        ],
        "restype": None
    }
```

This generates starter code with device pointers for inputs and outputs:

```cpp
extern "C" void solution(float* input_a, float* input_b, float* output_c, 
                          size_t m, size_t n, size_t k) {
    
}
```

This interface provides full flexibility: launch multiple kernels, control block/thread dimensions, and select algorithms as needed.

## Benchmark Setup

Before benchmarking, we prepare the GPU by clearing caches and running a warm-up workload to stabilize temperature:

```python
def prepare_gpu():
    torch.cuda.empty_cache()
    warmup_tensor = torch.rand(5000, 5000, device='cuda')
    for _ in range(10):
        torch.matmul(warmup_tensor, warmup_tensor.t())
    torch.cuda.synchronize()
    del warmup_tensor
    torch.cuda.empty_cache()
    time.sleep(0.5)
```

Warm-up is performed once at the start and again before each test case.

## Runtime Measurement

We use CUDA events to measure kernel execution time directly on the GPU:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
solution_func(*parameters)
end_event.record()
torch.cuda.synchronize()

runtime = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
```

The `torch.cuda.synchronize()` call ensures all CUDA operations complete before measuring elapsed time.

## Benchmarking Process

We use a dynamic convergence-based approach that adapts to kernel execution time:

1. Run the kernel once to measure initial runtime
2. **Short kernels** (< 1 second): Use convergence-based approach with up to 20 iterations
3. **Long kernels** (≥ 1 second): Use fixed number of iterations (~12) to avoid excessive time

For short kernels, we use coefficient of variation (CV) to determine when measurements have stabilized:

```python
mean_val = statistics.mean(gflops_measurements)
if len(gflops_measurements) > 1:
    stdev_val = statistics.stdev(gflops_measurements)
    cv = stdev_val / mean_val if mean_val > 0 else float('inf')
    if cv < target_cv:  # Default: 0.01 (1%)
        break
```

**Default Parameters**:
- Minimum iterations: 5
- Maximum iterations: 20
- Target CV: 0.01 (1%)
- Long kernel threshold: 1.0 second

After collecting measurements, we calculate mean runtime and mean GFLOPS:

```python
mean_runtime = statistics.mean(runtimes) if len(runtimes) > 1 else runtimes[0]
mean_gflops = statistics.mean(gflops_measurements) if gflops_measurements else None

benchmark_result = {
    "name": test_case["name"],
    "test_id": test_id,
    "runtime_ms": mean_runtime * 1000,
    "gflops": mean_gflops  # Only if problem supports FLOPS
}
```

After all test cases, we calculate average runtime and average GFLOPS across all test cases.

Thanks for reading! If you have any questions, please reach out to us on [Discord](https://discord.gg/YzBTfMxVQK).