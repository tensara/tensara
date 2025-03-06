from fastapi import HTTPException, status
import importlib
from typing import List, Dict, Any
from problem import Problem
import problems
import torch
import time
import ctypes


def load_problem_module(problem_type: str) -> Problem:
    """
    Load a Problem module from the pre-imported problems module.
    
    Args:
        problem_type: String identifier for the problem (e.g., "matrix_multiplication")
        
    Returns:
        An instantiated Problem subclass
    
    Raises:
        HTTPException: If the problem type cannot be found
    """
    try:
        # Convert problem_type to the appropriate attribute name
        module_name = f"problems.{problem_type}"
        module = importlib.import_module(module_name)
        
        # Get the problem class from the problems module
        problem_class = getattr(module, problem_type)
        return problem_class()
    
    except AttributeError as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Problem type '{problem_type}' not found: {str(e)}"
        )

def prepare_gpu():
    """
    Prepare the GPU for consistent benchmarking with a simple warm-up.
    """
    # Clear GPU caches
    torch.cuda.empty_cache()
    
    # Run a moderate workload to heat up the GPU to a stable temperature
    warmup_tensor = torch.rand(5000, 5000, device='cuda')
    for _ in range(5):
        torch.matmul(warmup_tensor, warmup_tensor.t())
    torch.cuda.synchronize()
    del warmup_tensor
    torch.cuda.empty_cache()
    
    # Let the GPU rest please
    time.sleep(1)

def lower_bound_memory_throughput(input_tensors, output_tensor, runtime_seconds):
    """
    Calculate memory throughput for a CUDA kernel execution.
    
    Args:
        input_tensors: List of input tensors
        output_tensor: Output tensor
        runtime_seconds: Runtime in seconds
        
    Returns:
        memory_throughput_gbps: Memory throughput in GB/s
    """
    # Calculate bytes read (inputs)
    input_bytes = sum(tensor.nelement() * tensor.element_size() for tensor in input_tensors)
    
    # Calculate bytes written (output)
    output_bytes = output_tensor.nelement() * output_tensor.element_size()
    
    # Total bytes transferred
    total_bytes = input_bytes + output_bytes
    
    # Calculate throughput in GB/s
    memory_throughput_gbps = (total_bytes / runtime_seconds) / 1e9
    
    return memory_throughput_gbps, input_bytes, output_bytes


def run_dynamic_benchmark(cuda_lib, problem, test_case, input_tensors, actual_output, 
                          min_iterations=5, max_iterations=15, target_cv=0.02):
    """
    Run a CUDA benchmark with dynamic stopping based on GFLOPS variance.
    
    Args:
        cuda_lib: CUDA library with the solution function
        problem: Problem definition with verification methods
        test_case: The specific test case to benchmark
        input_tensors: Input tensors for the CUDA function
        actual_output: Output tensor for the CUDA function
        min_iterations: Minimum number of iterations to run
        max_iterations: Maximum number of iterations to run
        target_cv: Target coefficient of variation to achieve
        
    Returns:
        benchmark_result: Dictionary with benchmark results
    """
    # Prepare pointers for CUDA
    input_ptrs = [ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)) 
                 for tensor in input_tensors]
    output_ptr = ctypes.cast(actual_output.data_ptr(), ctypes.POINTER(ctypes.c_float))
    extra_params = problem.get_extra_params(test_case)
    
    # Calculate FLOPS for this test case
    flops = problem.get_flops(test_case)
    
    # Warm up run
    cuda_lib.solution(*(input_ptrs + [output_ptr] + extra_params))
    torch.cuda.synchronize()
    
    # Collect runtime measurements
    runtimes = []
    gflops_measurements = []
    memory_throughputs = []
    
    print(f"Running dynamic benchmark with {min_iterations}-{max_iterations} iterations (target CV: {target_cv})")
    
    # Run iterations and collect measurements
    for iteration in range(max_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Start timing
        start_event.record()
        
        # Run the kernel
        cuda_lib.solution(*(input_ptrs + [output_ptr] + extra_params))
        
        # End timing
        end_event.record()
        torch.cuda.synchronize()
        
        # Get elapsed time in seconds
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        runtimes.append(elapsed_time)
        
        # Calculate GFLOPS
        gflops = (flops / elapsed_time) / 1e9  # Convert to GFLOPS
        gflops_measurements.append(gflops)
        
        # Calculate memory throughput
        throughput, _, _ = lower_bound_memory_throughput(input_tensors, actual_output, elapsed_time)
        memory_throughputs.append(throughput)
        
        # Print progress
        print(f"  Iteration {iteration+1}: {elapsed_time*1000:.2f} ms, {gflops:.2f} GFLOPS, {throughput:.2f} GB/s")
        
        # Check if we've done enough iterations and the variance is low enough
        if iteration + 1 >= min_iterations:
            mean_gflops = statistics.mean(gflops_measurements)
            
            # Can only calculate stdev with more than 1 sample
            if len(gflops_measurements) > 1:
                stdev_gflops = statistics.stdev(gflops_measurements)
                cv = stdev_gflops / mean_gflops if mean_gflops > 0 else float('inf')
                
                print(f"  Current CV: {cv:.4f} (target: {target_cv})")
                
                if cv < target_cv:
                    print(f"  Reached target coefficient of variation after {iteration+1} iterations")
                    break
    
    # Handle statistics
    mean_runtime = statistics.mean(runtimes)
    min_runtime = min(runtimes)  # Often a better indicator of peak performance
    median_runtime = statistics.median(runtimes)
    
    if len(runtimes) > 1:
        stdev_runtime = statistics.stdev(runtimes)
        runtime_cv = stdev_runtime / mean_runtime
    else:
        stdev_runtime = 0
        runtime_cv = 0
    
    # Calculate GFLOPS statistics
    mean_gflops = statistics.mean(gflops_measurements)
    max_gflops = max(gflops_measurements)
    if len(gflops_measurements) > 1:
        stdev_gflops = statistics.stdev(gflops_measurements)
        gflops_cv = stdev_gflops / mean_gflops
    else:
        stdev_gflops = 0
        gflops_cv = 0
    
    # Memory throughput statistics
    mean_throughput = statistics.mean(memory_throughputs)
    max_throughput = max(memory_throughputs)
    
    

    benchmark_result = {
        "test_id": test_case.get("id", 0),
        "name": test_case.get("name", "Unknown"),
        "status": "PASSED",
        "iterations_run": len(runtimes),
        "runtime_stats": {
            "mean_ms": mean_runtime * 1000,  
            "min_ms": min_runtime * 1000,
            "median_ms": median_runtime * 1000,
            "stdev_ms": stdev_runtime * 1000,
            "cv": runtime_cv
        },
        "performance_stats": {
            "flops": flops,
            "mean_gflops": mean_gflops,
            "max_gflops": max_gflops, 
            "stdev_gflops": stdev_gflops,
            "cv": gflops_cv
        },
        "memory_stats": {
            "mean_throughput_gbps": mean_throughput,
            "max_throughput_gbps": max_throughput,
        },
        "all_measurements": {
            "runtimes_ms": [rt * 1000 for rt in runtimes],
            "gflops": gflops_measurements,
            "throughputs_gbps": memory_throughputs
        }
    }
    
    return benchmark_result