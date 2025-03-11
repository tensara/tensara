import asyncio
from functools import lru_cache, wraps
import os
import threading
from fastapi import HTTPException
import importlib
from problem import Problem
import torch
import time
import ctypes
import statistics
from fastapi.responses import StreamingResponse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict



GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90",
    "A100-80GB": "80",
    "A10G": "86",
}

class NVCCError(Exception):
    pass

def get_nvidia_smi():
    """Get nvidia-smi output"""
    process = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return str(process.stdout)

def nvcc_command(gpu: str, srcs: list[Path | str], out: Path | str):
    """Get nvcc command for the given GPU, source files, and output file"""
    srcs = [str(src) for src in srcs]
    out = str(out)
    sm = GPU_COMPUTE_CAPABILITIES[gpu]
    
    # Building the command similar to your Makefile
    cmd = ["nvcc", "-std=c++20", "-O2", "-Xcompiler", "-fPIC"]
    
    # Add architecture flags
    cmd.extend([f"-arch=compute_{sm}", f"-code=sm_{sm}"])
    
    # Add shared library flag since, we are building a shared library
    if str(out).endswith('.so'):
        cmd.append("-shared")
    
    # Add output file and source files
    cmd.extend(["-o", out] + srcs)
    
    return cmd

def hash_dict(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    class HDict(dict):
        def __hash__(self):
            return hash(frozenset(self.items()))

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([HDict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: HDict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapped

@hash_dict
@lru_cache(maxsize=512)  # each binary is ~1MB, so 512MB cache
def run_nvcc_and_return_bytes(gpu: str, solution_code: str, output_name: str) -> bytes:
    """Compile source files with nvcc and return the path to the compiled binary
    
    Args:
        gpu (str): GPU type to use
        files (dict[str, str]): Code files (file name -> content)
        output_name (str): Output library name
        
    Returns:
        Path: Path to the compiled shared library
        
    Raises:
        NVCCError: If compilation fails
    """
    # Create a temporary file for output that won't be deleted
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".lib{output_name}.so")
    output_file.close()
    out_path = Path(output_file.name)
    out_path.unlink()  # Remove the file so nvcc can create it
    
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        
        # Write the source files
        (path / "solution.cu").write_text(solution_code)
        
        # For a shared library, we need the solution.cu file
        src_path = path / "solution.cu"
        
        # Compile with nvcc
        cmd = nvcc_command(gpu, [src_path], out_path)
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for compilation errors
        if process.returncode != 0:
            raise NVCCError(process.stderr)
    
    bytes_of_file = out_path.read_bytes()
    out_path.unlink()
    return bytes_of_file


def read_bytes_as_cuda_lib(compiled_lib: bytes):
    """Read bytes of the solution code and compile it into a CUDA library"""
    if isinstance(compiled_lib, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.so')
        temp_file_path = temp_file.name
        try:
            temp_file.write(compiled_lib)
            temp_file.close()

            cuda_lib = ctypes.CDLL(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    else:
        cuda_lib = ctypes.CDLL(compiled_lib)
    
    return cuda_lib

def create_streaming_response(generator_func):
    """Create a FastAPI StreamingResponse from a generator function"""
    return StreamingResponse(
        generator_func(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

def format_benchmark_summary(benchmark_results):
    """Format the benchmark summary statistics"""
    passed_results = [r for r in benchmark_results if r["status"] == "PASSED"]
    
    if passed_results:
        avg_gflops = statistics.mean([r["performance_stats"]["mean_gflops"] for r in passed_results])
        max_gflops = max([r["performance_stats"]["max_gflops"] for r in passed_results])
        avg_throughput = statistics.mean([r["memory_stats"]["mean_throughput_gbps"] for r in passed_results])
    else:
        avg_gflops = 0
        max_gflops = 0
        avg_throughput = 0
    
    return {
        "avg_gflops": avg_gflops,
        "max_gflops": max_gflops,
        "avg_memory_throughput_gbps": avg_throughput,
        "total_tests": len(benchmark_results),
        "passed_tests": len(passed_results)
    }


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
        module_name = f"problems.{problem_type}"
        module = importlib.import_module(module_name)
        
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
    for _ in range(10):
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

    
    if len(runtimes) > 1:
        mean_runtime = statistics.mean(runtimes)
        stdev_runtime = statistics.stdev(runtimes)
        min_runtime = min(runtimes)
    else:
        mean_runtime = 0
        stdev_runtime = 0

    mean_gflops = statistics.mean(gflops_measurements)
    min_gflops = min(gflops_measurements)
    if len(gflops_measurements) > 1:
        stdev_gflops = statistics.stdev(gflops_measurements)
    else:
        stdev_gflops = 0
    
    

    benchmark_result = {
        "name": test_case.get("name", "Unknown"),
        "status": "PASSED",
        "iterations_run": len(runtimes),
        "gflops_stats": {
            "mean_gflops": mean_gflops,
            "min_gflops": min_gflops, 
            "stdev_gflops": stdev_gflops,
        },
        "runtime_stats": {
            "mean_ms": mean_runtime * 1000,  
            "min_ms": min_runtime * 1000,
            "stdev_ms": stdev_runtime * 1000,
        },
    }
    
    return benchmark_result


def async_wrap_iter(it):
    """
    Wrap blocking iterator into an asynchronous one

    From: https://stackoverflow.com/questions/62294385/synchronous-generator-in-asyncio
    """
    loop = asyncio.get_event_loop()
    q = asyncio.Queue(1)
    exception = None
    _END = object()

    async def yield_queue_items():
        while True:
            next_item = await q.get()
            if next_item is _END:
                break
            yield next_item
        if exception is not None:
            # the iterator has raised, propagate the exception
            raise exception

    def iter_to_queue():
        nonlocal exception
        try:
            for item in it:
                # This runs outside the event loop thread, so we
                # must use thread-safe API to talk to the queue.
                asyncio.run_coroutine_threadsafe(q.put(item), loop).result()
        except Exception as e:
            exception = e
        finally:
            asyncio.run_coroutine_threadsafe(q.put(_END), loop).result()

    threading.Thread(target=iter_to_queue).start()
    return yield_queue_items()

