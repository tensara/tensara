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
import subprocess
import tempfile
from pathlib import Path



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
    
    time.sleep(0.5)

def run_dynamic_benchmark(cuda_lib, problem, test_case, input_tensors, actual_output, 
                          min_iterations=5, max_iterations=15, target_cv=0.02, long_kernel_threshold=1.0):
    """
    Run a CUDA benchmark with dynamic stopping based on GFLOPS variance.
    If kernel execution time exceeds threshold, run fixed number of iterations instead.
    
    Args:
        cuda_lib: CUDA library with the solution function
        problem: Problem definition with verification methods
        test_case: The specific test case to benchmark
        input_tensors: Input tensors for the CUDA function
        actual_output: Output tensor for the CUDA function
        min_iterations: Minimum number of iterations to run
        max_iterations: Maximum number of iterations to run
        target_cv: Target coefficient of variation to achieve
        long_kernel_threshold: Time in seconds above which CV convergence is skipped
        
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
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    cuda_lib.solution(*(input_ptrs + [output_ptr] + extra_params))
    end_event.record()
    torch.cuda.synchronize()
    
    initial_runtime = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    
    # Determine if this is a long-running kernel and how many iterations to run
    is_long_kernel = initial_runtime >= long_kernel_threshold
    
    if is_long_kernel:
        # For long kernels, use fixed number of iterations
        target_iterations = (min_iterations + max_iterations) // 2
    else:
        # For short kernels, use CV-based convergence with max_iterations cap
        target_iterations = max_iterations
    
    # Collect runtime measurements
    runtimes = [initial_runtime]  # Include the initial runtime
    gflops_measurements = [(flops / initial_runtime) / 1e9]  # Convert to GFLOPS

    for iteration in range(1, target_iterations):  # Start from 1 since we already did one iteration
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Start timing
        start_event.record()
        
        # Run the kernel
        cuda_lib.solution(*(input_ptrs + [output_ptr] + extra_params))
        
        # End timing
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        runtimes.append(elapsed_time)
        
        # Calculate GFLOPS
        gflops = (flops / elapsed_time) / 1e9  # Convert to GFLOPS
        gflops_measurements.append(gflops)
        
        # Check if we've done enough iterations and the variance is low enough
        # Only do this check for short kernels
        if not is_long_kernel and iteration + 1 >= min_iterations:
            mean_gflops = statistics.mean(gflops_measurements)
            
            # Can only calculate stdev with more than 1 sample
            if len(gflops_measurements) > 1:
                stdev_gflops = statistics.stdev(gflops_measurements)
                cv = stdev_gflops / mean_gflops if mean_gflops > 0 else float('inf')
                
                if cv < target_cv:
                    break

    
    if len(runtimes) > 1:
        mean_runtime = statistics.mean(runtimes)
        stdev_runtime = statistics.stdev(runtimes)
        min_runtime = min(runtimes)
    else:
        mean_runtime = runtimes[0]
        stdev_runtime = 0
        min_runtime = runtimes[0]

    mean_gflops = statistics.mean(gflops_measurements)
    min_gflops = min(gflops_measurements)
    if len(gflops_measurements) > 1:
        stdev_gflops = statistics.stdev(gflops_measurements)
    else:
        stdev_gflops = 0
    
    benchmark_result = {
        "status": "PASSED",
        "gflops": mean_gflops,
        "runtime_ms": mean_runtime * 1000,
        "stdev_gflops": stdev_gflops,
    }
    
    return benchmark_result

def convert_slug_to_module_name(slug: str) -> str:
    """
    Convert a problem slug to a module name
    """
    return slug.replace("-", "_")

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

