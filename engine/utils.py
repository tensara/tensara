from functools import lru_cache, wraps
import os
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
import importlib.util
from types import ModuleType
import multiprocessing as mp
import queue


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90",
    "A100-80GB": "80",
    "A10G": "86",
    "L40S": "89",
    "L4": "89",
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
    if str(out).endswith(".so"):
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
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".so")
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


def cast_to_ctype(data, argtypes, language="cuda"):
    """Cast data to ctypes"""
    data_casted = []
    if language == "cuda":
        for tensor, argtype in zip(data, argtypes):
            if isinstance(tensor, torch.Tensor):
                data_casted.append(ctypes.cast(tensor.data_ptr(), argtype))
            else:
                data_casted.append(argtype(tensor))
        return data_casted
    else:
        return data


def load_problem_module(problem_type: str, problem_def: str = None) -> Problem:
    """
    Load a Problem module either from a string definition or from pre-imported problems.

    Args:
        problem_type: String identifier for the problem (e.g., "matrix_multiplication")
        problem_def: Optional string containing the Python module definition

    Returns:
        An instantiated Problem subclass

    Raises:
        HTTPException: If the problem type cannot be found or loaded
    """
    try:
        if problem_def is not None:
            spec = importlib.util.spec_from_loader(problem_type, loader=None, origin="<string>")
            module = ModuleType(spec.name)
            exec(problem_def, module.__dict__)

            problem_class = getattr(module, problem_type)
            return problem_class()

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Problem type '{problem_type}' not found or failed to load: {str(e)}",
        )


def prepare_gpu():
    """
    Prepare the GPU for consistent benchmarking with a simple warm-up.
    """
    # Clear GPU caches
    torch.cuda.empty_cache()

    # Run a moderate workload to heat up the GPU to a stable temperature
    warmup_tensor = torch.rand(5000, 5000, device="cuda")
    for _ in range(10):
        torch.matmul(warmup_tensor, warmup_tensor.t())
    torch.cuda.synchronize()
    del warmup_tensor
    torch.cuda.empty_cache()

    time.sleep(0.5)


def run_dynamic_benchmark(
    solution_func, problem, test_id, test_case, input_tensors, actual_output, language="cuda"
):
    """
    Run a CUDA benchmark with dynamic stopping based on GFLOPS variance.
    If kernel execution time exceeds threshold, run fixed number of iterations instead.

    Args:
        solution_func: CUDA library with the solution function
        problem: Problem definition with verification methods
        test_case: The specific test case to benchmark
        input_tensors: Input tensors for the CUDA function
        actual_output: Output tensor for the CUDA function
        language: Programming language of the solution ("cuda" or "python")
        min_iterations: Minimum number of iterations to run
        max_iterations: Maximum number of iterations to run
        target_cv: Target coefficient of variation to achieve
        long_kernel_threshold: Time in seconds above which CV convergence is skipped

    Returns:
        benchmark_result: Dictionary with benchmark results
    """
    # Prepare pointers for CUDA
    extra_params = problem.get_extra_params(test_case)
    if language == "cuda":
        input_ptrs = cast_to_ctype(
            input_tensors, solution_func.argtypes[: len(input_tensors)], language
        )
        output_ptr = ctypes.cast(
            actual_output.data_ptr(), solution_func.argtypes[len(input_tensors)]
        )
        extra_params_casted = cast_to_ctype(
            extra_params, solution_func.argtypes[-len(extra_params) :], language
        )
    # Calculate FLOPS for this test case
    flops = problem.get_flops(test_case)

    runtimes = []
    min_iters = 3
    max_runtime = 1

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start = time.time()
    while time.time() - start < max_runtime or len(runtimes) < min_iters:
        start_event.record()

        if language == "cuda":
            solution_func(*(input_ptrs + [output_ptr] + extra_params_casted))
        elif language == "python":
            solution_func(*(list(input_tensors) + [actual_output] + list(extra_params)))

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        runtimes.append(elapsed_time)

    best_runtime = min(runtimes)
    best_gflops = (flops / best_runtime) / 1e9  # Convert to GFLOPS

    benchmark_result = {
        "name": test_case["name"],
        "test_id": test_id,
        "gflops": best_gflops,
        "runtime_ms": len(runtimes),
    }

    return benchmark_result


def convert_slug_to_module_name(slug: str) -> str:
    """
    Convert a problem slug to a module name
    """
    return slug.replace("-", "_")


def subproc_generator(timeout=None):
    def _subproc_generator(func):
        def subproc_wrapper(my_queue, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                for ev in result:
                    my_queue.put(ev)
            finally:
                my_queue.put(None)

        @wraps(func)
        def wrapper(*args, **kwargs):
            my_queue = mp.Queue()
            proc = mp.Process(target=subproc_wrapper, args=(my_queue,) + args, kwargs=kwargs)
            proc.start()
            while True:
                try:
                    event = my_queue.get(timeout=timeout)
                    yield event
                    if event is None:
                        break
                except queue.Empty:
                    yield {
                        "status": "TIME_LIMIT_EXCEEDED",
                        "message": "Time Limit Exceeded",
                        "details": f"Execution exceeded time limit of {timeout:.2f}s",
                    }
                    proc.terminate()
                    break

            proc.join()
            proc.close()

        return wrapper

    return _subproc_generator
