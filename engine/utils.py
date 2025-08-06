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
import math


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90",
    "H200": "90",
    "B200": "100",
    "A100-80GB": "80",
    "A10G": "86",
    "L40S": "89",
    "L4": "89",
}


class NVCCError(Exception):
    pass


class MojoError(Exception):
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

    cmd = ["nvcc", "-std=c++20", "-O2", "-Xcompiler", "-fPIC"]

    # architecture flags
    cmd.extend([f"-arch=compute_{sm}", f"-code=sm_{sm}"])

    if str(out).endswith(".so"):
        cmd.append("-shared")

    cmd.extend(["-o", out] + srcs)
    return cmd


def nvcc_command_executable(gpu: str, srcs: list[Path | str], out: Path | str):
    """Get nvcc command for compiling an executable

    Args:
        gpu (str): GPU type to use
        srcs (list): Source files to compile
        out (Path | str): Output executable path

    Returns:
        list: Command arguments for nvcc
    """

    srcs = [str(src) for src in srcs]
    out = str(out)
    sm = GPU_COMPUTE_CAPABILITIES[gpu]

    cmd = ["nvcc", "-std=c++20", "-O2", f"-arch=compute_{sm}", f"-code=sm_{sm}", "-o", out] + srcs

    return cmd


def mojo_command(srcs: list[Path | str], out: Path | str):
    """Get mojo command for the given source files and output file"""
    srcs = [str(src) for src in srcs]
    out = str(out)

    cmd = ["mojo", "build"]

    if str(out).endswith(".so"):
        cmd.append("--emit=shared-lib")

    cmd.extend(["-o", out] + srcs)

    return cmd


def sanitize_floats(obj):
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        elif math.isinf(obj):
            return "Infinity"
        else:
            return obj
    else:
        return obj


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
        solution_code (str): Code of the solution
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


# since mojo doesn't have GPU specific flags, we don't need gpu type as parameter
@hash_dict
@lru_cache(maxsize=512)  # each binary is ~1MB, so 512MB cache
def run_mojo_and_return_bytes(solution_code: str, output_name: str) -> bytes:
    """Compile source files with mojo and return the path to the compiled shared library

    Args:
        solution_code (str): Code of the solution
        output_name (str): Output library name

    Returns:
        Path: Path to the compiled shared library

    Raises:
        MojoError: If compilation fails
    """
    # Create a temporary file for output that won't be deleted
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".lib{output_name}.so")
    output_file.close()
    out_path = Path(output_file.name)
    out_path.unlink()  # Remove the file so mojo can create it

    with tempfile.TemporaryDirectory() as td:
        path = Path(td)

        (path / "solution.mojo").write_text(solution_code)
        src_path = path / "solution.mojo"

        cmd = mojo_command([src_path], out_path)
        process = subprocess.run(cmd, capture_output=True, text=True)

        # Check for compilation errors
        if process.returncode != 0:
            raise MojoError(process.stderr)

    bytes_of_file = out_path.read_bytes()
    out_path.unlink()
    return bytes_of_file


def read_bytes_as_lib(compiled_lib: bytes):
    """Read bytes of the solution code and compile it into a CUDA or Mojo library"""
    if isinstance(compiled_lib, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".so")
        temp_file_path = temp_file.name
        try:
            temp_file.write(compiled_lib)
            temp_file.close()

            lib = ctypes.CDLL(temp_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    else:
        lib = ctypes.CDLL(compiled_lib)

    return lib


def cast_to_ctype(data, argtypes, language="cuda"):
    """Cast data to ctypes"""
    data_casted = []
    if language == "cuda" or language == "mojo":
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
    solution_func,
    problem,
    test_id,
    test_case,
    input_tensors,
    actual_output,
    language="cuda",
    min_iterations=5,
    max_iterations=15,
    target_cv=0.02,
    long_kernel_threshold=1.0,
    param_func=None,
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
        language: Programming language of the solution ("cuda", "mojo", or "python")
        min_iterations: Minimum number of iterations to run
        max_iterations: Maximum number of iterations to run
        target_cv: Target coefficient of variation to achieve
        long_kernel_threshold: Time in seconds above which CV convergence is skipped

    Returns:
        benchmark_result: Dictionary with benchmark results
    """
    # Prepare pointers for CUDA
    if param_func is None:
        parameters = make_parameters(
            language, solution_func, input_tensors, actual_output, problem, test_case
        )
    else:
        parameters = param_func(
            language, solution_func, input_tensors, actual_output, problem, test_case
        )

    # Calculate FLOPS for this test case
    flops = problem.get_flops(test_case)

    # Warm up run
    prepare_gpu()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    solution_func(*parameters)
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
        solution_func(*parameters)

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
                cv = stdev_gflops / mean_gflops if mean_gflops > 0 else float("inf")

                if cv < target_cv:
                    break

    if len(runtimes) > 1:
        mean_runtime = statistics.mean(runtimes)
    else:
        mean_runtime = runtimes[0]

    mean_gflops = statistics.mean(gflops_measurements)

    benchmark_result = {
        "name": test_case["name"],
        "test_id": test_id,
        "gflops": mean_gflops,
        "runtime_ms": mean_runtime * 1000,
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


def make_solution_func(language: str, solution_code: str, compiled: bytes, problem: Problem):
    if language == "cuda":
        if not compiled:
            raise ValueError("Compiled bytes required for CUDA submissions")

        cuda_lib = read_bytes_as_lib(compiled)
        func_sig = problem.get_function_signature()
        cuda_lib.solution.argtypes = func_sig["argtypes"]
        cuda_lib.solution.restype = func_sig["restype"]
        return cuda_lib.solution

    elif language == "mojo":
        mojo_lib = read_bytes_as_lib(compiled)
        func_sig = problem.get_function_signature()
        mojo_lib.solution.argtypes = func_sig["argtypes"]
        mojo_lib.solution.restype = func_sig["restype"]
        return mojo_lib.solution

    elif language == "python":
        if not solution_code:
            raise ValueError("Source code required for Triton submissions")

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "triton_solution.py")

        # This is needed because @jit has to read the source code
        with open(temp_path, "w") as f:
            f.write(solution_code)

        spec = importlib.util.spec_from_file_location("triton_solution", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.solution
    else:
        raise ValueError(f"Unsupported language: {language}")


def make_parameters(language: str, solution_func, input_tensors, actual_output, problem, test_case):
    if language == "cuda" or language == "mojo":
        input_ptrs = cast_to_ctype(
            input_tensors, solution_func.argtypes[: len(input_tensors)], language
        )
        output_ptr = ctypes.cast(actual_output.data_ptr(), solution_func.argtypes[len(input_ptrs)])
        extra_params = problem.get_extra_params(test_case)
        extra_params_casted = cast_to_ctype(
            extra_params, solution_func.argtypes[-len(extra_params) :], language
        )
        return input_ptrs + [output_ptr] + extra_params_casted
    else:
        extra_params = problem.get_extra_params(test_case)
        return list(input_tensors) + [actual_output] + list(extra_params)


class SystemOutputCapture:
    """Class to capture system-level stdout/stderr including CUDA printf"""

    def __init__(self):
        self.stdout_content = ""
        self.stderr_content = ""

    def __enter__(self):
        # Create temp files
        self.tmp_out = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.tmp_err = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.tmp_out_path = self.tmp_out.name
        self.tmp_err_path = self.tmp_err.name

        # Backup original stdout/stderr
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)

        # Redirect to temp files
        os.dup2(self.tmp_out.fileno(), 1)
        os.dup2(self.tmp_err.fileno(), 2)

        # Close the temp file handles since we've redirected
        self.tmp_out.close()
        self.tmp_err.close()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Force CUDA to flush
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        # Restore stdout/stderr
        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)
        os.close(self.old_stdout)
        os.close(self.old_stderr)

        # Read captured output
        try:
            with open(self.tmp_out_path, "r") as f:
                self.stdout_content = f.read()
            with open(self.tmp_err_path, "r") as f:
                self.stderr_content = f.read()
        except Exception:
            self.stdout_content = ""
            self.stderr_content = ""

        # Clean up
        try:
            os.unlink(self.tmp_out_path)
            os.unlink(self.tmp_err_path)
        except OSError:
            pass


@hash_dict
@lru_cache(maxsize=512)
def run_nvcc_and_return_executable(gpu: str, solution_code: str) -> bytes:
    """Compile source files with nvcc and return the bytes of an executable binary

    Args:
        gpu (str): GPU type to use
        solution_code (str): Code of the solution

    Returns:
        bytes: Bytes of the compiled executable

    Raises:
        NVCCError: If compilation fails
    """
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    output_file.close()
    out_path = Path(output_file.name)
    out_path.unlink()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td)

        (path / "solution.cu").write_text(solution_code)
        src_path = path / "solution.cu"

        cmd = nvcc_command_executable(gpu, [src_path], out_path)

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            raise NVCCError(process.stderr)

    bytes_of_file = out_path.read_bytes()
    out_path.unlink()
    return bytes_of_file
