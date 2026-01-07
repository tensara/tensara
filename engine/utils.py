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
from numbers import Integral
import numpy as np
import threading
from collections import deque

JS_MAX_SAFE = 2**53 - 1

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90a",
    "H200": "90a",
    "B200": "100a",
    "A100-80GB": "80",
    "A10G": "86",
    "L40S": "89",
    "L4": "89",
}


# Container state tracking for diagnostics
_CONTAINER_START_TIME = time.time()
_REQUEST_COUNTER = 0


def get_container_diagnostics() -> dict:
    """Get container-level state for diagnostics."""
    global _REQUEST_COUNTER
    _REQUEST_COUNTER += 1
    return {
        "is_first_request": _REQUEST_COUNTER == 1,
        "request_number": _REQUEST_COUNTER,
        "container_uptime_s": round(time.time() - _CONTAINER_START_TIME, 2),
    }


class NVCCError(Exception):
    pass


class MojoError(Exception):
    pass


def get_nvidia_smi():
    """Get nvidia-smi output"""
    process = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return str(process.stdout)


def get_gpu_diagnostics() -> dict:
    """Query GPU state via nvidia-smi for diagnostics."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,temperature.memory,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,clocks_throttle_reasons.active,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split(", ")
        if len(parts) >= 11:
            # Helper to parse throttle reason fields (they can be "Active" or "Not Active" or "[Not Supported]")
            def parse_throttle_reason(val):
                val = val.strip()
                if val == "[Not Supported]":
                    return None
                return val == "Active"
            
            for i in range(len(parts)):
                if parts[i] == "N/A":
                    parts[i] = 0

            return {
                "gpu_temp_c": int(parts[0]),
                "gpu_mem_temp_c": int(parts[1]),
                "gpu_pstate": parts[2].strip(),
                "gpu_sm_clock_mhz": int(parts[3]),
                "gpu_mem_clock_mhz": int(parts[4]),
                "gpu_power_w": float(parts[5]),
                "gpu_power_limit_w": float(parts[6]),
                "gpu_throttle_active": parse_throttle_reason(parts[7]),
                "gpu_throttle_hw_thermal": parse_throttle_reason(parts[8]),
                "gpu_throttle_sw_thermal": parse_throttle_reason(parts[9]),
                "gpu_throttle_sw_power_cap": parse_throttle_reason(parts[10]),
                "gpu_utilization_pct": int(parts[11]),
                "gpu_memory_utilization_pct": int(parts[12]),
            }
        # Fallback for older nvidia-smi versions that might not support all fields
        elif len(parts) >= 6:
            return {
                "gpu_temp_c": int(parts[0]) if len(parts) > 0 else 0,
                "gpu_sm_clock_mhz": int(parts[3]) if len(parts) > 3 else 0,
                "gpu_mem_clock_mhz": int(parts[4]) if len(parts) > 4 else 0,
                "gpu_power_w": float(parts[5]) if len(parts) > 5 else 0.0,
                "gpu_utilization_pct": int(parts[11]) if len(parts) > 11 else 0,
            }
    return {}


def get_gpu_clock_and_temp() -> tuple[int, int]:
    """Get current SM clock (MHz) and temperature (C). Returns (0, 0) on error."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.sm,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


class GPUMonitor:
    """Monitor GPU metrics using pynvml in a separate thread during kernel execution."""
    
    def __init__(self, device_id=0, sample_interval_ms=1):
        """
        Initialize GPU monitor.
        
        Args:
            device_id: GPU device ID to monitor
            sample_interval_ms: Sampling interval in milliseconds
        """
        self.device_id = device_id
        self.sample_interval = sample_interval_ms / 1000.0  # Convert to seconds
        self.monitoring = False
        self.samples = deque()
        self.thread = None
        self.handle = None
        self.lock = threading.Lock()
        self._init_pynvml()
    
    def _init_pynvml(self):
        """Initialize pynvml library."""
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self.pynvml = pynvml
        except ImportError:
            # pynvml not installed - set to None, methods will handle gracefully
            self.handle = None
            self.pynvml = None
        except Exception:
            # Fallback if pynvml fails to initialize
            self.handle = None
            self.pynvml = None
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        if not self.pynvml or not self.handle:
            return
        
        while self.monitoring:
            try:
                # Get current timestamp
                timestamp = time.time()
                
                # Query GPU metrics
                try:
                    # Get clock speeds
                    sm_clock = self.pynvml.nvmlDeviceGetClockInfo(self.handle, self.pynvml.NVML_CLOCK_SM)
                    mem_clock = self.pynvml.nvmlDeviceGetClockInfo(self.handle, self.pynvml.NVML_CLOCK_MEM)
                    
                    # Get temperature
                    temp = self.pynvml.nvmlDeviceGetTemperature(self.handle, self.pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Get power usage
                    power = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert mW to W
                    
                    # Get utilization
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    
                    # Get performance state
                    pstate = self.pynvml.nvmlDeviceGetPerformanceState(self.handle)
                    
                    # Get clock throttle reasons
                    throttle_reasons = self.pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
                    
                    sample = {
                        "timestamp": timestamp,
                        "sm_clock_mhz": sm_clock,
                        "mem_clock_mhz": mem_clock,
                        "temp_c": temp,
                        "power_w": power,
                        "utilization_gpu_pct": util.gpu,
                        "utilization_memory_pct": util.memory,
                        "pstate": pstate,
                        "throttle_reasons": throttle_reasons,
                    }
                    
                    with self.lock:
                        self.samples.append(sample)
                        
                except Exception as e:
                    # If query fails, continue monitoring
                    pass
                
                # Sleep for sample interval
                time.sleep(self.sample_interval)
                
            except Exception:
                # If monitoring loop fails, break out
                break
    
    def start(self):
        """Start monitoring in a separate thread."""
        if not self.pynvml or not self.handle:
            return
        
        if self.monitoring:
            return  # Already monitoring
        
        self.monitoring = True
        with self.lock:
            self.samples.clear()
        
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return collected samples."""
        if not self.monitoring:
            return []
        
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        with self.lock:
            samples = list(self.samples)
            self.samples.clear()
        
        return samples
    
    def get_samples(self):
        """Get current samples without stopping monitoring."""
        with self.lock:
            return list(self.samples)
    
    def get_stats_for_period(self, start_time, end_time, samples=None):
        """
        Get statistics for a specific time period.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            samples: Optional list of samples to use instead of self.samples
            
        Returns:
            Dictionary with statistics (min, max, mean) for each metric
        """
        if samples is None:
            with self.lock:
                samples_to_use = list(self.samples)
        else:
            samples_to_use = samples
            
        period_samples = [
            s for s in samples_to_use 
            if start_time <= s["timestamp"] <= end_time
        ]
        
        if not period_samples:
            return {}
        
        stats = {}
        metrics = ["sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w", 
                   "utilization_gpu_pct", "utilization_memory_pct"]
        
        for metric in metrics:
            values = [s[metric] for s in period_samples if metric in s]
            if values:
                stats[f"{metric}_min"] = min(values)
                stats[f"{metric}_max"] = max(values)
                stats[f"{metric}_mean"] = statistics.mean(values) if len(values) > 1 else values[0]
                stats[f"{metric}_samples"] = len(values)
        
        return stats


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


def _scan_cuda_forbidden(source: str) -> str | None:
    """Return an error message if source contains forbidden CUDA/C++ tokens.

    Forbidden tokens per policy: #include <thrust/ , thrust::, std::sort, std::stable_sort, qsort(
    This scanner strips C/C++ comments and string literals first so commented-out occurrences are ignored.
    """
    if not isinstance(source, str):
        return None

    import re

    def _strip_comments_and_strings(s: str) -> str:
        # remove string literals (double-quoted then single-quoted) using two safe regex calls
        s = re.sub(r'"(?:\\.|[^"\\])*"', "", s, flags=re.DOTALL)
        s = re.sub(r"'(?:\\.|[^'\\])*'", "", s, flags=re.DOTALL)
        # remove block comments
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        # remove line comments
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
        return s

    cleaned = _strip_comments_and_strings(source)

    patterns = [
        r"#\s*include\s*<thrust/",
        r"\bthrust::",
        r"\bstd::sort\b",
        r"\bstd::stable_sort\b",
        r"\bqsort\s*\(",
    ]

    for p in patterns:
        if re.search(p, cleaned):
            return f"Forbidden C++/CUDA usage detected: pattern '{p}' found. This project disallows Thrust, std::sort/std::stable_sort and qsort; please implement using allowed primitives."

    return None


def _scan_triton_python_forbidden(source: str) -> str | None:
    if not isinstance(source, str):
        return None

    import ast

    try:
        tree = ast.parse(source)
    except Exception:
        # Can't parse -> be conservative
        return "Unable to parse Python source for safety checks — rejecting submission."

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.msg = None

        def visit_Attribute(self, node: ast.Attribute):
            # e.g., tl.sort, torch.sort, torch.topk
            try:
                if isinstance(node.attr, str) and node.attr in ("sort", "topk"):
                    if isinstance(node.value, ast.Name) and node.value.id in ("tl", "torch"):
                        self.msg = f"Forbidden call to '{node.value.id}.{node.attr}' detected."
            except Exception:
                pass
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            # detect eval/exec/open/__import__
            if isinstance(node.func, ast.Name) and node.func.id in (
                "eval",
                "exec",
                "open",
                "__import__",
            ):
                self.msg = f"Forbidden builtin '{node.func.id}' used in Python submission."

            # detect importlib.*(...) calls
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                self.msg = "Forbidden use of importlib detected."

            self.generic_visit(node)

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                if alias.name == "thrust":
                    self.msg = "Import of 'thrust' is forbidden."

        def visit_ImportFrom(self, node: ast.ImportFrom):
            if node.module == "thrust" or (
                isinstance(node.module, str) and node.module.endswith(".thrust")
            ):
                self.msg = "Import of 'thrust' is forbidden."
            # detect 'from builtin.sort import sort' as a trick
            if node.module and node.module.endswith("builtin"):
                for alias in node.names:
                    if alias.name == "sort":
                        self.msg = "Importing builtin sort is forbidden."

    v = Visitor()
    v.visit(tree)

    return v.msg


def _scan_mojo_forbidden(source: str) -> str | None:
    """Text-based scan to detect Mojo forbidden patterns: builtin.sort imports or bare sort on Span types.

    This is heuristic: we look for "from builtin.sort import sort", "builtin.sort.sort" or a bare "sort(" which may be ambiguous.
    """
    if not isinstance(source, str):
        return None

    import re

    # For Mojo (C-like), strip comments and string literals before scanning so commented-out uses are allowed
    def _strip_comments_and_strings(s: str) -> str:
        s = re.sub(r'"(?:\\.|[^"\\])*"', "", s, flags=re.DOTALL)
        s = re.sub(r"'(?:\\.|[^'\\])*'", "", s, flags=re.DOTALL)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
        return s

    cleaned = _strip_comments_and_strings(source)

    # explicit import
    if re.search(r"from\s+builtin\.sort\s+import\s+sort", cleaned):
        return "Forbidden Mojo import 'from builtin.sort import sort' detected."

    if re.search(r"\bbuiltin\.sort\.sort\b", cleaned):
        return "Forbidden access to 'builtin.sort.sort' detected."

    # bare sort( is ambiguous; only reject if we also see 'Span' nearby in the file (heuristic)
    if re.search(r"\bsort\s*\(", cleaned) and re.search(r"\bSpan\b", cleaned):
        return "Forbidden use of bare 'sort(' on Span types detected."

    return None


def reject_forbidden_patterns(language: str, source: str):
    """Run the appropriate scanner for the language and raise an error if forbidden patterns found."""
    if language == "cuda" or language == "c++":
        msg = _scan_cuda_forbidden(source)
        if msg:
            raise NVCCError(msg)
    elif language == "python" or language == "triton":
        msg = _scan_triton_python_forbidden(source)
        if msg:
            # reuse NVCCError for a consistent error type the runner knows how to handle
            raise NVCCError(msg)
    elif language == "mojo":
        msg = _scan_mojo_forbidden(source)
        if msg:
            raise MojoError(msg)


def mojo_command(srcs: list[Path | str], out: Path | str):
    """Get mojo command for the given source files and output file"""
    srcs = [str(src) for src in srcs]
    out = str(out)

    cmd = ["mojo", "build", "--optimization-level=3"]

    if str(out).endswith(".so"):
        cmd.append("--emit=shared-lib")

    cmd.extend(["-o", out] + srcs)

    return cmd


def mojo_command_executable(srcs: list[Path | str], out: Path | str):
    """Get mojo command for building an executable.

    This isolates flags for building an executable binary so they can be
    adjusted separately from the shared-library build path.
    """
    srcs = [str(src) for src in srcs]
    out = str(out)

    # Use the build subcommand but avoid the shared-lib emit flag.
    # If later needed, add executable-specific flags here (for example
    # --emit=exe or runtime flags) without changing `mojo_command`.
    cmd = ["mojo", "build", "--optimization-level=3", "-o", out] + srcs

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
    # Run forbidden-pattern rejection for CUDA/C++ sources
    reject_forbidden_patterns("cuda", solution_code)

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
    # Reject forbidden patterns for Mojo sources
    reject_forbidden_patterns("mojo", solution_code)

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
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise MojoError(
                "'mojo' CLI not found in PATH; please install Mojo or run compilation in an environment with the Mojo toolchain."
            )

        # Check for compilation errors
        if process.returncode != 0:
            raise MojoError(process.stderr)

    bytes_of_file = out_path.read_bytes()
    out_path.unlink()
    return bytes_of_file


@hash_dict
@lru_cache(maxsize=512)
def run_mojo_and_return_executable(solution_code: str, output_name: str) -> bytes:
    """Compile Mojo sources into an executable binary and return its bytes."""
    reject_forbidden_patterns("mojo", solution_code)

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_name}.bin")
    output_file.close()
    out_path = Path(output_file.name)
    out_path.unlink()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        (path / "solution.mojo").write_text(solution_code)
        src_path = path / "solution.mojo"
        cmd = mojo_command_executable([src_path], out_path)
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise MojoError(
                "'mojo' CLI not found in PATH; please install Mojo or run compilation in an environment with the Mojo toolchain."
            )

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


def cast_to_cute(data):
    """Cast data to CuTe tensors"""
    from cutlass.cute.runtime import from_dlpack

    data_casted = []
    for tensor in data:
        if isinstance(tensor, torch.Tensor):
            data_casted.append(from_dlpack(tensor, assumed_align=16))
        else:
            data_casted.append(tensor)
    return data_casted


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


BENCHMARK_SEED = 42


def reset_random_state(seed: int = BENCHMARK_SEED):
    """
    Reset all random state to ensure reproducible runs from the same "snapshot".
    Call this at the start of any runner function before test case generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)

    # Disable cuDNN benchmark mode (autotuner) for deterministic behavior
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def lock_gpu_clocks(device_id=0):
    """
    Lock GPU clocks to maximum values for consistent benchmarking.
    
    Args:
        device_id: GPU device ID to lock clocks for
        
    Returns:
        tuple: (graphics_clock, memory_clock) in MHz, or (None, None) on failure
    """
    try:
        # Get maximum clock speeds
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=clocks.max.graphics,clocks.max.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode != 0:
            print(f"Warning: Failed to query max clocks: {result.stderr}")
            return None, None
        
        parts = result.stdout.strip().split(", ")
        if len(parts) != 2:
            print(f"Warning: Unexpected clock query output: {result.stdout}")
            return None, None
        
        max_graphics_clock = int(parts[0])
        max_memory_clock = int(parts[1])
        
        # Lock graphics clock
        graphics_result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", f"-lgc", str(max_graphics_clock)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if graphics_result.returncode != 0:
            print(f"Warning: Failed to lock graphics clock: {graphics_result.stderr}")
            # Try to unlock if partial lock happened
            subprocess.run(["nvidia-smi", f"--id={device_id}", "-rgc"], capture_output=True)
            return None, None
        
        # Lock memory clock
        memory_result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", f"-lmc", str(max_memory_clock)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if memory_result.returncode != 0:
            print(f"Warning: Failed to lock memory clock: {memory_result.stderr}")
            # Unlock graphics clock if memory lock failed
            subprocess.run(["nvidia-smi", f"--id={device_id}", "-rgc"], capture_output=True)
            return None, None
        
        print(f"Locked GPU clocks: graphics={max_graphics_clock}MHz, memory={max_memory_clock}MHz")
        return max_graphics_clock, max_memory_clock
        
    except Exception as e:
        print(f"Warning: Exception while locking GPU clocks: {e}")
        return None, None


def unlock_gpu_clocks(device_id=0):
    """
    Unlock GPU clocks and restore to default (automatic) behavior.
    
    Args:
        device_id: GPU device ID to unlock clocks for
    """
    try:
        # Unlock graphics clock
        subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "-rgc"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        # Unlock memory clock
        subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "-rmc"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        print(f"Unlocked GPU clocks for device {device_id}")
    except Exception as e:
        print(f"Warning: Exception while unlocking GPU clocks: {e}")


class LockedGPUClocks:
    """Context manager to lock GPU clocks and automatically unlock on exit."""
    
    def __init__(self, device_id=0, lock=True):
        self.device_id = device_id
        self.lock = lock
        self.locked = False
        self.graphics_clock = None
        self.memory_clock = None
    
    def __enter__(self):
        if self.lock:
            self.graphics_clock, self.memory_clock = lock_gpu_clocks(self.device_id)
            if self.graphics_clock is not None and self.memory_clock is not None:
                self.locked = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.locked:
            unlock_gpu_clocks(self.device_id)
        return False  # Don't suppress exceptions


def prepare_gpu(lock_clocks=True):
    """
    Prepare the GPU for consistent benchmarking with a simple warm-up.
    
    Args:
        lock_clocks: If True, lock GPU clocks to maximum values
    """
    # Lock GPU clocks to maximum for consistent benchmarking
    if lock_clocks:
        lock_gpu_clocks(device_id=0)
    
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


# Pre-allocate L2 flush buffer using device's actual L2 cache size
_L2_FLUSH_BUFFER = None


# def flush_l2_cache():
#     """
#     Flush the GPU L2 cache by reading/writing a buffer sized to the device's L2 cache.
#     This ensures consistent cache state before each benchmark iteration.
#     """
#     global _L2_FLUSH_BUFFER

#     if _L2_FLUSH_BUFFER is None or _L2_FLUSH_BUFFER.device.type != "cuda":
#         l2_size = torch.cuda.get_device_properties(0).l2_cache_size + 16
#         _L2_FLUSH_BUFFER = torch.empty(l2_size, dtype=torch.uint8, device="cuda")

#     _L2_FLUSH_BUFFER.zero_()
#     torch.cuda.synchronize()


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
    num_warmup_iterations=5,
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
        num_warmup_iterations: Number of warmup iterations to run (default 5)

    Returns:
        benchmark_result: Dictionary with benchmark results
    """
    import json

    # Collect container diagnostics at start
    container_diag = get_container_diagnostics()

    # Prepare pointers for CUDA
    if param_func is None:
        parameters = make_parameters(
            language, solution_func, input_tensors, actual_output, problem, test_case
        )
    else:
        parameters = param_func(
            language, solution_func, input_tensors, actual_output, problem, test_case
        )

    if language == "cute":
        import cutlass.cute as cute

        solution_func = cute.compile(solution_func, *parameters)

    # Calculate FLOPS for this test case
    has_flops = problem.supports_flops()
    flops = problem.get_flops(test_case) if has_flops else None

    # === DIAGNOSTICS: GPU state before warmup ===
    gpu_before = get_gpu_diagnostics()

    # Prepare GPU (generic warmup)
    prepare_gpu()
    torch.cuda.synchronize()

    # === WARMUP: Run kernel warmup iterations with timing ===
    warmup_runtimes = []
    for _ in range(num_warmup_iterations):
        # Flush L2 cache before each warmup iteration for consistent cache state
        # flush_l2_cache()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        solution_func(*parameters)
        end_event.record()
        torch.cuda.synchronize()

        warmup_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
        warmup_runtimes.append(warmup_time)

    # === DIAGNOSTICS: GPU state after warmup ===
    gpu_after_warmup = get_gpu_diagnostics()

    # Use last warmup runtime to determine kernel characteristics
    initial_runtime = warmup_runtimes[-1] if warmup_runtimes else 0.0

    # Determine if this is a long-running kernel and how many iterations to run
    is_long_kernel = initial_runtime >= long_kernel_threshold

    if is_long_kernel:
        # For long kernels, use fixed number of iterations
        target_iterations = (min_iterations + max_iterations) // 2
    else:
        # For short kernels, use CV-based convergence with max_iterations cap
        target_iterations = max_iterations

    # === MEASURED ITERATIONS ===
    runtimes = []
    gflops_measurements = []
    iteration_data_list = []  # List of iteration data dictionaries with complete monitor samples
    
    # Initialize GPU monitor for real-time metrics during kernel execution
    gpu_monitor = None
    try:
        gpu_monitor = GPUMonitor(device_id=0, sample_interval_ms=1)
    except Exception as e:
        # Fallback if pynvml is not available or fails
        print(f"Warning: GPU monitor initialization failed: {e}, using fallback method")
        gpu_monitor = None

    for iteration in range(target_iterations):
        # Flush L2 cache before each iteration for consistent cache state
        # flush_l2_cache()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Start monitoring before kernel execution
        monitor_samples = []
        kernel_start_time = None
        kernel_end_time = None
        if gpu_monitor:
            if gpu_monitor.pynvml is not None and gpu_monitor.handle is not None:
                gpu_monitor.start()
                # Small delay to ensure monitor thread has started
                time.sleep(0.001)  # 1ms
                kernel_start_time = time.time()

        start_event.record()
        solution_func(*parameters)
        end_event.record()
        torch.cuda.synchronize()
        
        # Stop monitoring and collect samples
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
        runtime_ms = round(elapsed_time * 1000, 2)
        
        # Collect and filter monitor samples for this iteration
        kernel_samples = []
        if gpu_monitor and kernel_start_time is not None:
            kernel_end_time = time.time()
            monitor_samples = gpu_monitor.stop()
            
            if monitor_samples:
                # Filter samples to kernel execution period
                kernel_samples = [
                    s for s in monitor_samples 
                    if kernel_start_time <= s["timestamp"] <= kernel_end_time
                ]
        
        # Create iteration data structure with all monitoring data
        if kernel_samples and kernel_start_time is not None:
            # Calculate timestamps relative to kernel start (in milliseconds)
            timestamps = [(s["timestamp"] - kernel_start_time) * 1000.0 for s in kernel_samples]
            iteration_data = {
                "runtime_ms": runtime_ms,
                "temperatures": [s["temp_c"] for s in kernel_samples],
                "sm_clocks": [s["sm_clock_mhz"] for s in kernel_samples],
                "mem_clocks": [s["mem_clock_mhz"] for s in kernel_samples],
                "power_draws": [s["power_w"] for s in kernel_samples],
                "power_states": [s["pstate"] for s in kernel_samples],
                "timestamps": timestamps,  # Timestamps relative to kernel start (ms)
            }
        else:
            # No samples available
            iteration_data = {
                "runtime_ms": runtime_ms,
                "temperatures": [],
                "sm_clocks": [],
                "mem_clocks": [],
                "power_draws": [],
                "power_states": [],
                "timestamps": [],
            }
        
        iteration_data_list.append(iteration_data)
        
        runtimes.append(elapsed_time)

        # Calculate GFLOPS
        if has_flops and flops is not None and elapsed_time > 0:
            gflops = (flops / elapsed_time) / 1e9
            gflops_measurements.append(gflops)

        # Check if we've done enough iterations and the variance is low enough
        if not is_long_kernel and len(runtimes) >= min_iterations:
            if has_flops and gflops_measurements:
                mean_val = statistics.mean(gflops_measurements)
                if len(gflops_measurements) > 1:
                    stdev_val = statistics.stdev(gflops_measurements)
                    cv = stdev_val / mean_val if mean_val > 0 else float("inf")
                    if cv < target_cv:
                        break
            elif not has_flops and len(runtimes) > 1:
                mean_val = statistics.mean(runtimes)
                stdev_val = statistics.stdev(runtimes)
                cv = stdev_val / mean_val if mean_val > 0 else float("inf")
                if cv < target_cv:
                    break

    # === DIAGNOSTICS: GPU state after benchmark ===
    gpu_after_benchmark = get_gpu_diagnostics()

    # === CALCULATE TIMING STATS ===
    warmup_runtimes_ms = [r * 1000 for r in warmup_runtimes]
    measured_runtimes_ms = [r * 1000 for r in runtimes]

    warmup_mean_ms = statistics.mean(warmup_runtimes_ms) if warmup_runtimes_ms else 0.0
    measured_mean_ms = statistics.mean(measured_runtimes_ms) if measured_runtimes_ms else 0.0

    # CV of measured runtimes
    if len(measured_runtimes_ms) > 1:
        measured_stdev = statistics.stdev(measured_runtimes_ms)
        cv = measured_stdev / measured_mean_ms if measured_mean_ms > 0 else 0.0
    else:
        cv = 0.0

    # Min/max ratio
    if measured_runtimes_ms:
        min_max_ratio = max(measured_runtimes_ms) / min(measured_runtimes_ms) if min(measured_runtimes_ms) > 0 else 1.0
    else:
        min_max_ratio = 1.0

    # First vs rest ratio
    if len(measured_runtimes_ms) > 1:
        first_vs_rest_ratio = measured_runtimes_ms[0] / statistics.mean(measured_runtimes_ms[1:]) if statistics.mean(measured_runtimes_ms[1:]) > 0 else 1.0
    else:
        first_vs_rest_ratio = 1.0

    # Minimal logging output
    print(f"[{test_id}] {test_case['name']}: {measured_mean_ms:.2f}ms (cv={cv:.3f})")

    # === BUILD RESULT ===
    if len(runtimes) > 1:
        mean_runtime = statistics.mean(runtimes)
    else:
        mean_runtime = runtimes[0] if runtimes else 0.0

    benchmark_result = {
        "name": test_case["name"],
        "test_id": test_id,
        "runtime_ms": mean_runtime * 1000,
        "iterations": iteration_data_list,  # Complete per-iteration monitoring data
    }

    if gflops_measurements:
        mean_gflops = statistics.mean(gflops_measurements)
        benchmark_result["gflops"] = mean_gflops

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

    elif language == "python" or language == "cute":
        # Run Python/Triton AST checks to reject forbidden patterns
        if solution_code:
            reject_forbidden_patterns("triton", solution_code)

        filename = "triton_solution.py" if language == "python" else "cute_solution.py"
        if not solution_code:
            raise ValueError("Source code required for Triton submissions")

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)

        # This is needed because @jit has to read the source code
        with open(temp_path, "w") as f:
            f.write(solution_code)

        spec = importlib.util.spec_from_file_location(filename, temp_path)
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
    elif language == "cute":
        from cutlass.cute.runtime import from_dlpack

        input_ptrs = cast_to_cute(input_tensors)
        output_ptr = from_dlpack(actual_output, assumed_align=16)
        extra_params = problem.get_extra_params(test_case)
        extra_params_casted = cast_to_cute(extra_params)
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


def generate_ptx_sass(gpu: str, solution_code: str) -> tuple[str, str]:
    """Generate PTX and SASS outputs from CUDA source code

    Args:
        gpu (str): GPU type to use
        solution_code (str): Code of the solution

    Returns:
        tuple[str, str]: (PTX content, SASS content)

    Raises:
        Exception: If PTX/SASS generation fails
    """
    sm = GPU_COMPUTE_CAPABILITIES[gpu]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td)

        # Write source file
        src_path = path / "solution.cu"
        src_path.write_text(solution_code)

        ptx_path = path / "output.ptx"
        cubin_path = path / "output.cubin"

        # Generate PTX
        ptx_cmd = [
            "nvcc",
            "-ptx",
            str(src_path),
            "-o",
            str(ptx_path),
            "-lineinfo",
            "-std=c++20",
            f"-arch=compute_{sm}",
        ]
        ptx_process = subprocess.run(ptx_cmd, capture_output=True, text=True)

        if ptx_process.returncode != 0:
            raise Exception(f"PTX generation failed: {ptx_process.stderr}")

        ptx_content = ptx_path.read_text()

        # Generate CUBIN
        cubin_cmd = [
            "nvcc",
            "-cubin",
            str(src_path),
            "-o",
            str(cubin_path),
            "-std=c++20",
            f"-arch=compute_{sm}",
            f"-code=sm_{sm}",
        ]
        cubin_process = subprocess.run(cubin_cmd, capture_output=True, text=True)

        if cubin_process.returncode != 0:
            raise Exception(f"CUBIN generation failed: {cubin_process.stderr}")

        # Disassemble to SASS using nvdisasm
        sass_process = subprocess.run(
            ["nvdisasm", str(cubin_path), "--print-line-info"], capture_output=True, text=True
        )

        if sass_process.returncode != 0:
            # Try without line info if it fails
            sass_process = subprocess.run(
                ["nvdisasm", str(cubin_path)], capture_output=True, text=True
            )
            if sass_process.returncode != 0:
                raise Exception(f"SASS generation failed: {sass_process.stderr}")

        sass_content = sass_process.stdout

    return ptx_content, sass_content


def yield_ptx_sass(gpu: str, solution_code: str):
    """Generate and yield PTX/SASS assembly, with graceful error handling"""
    try:
        ptx_content, sass_content = generate_ptx_sass(gpu, solution_code)
        yield {"status": "PTX", "content": ptx_content}
        yield {"status": "SASS", "content": sass_content}
    except Exception as e:
        print(f"PTX/SASS generation failed: {e}")
        yield {"status": "WARNING", "message": f"PTX/SASS generation failed: {str(e)}"}


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
    # Reject forbidden patterns for CUDA executable sources
    reject_forbidden_patterns("cuda", solution_code)

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


def to_lossless_jsonable(x):
    """
    Recursively convert structures so that integers outside JS safe range
    are emitted as strings. Safe ints/floats/bools remain as-is.
    Tensors/ndarrays are converted to Python lists first.
    """
    # PyTorch tensors
    if isinstance(x, torch.Tensor):
        return to_lossless_jsonable(x.detach().cpu().tolist())

    # NumPy arrays
    if isinstance(x, np.ndarray):
        return to_lossless_jsonable(x.tolist())

    # NumPy integer scalars
    if isinstance(x, (np.integer,)):
        v = int(x)
        return str(v) if abs(v) > JS_MAX_SAFE else v

    # Plain Python integers
    if isinstance(x, Integral) and not isinstance(x, bool):
        return str(x) if abs(x) > JS_MAX_SAFE else int(x)

    # Floats - round to 6 decimal places
    if isinstance(x, float):
        return round(x, 6)

    # Bools/None/str are fine as-is
    if isinstance(x, (bool, type(None), str)):
        return x

    # Lists/tuples
    if isinstance(x, (list, tuple)):
        return [to_lossless_jsonable(v) for v in x]

    # Dicts
    if isinstance(x, dict):
        return {k: to_lossless_jsonable(v) for (k, v) in x.items()}

    # Fallback (e.g., custom objects) — let json handle or str()
    return x


class ReferenceSolutionContext:
    def __init__(self, dtype):
        self.dtype = dtype
        self.autocast_ctx = None

    def __enter__(self):
        self.old_fp32 = torch.backends.fp32_precision
        self.old_matmul_fp32 = torch.backends.cuda.matmul.fp32_precision
        self.old_cudnn_fp32 = torch.backends.cudnn.fp32_precision
        self.old_conv_fp32 = torch.backends.cudnn.conv.fp32_precision
        self.old_rnn_fp32 = torch.backends.cudnn.rnn.fp32_precision
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.old_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.autocast_ctx = torch.autocast("cuda", enabled=False, dtype=self.dtype)
        return self.autocast_ctx.__enter__()

    def __exit__(self, *args):
        self.autocast_ctx.__exit__(*args)
        torch.backends.fp32_precision = self.old_fp32
        torch.backends.cuda.matmul.fp32_precision = self.old_matmul_fp32
        torch.backends.cudnn.fp32_precision = self.old_cudnn_fp32
        torch.backends.cudnn.conv.fp32_precision = self.old_conv_fp32
        torch.backends.cudnn.rnn.fp32_precision = self.old_rnn_fp32
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.use_deterministic_algorithms(False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.old_cublas_config
