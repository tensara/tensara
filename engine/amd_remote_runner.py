#!/usr/bin/env python3
"""
AMD Remote Runner - Runs on AMD GPU VMs via dstack

This is a self-contained script that executes on AMD GPUs (MI300X, MI210, etc.)
provisioned by dstack. It:
1. Receives problem/code info via JSON payload
2. Compiles HIP kernels using hipcc
3. Runs checker (correctness tests) and/or benchmark phases
4. Outputs JSON events that match NVIDIA format for frontend compatibility

Key design decisions:
- Uses PyTorch ROCm which provides drop-in CUDA API compatibility
- Mirrors the NVIDIA runner.py logic for checker/benchmark
- Problems are cloned from the public tensara/problems repo
- Compilation uses filesystem caching for VM reuse optimization

Usage (on AMD VM):
    python amd_remote_runner.py '{"problem": "leaky-relu", "code": "...", ...}'
    
Environment requirements:
    - ROCm with PyTorch installed (rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0)
    - hipcc compiler available
    - Problems repo cloned to /workspace/problems
"""

import sys
import os
import json
import time
import hashlib
import tempfile
import subprocess
import ctypes
import gc
import statistics
import traceback
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional, Tuple
from dataclasses import dataclass

# ROCm PyTorch - uses same torch.cuda API
import torch


# ============================================================================
# Configuration
# ============================================================================

PROBLEMS_REPO_URL = "https://github.com/tensara/problems.git"
PROBLEMS_DIR = Path("/workspace/problems")
HIPCC_CACHE_DIR = Path("/tmp/hipcc_cache")

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ============================================================================
# JSON Event Output (matches NVIDIA format)
# ============================================================================

def emit_event(event: Dict[str, Any]) -> None:
    """Emit a JSON event to stdout for parsing by amd_task_runner.py"""
    # Prefix with JSON_EVENT: so we can filter from other output
    print(f"JSON_EVENT:{json.dumps(event)}", flush=True)


def emit_status(status: str, message: str = "", **kwargs) -> None:
    """Emit a status event"""
    event = {"status": status, "message": message, **kwargs}
    emit_event(event)


# ============================================================================
# HIP Compilation
# ============================================================================

class HIPCCError(Exception):
    """HIP compilation error"""
    pass


def get_code_hash(code: str) -> str:
    """Get a hash of the code for caching"""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def compile_hip_kernel(solution_code: str, cache_enabled: bool = True) -> bytes:
    """
    Compile HIP kernel to shared library
    
    Args:
        solution_code: HIP kernel source code
        cache_enabled: Whether to use filesystem cache
        
    Returns:
        Bytes of compiled .so file
        
    Raises:
        HIPCCError: If compilation fails
    """
    code_hash = get_code_hash(solution_code)
    
    # Check cache first
    if cache_enabled:
        HIPCC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = HIPCC_CACHE_DIR / f"{code_hash}.so"
        if cache_file.exists():
            print(f"[HIP] Cache hit: {code_hash}", file=sys.stderr)
            return cache_file.read_bytes()
    
    print(f"[HIP] Compiling kernel (hash: {code_hash})...", file=sys.stderr)
    
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src_file = td_path / "solution.hip"
        out_file = td_path / "solution.so"
        
        # Write source
        src_file.write_text(solution_code)
        
        # Compile with hipcc
        # Using flags that work well on MI300X
        cmd = [
            "hipcc",
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-o", str(out_file),
            str(src_file),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise HIPCCError("Compilation timed out (120s limit)")
        except FileNotFoundError:
            raise HIPCCError("hipcc not found - ensure ROCm is installed")
        
        if result.returncode != 0:
            raise HIPCCError(result.stderr or result.stdout or "Unknown compilation error")
        
        # Read compiled library
        compiled_bytes = out_file.read_bytes()
        
        # Cache the result
        if cache_enabled:
            try:
                cache_file = HIPCC_CACHE_DIR / f"{code_hash}.so"
                cache_file.write_bytes(compiled_bytes)
                print(f"[HIP] Cached: {code_hash}", file=sys.stderr)
            except Exception as e:
                print(f"[HIP] Cache write failed: {e}", file=sys.stderr)
        
        return compiled_bytes


def load_hip_library(compiled_bytes: bytes) -> ctypes.CDLL:
    """Load compiled HIP library from bytes
    
    Note: The temp file is intentionally NOT deleted because ROCm/HIP
    may need to access the .so file when loading GPU code objects
    during kernel execution. Temp files are cleaned up on process exit.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".so") as f:
        f.write(compiled_bytes)
        temp_path = f.name
    
    lib = ctypes.CDLL(temp_path)
    return lib


# ============================================================================
# Problem Loading
# ============================================================================

def ensure_problems_repo() -> Path:
    """
    Ensure the problems repo is cloned and up-to-date
    
    Returns:
        Path to the problems directory
    """
    if not PROBLEMS_DIR.exists():
        print(f"[Setup] Cloning problems repo to {PROBLEMS_DIR}...", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth", "1", PROBLEMS_REPO_URL, str(PROBLEMS_DIR)],
            check=True,
            capture_output=True,
        )
    return PROBLEMS_DIR


def load_problem_module(problem_slug: str, problem_def: Optional[str] = None):
    """
    Load a problem module
    
    Args:
        problem_slug: Problem identifier (e.g., "leaky-relu")
        problem_def: Optional inline problem definition (if provided, uses this instead of file)
        
    Returns:
        Instantiated Problem subclass
    """
    import importlib.util
    from types import ModuleType
    
    # Convert slug to module name (leaky-relu -> leaky_relu)
    module_name = problem_slug.replace("-", "_")
    
    # CRITICAL: Load the base Problem class from /workspace/problem.py
    # This file is uploaded alongside amd_remote_runner.py by dstack
    # Problem definitions in tensara/problems do `from problem import Problem`
    # so we need to register this module before loading problem files
    workspace_problem_file = Path("/workspace/problem.py")
    if workspace_problem_file.exists():
        print(f"[Problem] Loading base Problem class from {workspace_problem_file}", file=sys.stderr)
        base_spec = importlib.util.spec_from_file_location("problem", workspace_problem_file)
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules["problem"] = base_module
        base_spec.loader.exec_module(base_module)
    else:
        raise FileNotFoundError(
            f"Base Problem class not found at {workspace_problem_file}. "
            "Ensure problem.py is uploaded alongside amd_remote_runner.py."
        )
    
    if problem_def:
        # Load from inline definition
        spec = importlib.util.spec_from_loader(module_name, loader=None, origin="<string>")
        module = ModuleType(spec.name)
        exec(problem_def, module.__dict__)
        problem_class = getattr(module, module_name)
        return problem_class()
    else:
        # Load from problems repo
        problems_dir = ensure_problems_repo()
        problem_file = problems_dir / "problems" / problem_slug / "def.py"
        
        if not problem_file.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_file}")
        
        print(f"[Problem] Loading problem definition from {problem_file}", file=sys.stderr)
        
        spec = importlib.util.spec_from_file_location(module_name, problem_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        problem_class = getattr(module, module_name)
        return problem_class()


# ============================================================================
# Solution Function Setup (mirrors utils.py)
# ============================================================================

def cast_to_ctype(data: List, argtypes: List, language: str = "hip"):
    """Cast data to ctypes for calling HIP kernel"""
    data_casted = []
    for tensor, argtype in zip(data, argtypes):
        if isinstance(tensor, torch.Tensor):
            data_casted.append(ctypes.cast(tensor.data_ptr(), argtype))
        else:
            data_casted.append(argtype(tensor))
    return data_casted


def make_solution_func(compiled_bytes: bytes, problem):
    """
    Create a callable solution function from compiled HIP library
    
    Args:
        compiled_bytes: Compiled .so file bytes
        problem: Problem instance
        
    Returns:
        Callable function
    """
    hip_lib = load_hip_library(compiled_bytes)
    func_sig = problem.get_function_signature()
    hip_lib.solution.argtypes = func_sig["argtypes"]
    hip_lib.solution.restype = func_sig["restype"]
    return hip_lib.solution


def make_parameters(solution_func, input_tensors, actual_output, problem, test_case):
    """Prepare parameters for calling the HIP kernel"""
    input_ptrs = cast_to_ctype(
        input_tensors, solution_func.argtypes[:len(input_tensors)]
    )
    output_ptr = ctypes.cast(actual_output.data_ptr(), solution_func.argtypes[len(input_ptrs)])
    extra_params = problem.get_extra_params(test_case)
    extra_params_casted = cast_to_ctype(
        extra_params, solution_func.argtypes[-len(extra_params):] if extra_params else []
    )
    return input_ptrs + [output_ptr] + extra_params_casted


# ============================================================================
# GPU Utilities
# ============================================================================

def prepare_gpu():
    """Prepare GPU for consistent benchmarking"""
    torch.cuda.empty_cache()
    
    # Warm-up run
    warmup_tensor = torch.rand(5000, 5000, device="cuda")
    for _ in range(10):
        torch.matmul(warmup_tensor, warmup_tensor.t())
    torch.cuda.synchronize()
    del warmup_tensor
    torch.cuda.empty_cache()
    
    time.sleep(0.5)


# ============================================================================
# Checker Phase (Correctness Testing)
# ============================================================================

def run_checker(
    problem_name: str,
    problem,
    solution_func,
    dtype: torch.dtype,
) -> Iterator[Dict[str, Any]]:
    """
    Check submitted solution against reference implementation
    
    Yields JSON events as tests complete:
        CHECKING -> TEST_RESULT (for each test) -> CHECKED or WRONG_ANSWER
    """
    try:
        test_cases = problem.generate_test_cases(dtype)
        total_tests = len(test_cases)
        test_results = []
        passed_tests = 0
        
        yield {"status": "CHECKING", "message": "Running test cases...", "total_tests": total_tests}
        
        start_time = time.time()
        time_limit = getattr(problem, 'time_limit', 100)
        
        for test_id, test_case in enumerate(test_cases, 1):
            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s",
                }
                return
            
            test_name = test_case["name"]
            input_tensors = test_case["create_inputs"]()
            
            # Get reference solution
            expected_output = problem.reference_solution(*input_tensors).cpu()
            
            # Create output tensor
            actual_output = torch.zeros_like(expected_output, device="cuda").contiguous()
            
            # Run user's solution
            parameters = make_parameters(solution_func, input_tensors, actual_output, problem, test_case)
            solution_func(*parameters)
            torch.cuda.synchronize()
            
            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s",
                }
                return
            
            # Verify result
            is_correct, debug_info = problem.verify_result(
                expected_output, actual_output.cpu(), dtype
            )
            
            # Cleanup
            del input_tensors, expected_output, actual_output, parameters
            gc.collect()
            torch.cuda.empty_cache()
            
            test_result = {
                "test_id": test_id,
                "name": test_name,
            }
            
            if is_correct:
                test_result["status"] = "PASSED"
                passed_tests += 1
            else:
                test_result["status"] = "FAILED"
                test_results.append(test_result)
                yield {
                    "status": "WRONG_ANSWER",
                    "debug_info": debug_info,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests,
                    "test_results": test_results,
                }
                return
            
            test_results.append(test_result)
            
            yield {
                "status": "TEST_RESULT",
                "result": test_result,
                "total_tests": total_tests,
            }
        
        # All tests passed
        yield {
            "status": "CHECKED",
            "test_results": test_results,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
        }
        
    except HIPCCError as e:
        yield {
            "status": "COMPILE_ERROR",
            "message": "HIP Compilation Failed",
            "details": str(e),
        }
    
    except RuntimeError as e:
        yield {
            "status": "RUNTIME_ERROR",
            "message": str(e),
            "details": traceback.format_exc(),
        }
    
    except Exception as e:
        yield {
            "status": "ERROR",
            "message": str(e.__class__.__name__),
            "details": traceback.format_exc(),
        }


# ============================================================================
# Benchmark Phase
# ============================================================================

def run_dynamic_benchmark(
    solution_func,
    problem,
    test_id: int,
    test_case: Dict,
    input_tensors: List,
    actual_output: torch.Tensor,
    min_iterations: int = 5,
    max_iterations: int = 20,
    target_cv: float = 0.02,
    long_kernel_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Run benchmark with dynamic stopping based on variance
    
    Returns:
        Dictionary with runtime_ms, gflops (if applicable), etc.
    """
    parameters = make_parameters(solution_func, input_tensors, actual_output, problem, test_case)
    
    # Calculate FLOPS if supported
    has_flops = problem.supports_flops()
    flops = problem.get_flops(test_case) if has_flops else None
    
    # Warm up
    prepare_gpu()
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    solution_func(*parameters)
    end_event.record()
    torch.cuda.synchronize()
    
    initial_runtime = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    
    # Determine iteration count
    is_long_kernel = initial_runtime >= long_kernel_threshold
    target_iterations = (min_iterations + max_iterations) // 2 if is_long_kernel else max_iterations
    
    # Collect measurements
    runtimes = [initial_runtime]
    gflops_measurements = []
    
    if has_flops and flops and initial_runtime > 0:
        gflops_measurements.append((flops / initial_runtime) / 1e9)
    
    for iteration in range(1, target_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        solution_func(*parameters)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        runtimes.append(elapsed_time)
        
        if has_flops and flops and elapsed_time > 0:
            gflops = (flops / elapsed_time) / 1e9
            gflops_measurements.append(gflops)
        
        # Check for early stopping (short kernels only)
        if not is_long_kernel and iteration + 1 >= min_iterations:
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
    
    mean_runtime = statistics.mean(runtimes) if len(runtimes) > 1 else runtimes[0]
    
    benchmark_result = {
        "name": test_case["name"],
        "test_id": test_id,
        "runtime_ms": mean_runtime * 1000,
    }
    
    if gflops_measurements:
        benchmark_result["gflops"] = statistics.mean(gflops_measurements)
    
    return benchmark_result


def run_benchmark(
    problem_name: str,
    problem,
    solution_func,
    dtype: torch.dtype,
) -> Iterator[Dict[str, Any]]:
    """
    Run benchmark on solution
    
    Yields JSON events:
        BENCHMARKING -> BENCHMARK_RESULT (for each test) -> BENCHMARKED
    """
    try:
        yield {"status": "BENCHMARKING", "message": "Running benchmarks..."}
        
        test_cases = problem.generate_test_cases(dtype)
        total_tests = len(test_cases)
        benchmark_results = []
        
        prepare_gpu()
        
        for test_id, test_case in enumerate(test_cases, 1):
            try:
                input_tensors = test_case["create_inputs"]()
                expected_output = problem.reference_solution(*input_tensors).cpu()
                actual_output = torch.zeros_like(expected_output, device="cuda").contiguous()
                
                benchmark_result = run_dynamic_benchmark(
                    solution_func,
                    problem,
                    test_id,
                    test_case,
                    input_tensors,
                    actual_output,
                )
                
                benchmark_results.append(benchmark_result)
                
                yield {
                    "status": "BENCHMARK_RESULT",
                    "result": benchmark_result,
                    "total_tests": total_tests,
                }
                
                # Cleanup
                del input_tensors, expected_output, actual_output
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                yield {
                    "status": "RUNTIME_ERROR",
                    "message": str(e),
                    "details": traceback.format_exc(),
                }
                return
        
        # Calculate summary statistics
        test_count = len(benchmark_results)
        if test_count > 0:
            avg_runtime_ms = statistics.mean([r["runtime_ms"] for r in benchmark_results])
            
            has_gflops = any(("gflops" in r) and (r["gflops"] is not None) for r in benchmark_results)
            if has_gflops:
                avg_gflops = statistics.mean([
                    r["gflops"] for r in benchmark_results 
                    if ("gflops" in r) and (r["gflops"] is not None)
                ])
            else:
                avg_gflops = None
        else:
            avg_runtime_ms = 0
            avg_gflops = None
        
        summary = {
            "status": "BENCHMARKED",
            "test_results": benchmark_results,
            "avg_runtime_ms": avg_runtime_ms,
            "total_tests": test_count,
        }
        if avg_gflops is not None:
            summary["avg_gflops"] = avg_gflops
        
        yield summary
        
    except HIPCCError as e:
        yield {
            "status": "COMPILE_ERROR",
            "message": "HIP Compilation Failed",
            "details": str(e),
        }
    
    except RuntimeError as e:
        yield {
            "status": "RUNTIME_ERROR",
            "message": str(e),
            "details": traceback.format_exc(),
        }
    
    except Exception as e:
        yield {
            "status": "ERROR",
            "message": str(e.__class__.__name__),
            "details": traceback.format_exc(),
        }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for AMD remote runner
    
    Expects JSON payload as command-line argument or stdin with:
        - problem: Problem slug (e.g., "leaky-relu")
        - problem_def: Optional inline problem definition
        - solution_code: HIP kernel source code
        - dtype: Data type (float32, float16, bfloat16)
        - endpoint: "checker", "benchmark", or "full" (checker then benchmark)
    """
    try:
        # Read payload
        if len(sys.argv) > 1:
            payload_str = sys.argv[1]
        else:
            payload_str = sys.stdin.read()
        
        payload = json.loads(payload_str)
        
        # Extract fields
        problem_slug = payload.get("problem", "unknown")
        problem_def = payload.get("problem_def")
        solution_code = payload.get("solution_code", "")
        dtype_str = payload.get("dtype", "float32")
        endpoint = payload.get("endpoint", "full")
        
        print(f"[AMD Runner] Starting execution", file=sys.stderr)
        print(f"[AMD Runner] Problem: {problem_slug}", file=sys.stderr)
        print(f"[AMD Runner] Endpoint: {endpoint}", file=sys.stderr)
        print(f"[AMD Runner] Dtype: {dtype_str}", file=sys.stderr)
        print(f"[AMD Runner] Code length: {len(solution_code)} chars", file=sys.stderr)
        
        # Verify GPU is available
        if not torch.cuda.is_available():
            emit_event({
                "status": "ERROR",
                "message": "No GPU available",
                "details": "torch.cuda.is_available() returned False. Ensure ROCm is properly installed.",
            })
            sys.exit(1)
        
        print(f"[AMD Runner] GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        
        # Convert dtype
        dtype = DTYPE_MAP.get(dtype_str, torch.float32)
        
        # Emit initial status
        emit_event({
            "status": "COMPILING",
            "message": "Compiling HIP kernel...",
        })
        
        # Compile solution
        try:
            compiled_bytes = compile_hip_kernel(solution_code)
        except HIPCCError as e:
            emit_event({
                "status": "COMPILE_ERROR",
                "message": "HIP Compilation Failed",
                "details": str(e),
            })
            sys.exit(1)
        
        emit_event({
            "status": "COMPILED",
            "message": "Compilation successful",
        })
        
        # Load problem
        try:
            problem = load_problem_module(problem_slug, problem_def)
        except Exception as e:
            emit_event({
                "status": "ERROR",
                "message": f"Failed to load problem: {problem_slug}",
                "details": str(e),
            })
            sys.exit(1)
        
        # Create solution function
        solution_func = make_solution_func(compiled_bytes, problem)
        
        # Run requested endpoint
        if endpoint == "checker":
            for event in run_checker(problem_slug, problem, solution_func, dtype):
                emit_event(event)
                
        elif endpoint == "benchmark":
            for event in run_benchmark(problem_slug, problem, solution_func, dtype):
                emit_event(event)
                
        elif endpoint == "full":
            # Run checker first
            checker_passed = False
            for event in run_checker(problem_slug, problem, solution_func, dtype):
                emit_event(event)
                if event.get("status") == "CHECKED":
                    checker_passed = True
                elif event.get("status") in ("WRONG_ANSWER", "COMPILE_ERROR", "RUNTIME_ERROR", "ERROR"):
                    checker_passed = False
                    break
            
            # Only run benchmark if checker passed
            if checker_passed:
                for event in run_benchmark(problem_slug, problem, solution_func, dtype):
                    emit_event(event)
        else:
            emit_event({
                "status": "ERROR",
                "message": f"Unknown endpoint: {endpoint}",
                "details": f"Valid endpoints are: checker, benchmark, full",
            })
            sys.exit(1)
        
        print(f"[AMD Runner] Execution completed", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        emit_event({
            "status": "ERROR",
            "message": "Invalid JSON payload",
            "details": str(e),
        })
        sys.exit(1)
        
    except Exception as e:
        emit_event({
            "status": "ERROR",
            "message": str(e.__class__.__name__),
            "details": traceback.format_exc(),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
