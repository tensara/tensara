import modal
from pathlib import Path
import ctypes
import torch
import json
import tempfile
import subprocess
from typing import Iterator, Dict
from fastapi.responses import StreamingResponse
from problem import Problem
import gc
from utils import load_problem_module, prepare_gpu, run_dynamic_benchmark
import statistics

GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90",
    "A100-80GB": "80",
    "A10G": "86",
}

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
CURR_DIR = Path(__file__).parent

image = modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.11").pip_install(
    "torch",
    "numpy",
    "fastapi[standard]",
)

app = modal.App("tensara-python-engine", image=image)

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

def run_nvcc(gpu: str, files: Dict[str, str], output_name: str) -> Path:
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
        for name, content in files.items():
            (path / name).write_text(content)
        
        # For a shared library, we need the solution.cu file
        src_path = path / "solution.cu"
        
        # Compile with nvcc
        cmd = nvcc_command(gpu, [src_path], out_path)
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for compilation errors
        if process.returncode != 0:
            raise NVCCError(process.stderr)
            
    return out_path

def run_checker(problem: Problem, solution_code: str, gpu: str = "T4") -> Iterator[str]:
    """
    Check a submitted CUDA solution against the reference implementation
    and stream results as they become available
    
    Args:
        problem: Problem instance
        solution_code: CUDA code for the submitted solution
        gpu: GPU type to use
        
    Returns:
        Iterator that yields JSON strings with test results
    """
    lib_path = None
    
    try:
        # Send compilation status
        yield json.dumps({"status": "compiling"})

        files = {
            "solution.cu": solution_code
        }

        # Compile the solution
        lib_path = run_nvcc(gpu, files, "solution")
        
        # Send running status
        yield json.dumps({"status": "running"})

        # Load the compiled library
        cuda_lib = ctypes.CDLL(str(lib_path))
        
        # Set function signature
        func_sig = problem.get_function_signature()
        cuda_lib.solution.argtypes = func_sig["argtypes"]
        cuda_lib.solution.restype = func_sig["restype"]
        
        # Get test cases
        test_cases = problem.generate_test_cases()
        total_tests = len(test_cases)
        test_results = []
        passed_tests = 0
        has_failed = False

        # Run each test case
        for test_id, test_case in enumerate(test_cases, 1):
            test_name = test_case["name"]
            try:
                input_tensors = test_case["create_inputs"]()
                
                # Get the reference solution and move it to CPU
                expected_output = problem.reference_solution(*input_tensors).cpu()

                # Create actual_output with the same shape as expected_output
                actual_output = torch.zeros_like(expected_output, device='cuda')  # Ensure it's on GPU

                # Prepare pointers for CUDA
                input_ptrs = [ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)) 
                             for tensor in input_tensors]
                output_ptr = ctypes.cast(actual_output.data_ptr(), ctypes.POINTER(ctypes.c_float))
                extra_params = problem.get_extra_params(test_case)

                # Call the CUDA solution
                cuda_lib.solution(*(input_ptrs + [output_ptr] + extra_params))
                torch.cuda.synchronize()

                # Move to CPU for comparison
                is_correct, debug_info = problem.verify_result(expected_output, actual_output.cpu())

                # Clean up memory
                del input_tensors, expected_output, actual_output, input_ptrs, output_ptr
                gc.collect()
                torch.cuda.empty_cache()

                if is_correct:
                    status = "PASSED"
                    passed_tests += 1
                else:
                    status = "FAILED"
                    has_failed = True
                    
                test_result = {
                    "test_id": test_id,
                    "name": test_name,
                    "status": status
                }
                
                if status == "FAILED":
                    test_result["debug_info"] = debug_info
                    
                test_results.append(test_result)
                
                obj = {
                    "status": "test_result",
                    "result": test_result,
                    "totalTests": total_tests,
                }
                yield json.dumps(obj)

                
                
            except Exception as e:
                status = "FAILED"
                has_failed = True
                debug_info = {"error": str(e)}
                
                test_result = {
                    "test_id": test_id,
                    "name": test_name,
                    "status": status,
                    "debug_info": debug_info
                }
                
                test_results.append(test_result)
                
                obj = {
                    "status": "test_result",
                    "result": test_result,
                    "totalTests": total_tests,
                }
                yield json.dumps(obj)
        
        # Final status message
        obj = {
            "status": "complete",
            "passed": not has_failed,
            "test_results": test_results,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
        }
        yield json.dumps(obj)
        
    except NVCCError as e:
        obj = {
            "status": "error",
            "error": "Compilation failed",
            "details": e.args[0],
            "test_results": [],
            "passed_tests": 0,
            "total_tests": 0,
        }
        yield json.dumps(obj)


def run_benchmark(problem, solution_code, gpu="T4"):
    """
    Run benchmark on compiled CUDA solution
    
    Args:
        problem: Problem module containing test cases and verification methods
        solution_code: CUDA C code to compile and benchmark
        gpu: GPU type to benchmark on
    
    Yields:
        Dictionary objects with benchmark status updates
    """
    try:
        # Compile the solutionyield json.dumps({"status": "compiling"})
        yield json.dumps({"status": "compiling"})

        files = {
            "solution.cu": solution_code
        }
        lib_path = run_nvcc(gpu, files, "solution")
        

        yield json.dumps({"status": "running"})

        # Load the compiled library
        cuda_lib = ctypes.CDLL(str(lib_path))
        
        # Set function signature
        func_sig = problem.get_function_signature()
        cuda_lib.solution.argtypes = func_sig["argtypes"]
        cuda_lib.solution.restype = func_sig["restype"]
            
        # Get test cases from the problem
        test_cases = problem.generate_test_cases()

        total_tests = len(test_cases)
        
        # Initialize statistics
        benchmark_results = []
        
        # Prepare GPU for benchmarking (one-time setup at the beginning)
        prepare_gpu()
        
        # Run each test case
        for test_id, test_case in enumerate(test_cases, 1):
            test_name = test_case["name"]
            
            try:
                # Create inputs and reference output
                input_tensors = test_case["create_inputs"]()
                expected_output = problem.reference_solution(*input_tensors).cpu()
                actual_output = torch.zeros_like(expected_output, device='cuda')
                
                # Prepare pointers for CUDA
                input_ptrs = [ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)) 
                             for tensor in input_tensors]
                output_ptr = ctypes.cast(actual_output.data_ptr(), ctypes.POINTER(ctypes.c_float))
                extra_params = problem.get_extra_params(test_case)
                
                # First run to verify correctness
                cuda_lib.solution(*(input_ptrs + [output_ptr] + extra_params))
                torch.cuda.synchronize()
                
                
                # Run the dynamic benchmark
                benchmark_result = run_dynamic_benchmark(
                    cuda_lib, 
                    problem, 
                    test_case, 
                    input_tensors, 
                    actual_output,
                    min_iterations=5,
                    max_iterations=15,
                    target_cv=0.02  # 2% target coefficient of variation
                )
                
                benchmark_results.append(benchmark_result)
                
                yield {
                    "status": "benchmark_result",
                    "result": benchmark_result,
                    "totalTests": total_tests,
                }
                
                # Clean up memory
                del input_tensors, expected_output, actual_output, input_ptrs, output_ptr
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                benchmark_result = {
                    "test_id": test_id,
                    "name": test_name,
                    "status": "FAILED",
                    "debug_info": {"error": str(e)}
                }
                
                benchmark_results.append(benchmark_result)
                
                yield {
                    "status": "benchmark_result",
                    "result": benchmark_result,
                    "totalTests": total_tests,
                }
        
        # Calculate overall statistics
        passed_results = [r for r in benchmark_results if r["status"] == "PASSED"]
        
        if passed_results:
            avg_gflops = statistics.mean([r["performance_stats"]["mean_gflops"] for r in passed_results])
            max_gflops = max([r["performance_stats"]["max_gflops"] for r in passed_results])
            avg_throughput = statistics.mean([r["memory_stats"]["mean_throughput_gbps"] for r in passed_results])
        else:
            avg_gflops = 0
            max_gflops = 0
            avg_throughput = 0
        
        # Final status message
        yield {
            "status": "complete",
            "benchmark_results": benchmark_results,
            "summary": {
                "avg_gflops": avg_gflops,
                "max_gflops": max_gflops,
                "avg_memory_throughput_gbps": avg_throughput,
                "total_tests": total_tests,
                "passed_tests": len(passed_results)
            }
        }
    except Exception as e:
        # Handle any unexpected errors
        import traceback
        yield {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def format_for_sse(data: dict) -> str:
    """Convert dictionary to SSE-compatible JSON string"""
    return "data: " + json.dumps(data) + "\n\n"

# Generic checker function that contains the common logic
async def generic_checker(item: dict):
    """Generic function for checking CUDA solutions on GPU"""
    solution_code = item["solution_code"]
    problem_name = item["problem"]
     # Use the GPU type from the request if provided, otherwise it will default
    # to the smallest GPU (T4)
    gpu = item.get("gpu") or "T4"

    problem = load_problem_module(problem_name)

    def json_stream():
        for result in run_checker(problem, solution_code, gpu=gpu):
            yield format_for_sse(result)

    return StreamingResponse(
        json_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

async def generic_benchmark(item: dict):
    """Generic function for benchmarking CUDA solutions on GPU"""
    solution_code = item["solution_code"]
    problem_name = item["problem"]
    # Use the GPU type from the request if provided, otherwise it will default
    # to the smallest GPU (T4)
    gpu = item.get("gpu") or "T4"

    problem = load_problem_module(problem_name)

    def json_stream():
        for result in run_benchmark(problem, solution_code, gpu=gpu):
            yield format_for_sse(result)

    return StreamingResponse(
        json_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# GPU-specific endpoints
@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def checker_t4(item: dict):
    return await generic_checker(item)

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def checker_a10g(item: dict):
    return await generic_checker(item)

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def checker_h100(item: dict):
    return await generic_checker(item)

@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def checker_a100_80gb(item: dict):
    return await generic_checker(item)

@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def benchmark_t4(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def benchmark_a10g(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def benchmark_h100(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def benchmark_a100_80gb(item: dict):
    return await generic_benchmark(item)
    
