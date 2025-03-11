import json
import ctypes
import time
import torch
import gc
from typing import Iterator
from problem import Problem
import utils

def run_checker(problem_name: str, compiled: bytes) -> Iterator[str]:
    """
    Check a submitted CUDA solution against the reference implementation
    and stream results as they become available
    
    Args:
        problem_name: Name of the problem
        compiled: Compiled CUDA code for the submitted solution
        
    Returns:
        Iterator that yields JSON strings with test results
    """
    
    try:
        problem = utils.load_problem_module(problem_name)
        cuda_lib = utils.read_bytes_as_cuda_lib(compiled)
        func_sig = problem.get_function_signature()
        cuda_lib.solution.argtypes = func_sig["argtypes"]
        cuda_lib.solution.restype = func_sig["restype"]

        test_cases = problem.generate_test_cases()
        total_tests = len(test_cases)
        test_results = []
        passed_tests = 0
        has_failed = False

        yield {"status": "running"}

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
                
                yield {
                    "status": "test_result",
                    "result": test_result,
                    "totalTests": total_tests,
                }
                
                
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
                
                yield {
                    "status": "test_result",
                    "result": test_result,
                    "totalTests": total_tests,
                }
        
        # Final status message
        yield {
            "status": "complete",
            "passed": not has_failed,
            "test_results": test_results,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
        }

    except utils.NVCCError as e:
        yield {
            "status": "error",
            "error": "Compilation failed",
            "details": e.args[0],
            "test_results": [],
            "passed_tests": 0,
            "total_tests": 0,
        }


def run_benchmark(problem_name: str, compiled_lib: bytes):
    """
    Run benchmark on compiled CUDA solution
    
    Args:
        problem_name: Name of the problem
        compiled_lib: Compiled CUDA code for the submitted solution
    
    Yields:
        Dictionary objects with benchmark status updates
    """
    try:
        # Compile the solution
        yield json.dumps({"status": "compiling"})
        problem = utils.load_problem_module(problem_name)
        cuda_lib = utils.read_bytes_as_cuda_lib(compiled_lib)
        yield json.dumps({"status": "running"})
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
        utils.prepare_gpu()
        
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
                benchmark_result = utils.run_dynamic_benchmark(
                    cuda_lib, 
                    problem, 
                    test_case, 
                    input_tensors, 
                    actual_output,
                    min_iterations=10,
                    max_iterations=50,
                    target_cv=0.01  # 1% target coefficient of variation for high accuracy
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
        
        # Calculate overall statistics - compute the average GFLOPS across all tests
        test_results = benchmark_results
        test_count = len(test_results)
        avg_gflops = 0
        
        if test_count > 0:
            gflops_values = [result["performance_stats"]["mean_gflops"] 
                            for result in test_results 
                            if result["status"] == "PASSED" and "performance_stats" in result]
            
            if gflops_values:
                avg_gflops = sum(gflops_values) / len(gflops_values)
        
        # Compute additional important aggregate metrics from run_dynamic_benchmark
        all_runtime_stats = {}
        all_performance_stats = {}
        all_memory_stats = {}
        
        if test_count > 0:
            # Collect all the important metrics
            for result in test_results:
                if result["status"] == "PASSED":
                    # Get the detailed statistics
                    if "runtime_stats" in result:
                        for key, value in result["runtime_stats"].items():
                            all_runtime_stats.setdefault(key, []).append(value)
                    
                    if "performance_stats" in result:
                        for key, value in result["performance_stats"].items():
                            all_performance_stats.setdefault(key, []).append(value)
                    
                    if "memory_stats" in result:
                        for key, value in result["memory_stats"].items():
                            all_memory_stats.setdefault(key, []).append(value)
        
        # Calculate aggregates for the important metrics
        runtime_summary = {}
        performance_summary = {}
        memory_summary = {}
        
        # Process runtime stats
        if all_runtime_stats:
            for key, values in all_runtime_stats.items():
                if key != "cv":  # CV doesn't make sense to average
                    runtime_summary[f"avg_{key}"] = sum(values) / len(values)
                    runtime_summary[f"min_{key}"] = min(values)
                    runtime_summary[f"max_{key}"] = max(values)
        
        # Process performance stats
        if all_performance_stats:
            for key, values in all_performance_stats.items():
                if key != "cv":  # CV doesn't make sense to average
                    performance_summary[f"avg_{key}"] = sum(values) / len(values)
                    performance_summary[f"min_{key}"] = min(values)
                    performance_summary[f"max_{key}"] = max(values)
        
        # Process memory stats
        if all_memory_stats:
            for key, values in all_memory_stats.items():
                memory_summary[f"avg_{key}"] = sum(values) / len(values)
                memory_summary[f"min_{key}"] = min(values)
                memory_summary[f"max_{key}"] = max(values)
        
        # Return the required format along with the detailed metrics
        yield {
            "status": "success",
            "test_results": test_results,
            "average_gflops": avg_gflops,
            "total_tests": test_count,
        }
    except Exception as e:
        # Handle any unexpected errors
        import traceback
        yield {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
