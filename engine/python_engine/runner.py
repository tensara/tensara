import ctypes
import torch
import gc
from typing import Iterator
import statistics
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
        problem = utils.load_problem_module(problem_name)
        cuda_lib = utils.read_bytes_as_cuda_lib(compiled_lib)
        yield {"status": "running"}
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
                input_ptrs = [ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)) 
                             for tensor in input_tensors]
                
                benchmark_result = utils.run_dynamic_benchmark(
                    cuda_lib, 
                    problem, 
                    test_case, 
                    input_tensors, 
                    actual_output,
                    min_iterations=5,
                    max_iterations=20,
                    target_cv=0.01  # 1% target coefficient of variation
                )
                
                benchmark_results.append(benchmark_result)
                
                yield {
                    "status": "test_result",
                    "result": benchmark_result,
                    "totalTests": total_tests,
                }
                
                # Clean up memory
                del input_tensors, expected_output, actual_output, input_ptrs
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                benchmark_result = {
                    "test_id": test_id,
                    "name": test_name,
                    "status": "FAILED",
                    "debug_info": {"error": str(e)}
                }
        
        test_results = benchmark_results
        test_count = len(test_results)

        if test_count > 0:
            # Average mean GFLOPS
            avg_gflops = statistics.mean([r["gflops"] for r in test_results])

            # Average runtime in milliseconds
            avg_runtime_ms = statistics.mean([r["runtime_ms"] for r in test_results])

            # Calculate average standard deviation of GFLOPS within tests
            avg_stdev_gflops = statistics.mean([r["stdev_gflops"] for r in test_results])
        else:
            avg_gflops = 0
            avg_runtime_ms = 0
            avg_stdev_gflops = 0
            
        # Return final summary with additional metrics
        yield {
            "status": "success",
            "test_results": test_results,
            "gflops": avg_gflops,
            "runtime_ms": avg_runtime_ms,
            "stdev_gflops": avg_stdev_gflops,
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
