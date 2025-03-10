import json
import ctypes
import torch
import gc
from typing import Iterator
from problem import Problem
import utils

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
    
    try:
        # Send compilation status
        yield json.dumps({"status": "compiling"})

        files = {
            "solution.cu": solution_code
        }

        cuda_lib = utils.setup_solution(problem, solution_code, gpu)

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
        
    except utils.NVCCError as e:
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
        lib_path = utils.run_nvcc(gpu, files, "solution")

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
        benchmark_summary = utils.format_benchmark_summary(benchmark_results)
        yield json.dumps(benchmark_summary)

    except Exception as e:
        # Handle any unexpected errors
        import traceback
        yield {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
