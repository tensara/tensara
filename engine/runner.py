import ctypes
import torch
import gc
from typing import Iterator
import statistics
import utils
import os
import importlib.util
import tempfile
import shutil
import traceback
import time


def run_checker(problem_name: str, problem_def: str, solution_func, dtype: str, language: str, param_func = None) -> Iterator[str]:
    """
    Check a submitted solution against the reference implementation
    and stream results as they become available
    
    Args:
        problem_name: Name of the problem
        problem_def: Problem instance
        solution_func: Callable function for the submitted solution
        dtype: Data type for the problem
        language: Programming language of the solution ("cuda", "python", or "mojo")
        
    Returns:
        Iterator that yields JSON strings with test results
    """
    
    try:

        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)
        
        test_cases = problem.generate_test_cases(dtype)
        total_tests = len(test_cases)
        test_results = []
        passed_tests = 0

        yield {"status": "CHECKING"}

        start_time = time.time()
        time_limit = problem.time_limit

        for test_id, test_case in enumerate(test_cases, 1):
            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)"
                }
                return

            test_name = test_case["name"]
            input_tensors = test_case["create_inputs"]()
            
            # Get the reference solution and move it to CPU
            with torch.autocast("cuda", enabled=False, dtype=dtype):
                old_tf32_setting = torch.backends.cuda.matmul.allow_tf32
                old_cudnn_tf32_setting = torch.backends.cudnn.allow_tf32
                
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

                expected_output = problem.reference_solution(*input_tensors).cpu()

                torch.backends.cuda.matmul.allow_tf32 = old_tf32_setting
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32_setting

            # Create actual_output with the same shape as expected_output
            actual_output = torch.zeros_like(expected_output, device='cuda')  # Ensure it's on GPU

            if param_func is None:
                parameters = utils.make_parameters(language, solution_func, input_tensors, actual_output, problem, test_case)
            else:
                parameters = param_func(language, solution_func, input_tensors, actual_output, problem, test_case)
            solution_func(*parameters)

            torch.cuda.synchronize()

            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)"
                }
                return

            # Move to CPU for comparison
            is_correct, debug_info = problem.verify_result(expected_output, actual_output.cpu(), dtype)

            # Clean up memory
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
                
        if language == "python":
            try:
                temp_dir = os.path.dirname(solution_func.__code__.co_filename)
                shutil.rmtree(temp_dir)
            except Exception as e:
                pass

        # Final status message
        yield {
            "status": "CHECKED",
            "test_results": test_results,
            "total_tests": total_tests
        }


    except utils.NVCCError as e:
        yield {
            "status": "RUNTIME_ERROR",
            "message": "NVCC: Compilation Failed",
            "details": e.args[0],
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

def run_sanity_check(problem_name: str, problem_def: str, solution_func, dtype: str, language: str, param_func = None):
    """
    Run sanity check on compiled CUDA solution
    """
    try:

        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)

        test_cases = problem.generate_test_cases(dtype)
        total_tests = len(test_cases)

        start_time = time.time()
        time_limit = problem.time_limit

        # Only take the first test case
        test_cases = test_cases[:1]
        for test_id, test_case in enumerate(test_cases, start=1):
            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)"
                }
                return

            test_name = test_case["name"]
            input_tensors = test_case["create_inputs"]()
            
            # Get the reference solution and move it to CPU
            with torch.autocast("cuda", enabled=False, dtype=dtype):
                old_tf32_setting = torch.backends.cuda.matmul.allow_tf32
                old_cudnn_tf32_setting = torch.backends.cudnn.allow_tf32
                
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

                expected_output = problem.reference_solution(*input_tensors).cpu()

                torch.backends.cuda.matmul.allow_tf32 = old_tf32_setting
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32_setting

            # Create actual_output with the same shape as expected_output
            actual_output = torch.zeros_like(expected_output, device='cuda')  # Ensure it's on GPU

            if param_func is None:
                parameters = utils.make_parameters(language, solution_func, input_tensors, actual_output, problem, test_case)
            else:
                parameters = param_func(language, solution_func, input_tensors, actual_output, problem, test_case)
            solution_func(*parameters)

            torch.cuda.synchronize()

            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)"
                }
                return

            # Move to CPU for comparison
            is_correct, debug_info = problem.verify_result(expected_output, actual_output.cpu(), dtype)

            # Clean up memory
            del input_tensors, expected_output, actual_output, parameters
            gc.collect()
            torch.cuda.empty_cache()

            test_result = {
                "test_id": test_id,
                "name": test_name,
            }

            if is_correct:
                test_result["status"] = "PASSED"
                yield {
                    "status": "SANITY_CHECK_PASSED",
                    "total_tests": total_tests
                }
            else:
                test_result["status"] = "FAILED"
                yield {
                    "status": "WRONG_ANSWER",
                    "debug_info": debug_info,
                    "total_tests": total_tests,
                }
                return
                
        if language == "python":
            try:
                temp_dir = os.path.dirname(solution_func.__code__.co_filename)
                shutil.rmtree(temp_dir)
            except Exception as e:
                pass

    except utils.NVCCError as e:
        yield {
            "status": "RUNTIME_ERROR",
            "message": "NVCC: Compilation Failed",
            "details": e.args[0],
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


def run_benchmark(problem_name: str, problem_def: str, solution_func, dtype: str, language: str, param_func = None):
    """
    Run benchmark on compiled CUDA solution
    
    Args:
        problem_name: Name of the problem
        problem_def: Problem instance  
        solution_func: Callable function for the submitted solution
        dtype: Data type for the problem
        language: Programming language of the solution ("cuda", "python", or "mojo")
    
    Yields:
        Dictionary objects with benchmark status updates
    """
    try:
        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)

        yield {"status": "BENCHMARKING"}
            
        # Get test cases from the problem
        test_cases = problem.generate_test_cases(dtype)
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
                
                benchmark_result = utils.run_dynamic_benchmark(
                    solution_func, 
                    problem, 
                    test_id,
                    test_case, 
                    input_tensors, 
                    actual_output,
                    language=language,
                    min_iterations=5,
                    max_iterations=20,
                    target_cv=0.01,  # 1% target coefficient of variation
                    param_func=param_func
                )
                
                benchmark_results.append(benchmark_result)
                
                yield {
                    "status": "BENCHMARK_RESULT",
                    "result": benchmark_result,
                    "total_tests": total_tests,
                }
                
                # Clean up memory
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
        
        test_results = benchmark_results
        test_count = len(test_results)

        if test_count > 0:
            # Average mean GFLOPS
            avg_gflops = statistics.mean([r["gflops"] for r in test_results])

            # Average runtime in milliseconds
            avg_runtime_ms = statistics.mean([r["runtime_ms"] for r in test_results])
        else:
            avg_gflops = 0
            avg_runtime_ms = 0
        
        if language == "python":
            try:
                temp_dir = os.path.dirname(solution_func.__code__.co_filename)
                shutil.rmtree(temp_dir)
            except Exception as e:
                pass


        # Return final summary with additional metrics
        yield {
            "status": "BENCHMARKED",
            "test_results": test_results,
            "avg_gflops": avg_gflops,
            "avg_runtime_ms": avg_runtime_ms,
            "total_tests": test_count,
        }
    
    except utils.NVCCError as e:
        yield {
            "status": "RUNTIME_ERROR",
            "message": "NVCC: Compilation Failed",
            "details": e.args[0],
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
            "message": str(e),
            "details": traceback.format_exc()
        }
