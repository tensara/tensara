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


def run_checker(problem_name: str, problem_def: str, compiled: bytes | None, solution: str | None, dtype: str, language: str) -> Iterator[dict]:
    """
    Check a submitted solution against the reference implementation
    and stream results as they become available
    
    Args:
        problem_name: Name of the problem
        problem_def: Problem instance definition string
        compiled: Compiled CUDA code for the submitted solution (only for CUDA)
        solution: Source code of the solution (only for Triton)
        dtype: Data type for the problem
        language: Programming language of the solution ("cuda" or "python")
        
    Returns:
        Iterator that yields dictionaries with test results
    """
    
    try:
        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)
        
        # Load solution function based on language
        if language == "cuda":
            if not compiled:
                raise ValueError("Compiled bytes required for CUDA submissions")
                
            cuda_lib = utils.read_bytes_as_cuda_lib(compiled)
            func_sig = problem.get_function_signature()
            cuda_lib.solution.argtypes = func_sig["argtypes"]
            cuda_lib.solution.restype = func_sig["restype"]
            solution_func = cuda_lib.solution
        elif language == "python":
            if not solution:
                raise ValueError("Source code required for Triton submissions")
            
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "triton_solution.py")
            
            # This is needed because @jit has to read the source code
            with open(temp_path, 'w') as f:
                f.write(solution)
                
            spec = importlib.util.spec_from_file_location("triton_solution", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            solution_func = module.solution
        else:
            raise ValueError(f"Unsupported language: {language}")

        test_cases = problem.generate_test_cases(dtype)
        total_tests = len(test_cases)
        test_results = []
        passed_tests = 0

        yield {"status": "CHECKING"}

        start_time = time.time()
        time_limit = problem.time_limit

        # Run each test case
        for test_id, test_case in enumerate(test_cases, 1):
            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)"
                }
                return

            test_name = test_case["name"]
            all_params = test_case["create_inputs"]()
            
            # For reference solution, we pass all inputs to get the expected output
            with torch.autocast("cuda", enabled=False, dtype=dtype):
                old_tf32_setting = torch.backends.cuda.matmul.allow_tf32
                old_cudnn_tf32_setting = torch.backends.cudnn.allow_tf32
                
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

                expected_output = problem.reference_solution(*all_params).cpu()

                torch.backends.cuda.matmul.allow_tf32 = old_tf32_setting
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32_setting

            # Create actual_output with the same shape as expected_output
            actual_output = torch.zeros_like(expected_output, device='cuda')  # Ensure it's on GPU

            # For user solution, we need to handle CUDA vs Python differently
            if language == "cuda":
                # Convert inputs to ctypes for CUDA
                param_count = len(all_params)
                
                # Process all parameters for CUDA
                all_inputs = []
                for param, argtype in zip(all_params, solution_func.argtypes[:param_count]):
                    if isinstance(param, torch.Tensor):
                        all_inputs.append(ctypes.cast(param.data_ptr(), argtype))
                    else:
                        all_inputs.append(argtype(param))
                
                # Add output tensor
                output_ptr = ctypes.cast(actual_output.data_ptr(), solution_func.argtypes[param_count])
                all_inputs.append(output_ptr)
                
                # Call the solution with all parameters
                solution_func(*all_inputs)
            else:
                # For Python/Triton, pass all parameters plus the output tensor
                solution_func(*(list(all_params) + [actual_output]))

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
            del all_params, expected_output, actual_output
            if language == "cuda":
                del all_inputs, output_ptr
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
            shutil.rmtree(temp_dir)

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



def run_benchmark(problem_name: str, problem_def: str, compiled: bytes | None, solution: str | None, dtype: str, language: str) -> Iterator[dict]:
    """
    Run benchmark on compiled CUDA solution
    
    Args:
        problem_name: Name of the problem
        problem_def: Problem instance definition string
        compiled: Compiled CUDA code for the submitted solution (only for CUDA)
        solution: Source code of the solution (only for Triton)
        dtype: Data type for the problem
        language: Programming language of the solution ("cuda" or "python")
    
    Yields:
        Dictionary objects with benchmark status updates
    """
    try:
        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)

        # Load solution function based on language
        if language == "cuda":
            if not compiled:
                raise ValueError("Compiled bytes required for CUDA submissions")
                
            cuda_lib = utils.read_bytes_as_cuda_lib(compiled)
            func_sig = problem.get_function_signature()
            cuda_lib.solution.argtypes = func_sig["argtypes"]
            cuda_lib.solution.restype = func_sig["restype"]
            solution_func = cuda_lib.solution
        elif language == "python":
            if not solution:
                raise ValueError("Source code required for Triton submissions")
            
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "triton_solution.py")
            
            # This is needed because @jit has to read the source code
            with open(temp_path, 'w') as f:
                f.write(solution)
                
            spec = importlib.util.spec_from_file_location("triton_solution", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            solution_func = module.solution
        else:
            raise ValueError(f"Unsupported language: {language}")

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
                # Get all parameters from the test case
                all_params = test_case["create_inputs"]()
                # Calculate reference output using all parameters
                expected_output = problem.reference_solution(*all_params).cpu()
                actual_output = torch.zeros_like(expected_output, device='cuda')
                
                benchmark_result = utils.run_dynamic_benchmark(
                    solution_func, 
                    problem, 
                    test_id,
                    test_case, 
                    all_params, 
                    actual_output,
                    language=language,
                    min_iterations=5,
                    max_iterations=20,
                    target_cv=0.01  # 1% target coefficient of variation
                )
                
                benchmark_results.append(benchmark_result)
                
                yield {
                    "status": "BENCHMARK_RESULT",
                    "result": benchmark_result,
                    "total_tests": total_tests,
                }
                
                # Clean up memory
                del all_params, expected_output, actual_output
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
            shutil.rmtree(temp_dir)

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
