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


def run_checker(problem_name: str, problem_def: str, compiled: bytes | None, solution: str | None, dtype: str, language: str) -> Iterator[str]:
    """
    Check a submitted solution against the reference implementation
    and stream results as they become available
    
    Args:
        problem_name: Name of the problem
        problem_def: Problem instance
        compiled: Compiled CUDA code for the submitted solution (only for CUDA)
        solution: Source code of the solution (only for Triton)
        dtype: Data type for the problem
        language: Programming language of the solution ("cuda", "python", or "mojo")
        
    Returns:
        Iterator that yields JSON strings with test results
    """
    
    try:

        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)
        

        if language == "cuda":
            if not compiled:
                raise ValueError("Compiled bytes required for CUDA submissions")
                
            cuda_lib = utils.read_bytes_as_lib(compiled)
            func_sig = problem.get_function_signature()
            cuda_lib.solution.argtypes = func_sig["argtypes"]
            cuda_lib.solution.restype = func_sig["restype"]
            solution_func = cuda_lib.solution

        elif language == "mojo":
            mojo_lib = utils.read_bytes_as_lib(compiled)
            func_sig = problem.get_function_signature()
            mojo_lib.solution.argtypes = func_sig["argtypes"]
            mojo_lib.solution.restype = func_sig["restype"]
            solution_func = mojo_lib.solution

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

            if language == "cuda" or language == "mojo":
                input_ptrs = []
                for tensor, argtype in zip(input_tensors, solution_func.argtypes[:len(input_tensors)]):
                    if isinstance(tensor, torch.Tensor):
                        input_ptrs.append(ctypes.cast(tensor.data_ptr(), argtype))
                    else:
                        input_ptrs.append(argtype(tensor))
                output_ptr = ctypes.cast(actual_output.data_ptr(), solution_func.argtypes[len(input_ptrs)])
                extra_params = problem.get_extra_params(test_case)
                extra_params_casted = utils.cast_to_ctype(extra_params, solution_func.argtypes[-len(extra_params):], language)
                solution_func(*(input_ptrs + [output_ptr] + extra_params_casted))
            else:
                extra_params = problem.get_extra_params(test_case)
                solution_func(*(list(input_tensors) + [actual_output] + list(extra_params)))

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
            del input_tensors, expected_output, actual_output
            if language == "cuda" or language == "mojo":
                del input_ptrs, output_ptr
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


@utils.subproc_generator(timeout=60)
def run_sample_case(problem_name, problem_def, compiled, solution, dtype, language):
    """
    Run the sample test case of a problem and return result + output.
    """
    try:
        import io
        import contextlib
        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)
        sample = problem.generate_sample(dtype)
        input_tensors = sample["create_inputs"]()
        expected_output = problem.reference_solution(*input_tensors).cpu()
        actual_output = torch.zeros_like(expected_output, device="cuda")
        
        # For CUDA, use system-level capture
        if language in ("cuda", "mojo"):
            with utils.SystemOutputCapture() as capture:
                lib = utils.read_bytes_as_lib(compiled)
                sig = problem.get_function_signature()
                lib.solution.argtypes = sig["argtypes"]
                lib.solution.restype = sig["restype"]
                # input_ptrs = [ctypes.cast(t.data_ptr(), typ) for t, typ in zip(input_tensors, sig["argtypes"][:len(input_tensors)])]
                input_ptrs = []
                for t, typ in zip(input_tensors, sig["argtypes"][:len(input_tensors)]):
                    if isinstance(t, torch.Tensor):
                        input_ptrs.append(ctypes.cast(t.data_ptr(), typ))
                    elif isinstance(t, (int, float)):
                        input_ptrs.append(typ(t))  # pass by value
                    else:
                        raise TypeError(f"Unsupported input type: {type(t)}")

                output_ptr = ctypes.cast(actual_output.data_ptr(), sig["argtypes"][len(input_ptrs)])
                extra_params = problem.get_extra_params(sample)
                extra_params_casted = utils.cast_to_ctype(extra_params, sig["argtypes"][-len(extra_params):], language)
                lib.solution(*(input_ptrs + [output_ptr] + extra_params_casted))
            
            captured_stdout = capture.stdout_content
            captured_stderr = capture.stderr_content
        else:
            # Triton case - original approach
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
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
                # temp_path = utils.write_temp_triton(solution)
                # soln_func = utils.import_module_from_path(temp_path)
                solution_func(*(list(input_tensors) + [actual_output] + problem.get_extra_params(sample)))
            captured_stdout = stdout_buf.getvalue()
            captured_stderr = stderr_buf.getvalue()
            
        torch.cuda.synchronize()
        is_correct, debug_info = problem.verify_result(expected_output, actual_output.cpu(), dtype)
        yield {
            "status": "PASSED" if is_correct else "FAILED",
            "input": [
                t.cpu().numpy().tolist() if isinstance(t, torch.Tensor) else t
                for t in input_tensors
            ],
            "output": actual_output.cpu().numpy().tolist(),
            "expected_output": expected_output.cpu().numpy().tolist(),
            "debug_info": debug_info,
            "stdout": captured_stdout,
            "stderr": captured_stderr,
        }

    except Exception as e:
        yield {
            "status": "ERROR",
            "message": str(e),
            "details": traceback.format_exc()
        }


def run_sanity_check(problem_name: str, problem_def: str, compiled: bytes | None, solution: str | None, dtype: str, language: str):
    """
    Run sanity check on compiled CUDA solution
    """
    try:

        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)

        if language == "cuda":
            if not compiled:
                raise ValueError("Compiled bytes required for CUDA submissions")
                
            cuda_lib = utils.read_bytes_as_lib(compiled)
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

            if language == "cuda":
                input_ptrs = []
                for tensor, argtype in zip(input_tensors, solution_func.argtypes[:len(input_tensors)]):
                    if isinstance(tensor, torch.Tensor):
                        input_ptrs.append(ctypes.cast(tensor.data_ptr(), argtype))
                    else:
                        input_ptrs.append(argtype(tensor))
                output_ptr = ctypes.cast(actual_output.data_ptr(), solution_func.argtypes[len(input_ptrs)])
                extra_params = problem.get_extra_params(test_case)
                extra_params_casted = utils.cast_to_ctype(extra_params, solution_func.argtypes[-len(extra_params):], language)
                solution_func(*(input_ptrs + [output_ptr] + extra_params_casted))
            else:
                extra_params = problem.get_extra_params(test_case)
                solution_func(*(list(input_tensors) + [actual_output] + list(extra_params)))

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
            del input_tensors, expected_output, actual_output
            if language == "cuda":
                del input_ptrs, output_ptr
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
            shutil.rmtree(temp_dir)

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


def run_benchmark(problem_name: str, problem_def: str, compiled: bytes | None, solution: str | None, dtype: str, language: str):
    """
    Run benchmark on compiled CUDA solution
    
    Args:
        problem_name: Name of the problem
        problem_def: Problem instance  
        compiled: Compiled CUDA code for the submitted solution
        solution: Source code of the solution (only for Triton)
        dtype: Data type for the problem
        language: Programming language of the solution ("cuda", "python", or "mojo")
    
    Yields:
        Dictionary objects with benchmark status updates
    """
    try:
        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)

        if language == "cuda":
            if not compiled:
                raise ValueError("Compiled bytes required for CUDA submissions")
                
            cuda_lib = utils.read_bytes_as_lib(compiled)
            func_sig = problem.get_function_signature()
            cuda_lib.solution.argtypes = func_sig["argtypes"]
            cuda_lib.solution.restype = func_sig["restype"]
            solution_func = cuda_lib.solution
            
        elif language == "mojo":
            mojo_lib = utils.read_bytes_as_lib(compiled)
            func_sig = problem.get_function_signature()
            mojo_lib.solution.argtypes = func_sig["argtypes"]
            mojo_lib.solution.restype = func_sig["restype"]
            solution_func = mojo_lib.solution

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
                    target_cv=0.01  # 1% target coefficient of variation
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
