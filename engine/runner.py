import subprocess
import torch
import gc
from typing import Iterator
import statistics
import utils
import os
import tempfile
import shutil
import traceback
import time
import queue
import threading
import io
import contextlib


def run_checker(
    problem_name: str, problem_def: str, solution_func, dtype: str, language: str, param_func=None
) -> Iterator[str]:
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
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)",
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
            actual_output = torch.zeros_like(expected_output, device="cuda")  # Ensure it's on GPU

            if param_func is None:
                parameters = utils.make_parameters(
                    language, solution_func, input_tensors, actual_output, problem, test_case
                )
            else:
                parameters = param_func(
                    language, solution_func, input_tensors, actual_output, problem, test_case
                )
            solution_func(*parameters)

            torch.cuda.synchronize()

            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)",
                }
                return

            # Move to CPU for comparison
            is_correct, debug_info = problem.verify_result(
                expected_output, actual_output.cpu(), dtype
            )

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
            except Exception:
                pass

        # Final status message
        yield {"status": "CHECKED", "test_results": test_results, "total_tests": total_tests}

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
def run_sample_case(problem_name, problem_def, solution_func, dtype, language, param_func=None):
    """
    Run the sample test case of a problem and return result + output.
    """
    try:
        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)

        sample = problem.generate_sample(dtype)
        input_tensors = sample["create_inputs"]()
        expected_output = problem.reference_solution(*input_tensors).cpu()
        actual_output = torch.zeros_like(expected_output, device="cuda")
        if param_func is None:
            parameters = utils.make_parameters(
                language, solution_func, input_tensors, actual_output, problem, sample
            )
        else:
            parameters = param_func(
                language, solution_func, input_tensors, actual_output, problem, sample
            )

        if language in ("cuda", "mojo"):
            with utils.SystemOutputCapture() as capture:
                solution_func(*parameters)

            captured_stdout = capture.stdout_content
            captured_stderr = capture.stderr_content
        else:
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                solution_func(*parameters)

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
        yield {"status": "ERROR", "message": str(e), "details": traceback.format_exc()}


def run_sanity_check(
    problem_name: str, problem_def: str, solution_func, dtype: str, language: str, param_func=None
):
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
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)",
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
            actual_output = torch.zeros_like(expected_output, device="cuda")  # Ensure it's on GPU

            if param_func is None:
                parameters = utils.make_parameters(
                    language, solution_func, input_tensors, actual_output, problem, test_case
                )
            else:
                parameters = param_func(
                    language, solution_func, input_tensors, actual_output, problem, test_case
                )
            solution_func(*parameters)

            torch.cuda.synchronize()

            if time.time() - start_time > time_limit:
                yield {
                    "status": "TIME_LIMIT_EXCEEDED",
                    "message": "Time Limit Exceeded",
                    "details": f"Execution exceeded time limit of {time_limit:.2f}s (took {time.time() - start_time:.2f}s)",
                }
                return

            # Move to CPU for comparison
            is_correct, debug_info = problem.verify_result(
                expected_output, actual_output.cpu(), dtype
            )

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
                yield {"status": "SANITY_CHECK_PASSED", "total_tests": total_tests}
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
            except Exception:
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


def run_benchmark(
    problem_name: str, problem_def: str, solution_func, dtype: str, language: str, param_func=None
):
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
                actual_output = torch.zeros_like(expected_output, device="cuda")

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
                    param_func=param_func,
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
            except Exception:
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
        yield {"status": "ERROR", "message": str(e), "details": traceback.format_exc()}


def run_sandbox(compiled_lib: bytes, solution_code: str):
    """
    Run sandbox on compiled CUDA solution with real-time output streaming
    """
    try:
        if not compiled_lib:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Compilation Failed",
                "details": "No compiled library provided",
            }
            return

        yield {
            "status": "SANDBOX_RUNNING",
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(compiled_lib)
            f.flush()
            compiled_lib_path = f.name

        try:
            os.chmod(compiled_lib_path, 0o755)

            process = subprocess.Popen(
                [compiled_lib_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Function to read from a pipe and put lines in queue
            def read_pipe(pipe, stream_name, output_queue):
                try:
                    for line in iter(pipe.readline, ""):
                        if line:
                            output_queue.put((stream_name, line.rstrip("\n")))
                    pipe.close()
                except Exception as e:
                    output_queue.put(("error", f"Error reading {stream_name}: {str(e)}"))
                finally:
                    output_queue.put((stream_name, None))  # Signal end of stream

            output_queue = queue.Queue()

            stdout_thread = threading.Thread(
                target=read_pipe, args=(process.stdout, "stdout", output_queue)
            )
            stderr_thread = threading.Thread(
                target=read_pipe, args=(process.stderr, "stderr", output_queue)
            )

            stdout_thread.start()
            stderr_thread.start()

            start_time = time.time()
            streams_finished = set()
            all_stdout = []
            all_stderr = []
            timeout = 4 * 60

            while len(streams_finished) < 2:
                if time.time() - start_time > timeout:
                    process.kill()
                    yield {
                        "status": "SANDBOX_TIMEOUT",
                        "message": "Binary execution timed out",
                        "details": f"Execution exceeded {timeout} second timeout",
                    }
                    try:
                        os.unlink(compiled_lib_path)
                    except:
                        pass
                    return

                try:
                    stream_name, line = output_queue.get(timeout=0.1)

                    if line is None:
                        streams_finished.add(stream_name)
                        continue

                    if stream_name == "stdout":
                        all_stdout.append(line)
                    elif stream_name == "stderr":
                        all_stderr.append(line)

                    yield {
                        "status": "SANDBOX_OUTPUT",
                        "stream": stream_name,
                        "line": line,
                        "timestamp": time.time(),
                    }

                except queue.Empty:
                    if process.poll() is not None:
                        continue
                    continue

            # Wait for threads to complete
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            # Get return code
            return_code = process.wait()

            # Clean up temporary file
            try:
                os.unlink(compiled_lib_path)
            except:
                pass

            # Yield final result
            full_stdout = "\n".join(all_stdout)
            full_stderr = "\n".join(all_stderr)

            if return_code == 0:
                yield {
                    "status": "SANDBOX_SUCCESS",
                    "stdout": full_stdout,
                    "stderr": full_stderr,
                    "return_code": return_code,
                }
            else:
                yield {
                    "status": "SANDBOX_ERROR",
                    "message": f"Binary execution failed with return code {return_code}",
                    "stdout": full_stdout,
                    "stderr": full_stderr,
                    "return_code": return_code,
                }

        except PermissionError:
            try:
                os.unlink(compiled_lib_path)
            except:
                pass

            yield {
                "status": "SANDBOX_ERROR",
                "message": "Permission denied executing binary",
                "details": "Unable to execute the compiled binary due to permission restrictions",
            }

        except Exception as e:
            try:
                os.unlink(compiled_lib_path)
            except:
                pass

            yield {
                "status": "SANDBOX_ERROR",
                "message": f"Unexpected error during execution: {str(e)}",
                "details": traceback.format_exc(),
            }

    except Exception as e:  # Removed utils.NVCCError since it's not defined here
        yield {
            "status": "ERROR",
            "message": str(e.__class__.__name__),
            "details": traceback.format_exc(),
        }
