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


MAX_SANDBOX_OUTPUT_BYTES = 64 * 1024  # Cap console streaming to 64 KiB for responsiveness


def run_checker(
    problem_name: str,
    problem_def: str,
    solution_func,
    dtype: str,
    language: str,
    compiled_lib: bytes = None,
    solution_code: str = None,
    param_func=None,
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
        compiled_lib: Compiled library bytes for CUDA/Mojo (enables anti-cheat fresh loading)
        solution_code: Source code for Python/Triton/CuTe (enables anti-cheat fresh loading)
        param_func: None for general submissions, non-None only for baseline submissions

    Returns:
        Iterator that yields JSON strings with test results
    """

    # Track temp files/dirs for cleanup
    temp_paths_to_cleanup = []
    
    # Determine if we can do fresh loading (prevents static variable cheats)
    can_fresh_load = (
        (language in ("cuda", "mojo") and compiled_lib is not None) or
        (language in ("python", "cute") and solution_code is not None)
    )
    
    def get_fresh_solution_func():
        """Load a fresh copy of the solution function with reset static state."""
        if language in ("cuda", "mojo") and compiled_lib is not None:
            func, temp_path = utils.load_fresh_solution_func(compiled_lib, problem, language)
            temp_paths_to_cleanup.append(temp_path)
            return func
        elif language in ("python", "cute") and solution_code is not None:
            func, temp_dir = utils.load_fresh_python_solution(solution_code, language)
            temp_paths_to_cleanup.append(temp_dir)
            return func
        else:
            # Fallback to original solution_func (no fresh loading)
            return solution_func
    
    def cleanup_temp_paths():
        """Clean up all temp files and directories created during checker."""
        for path in temp_paths_to_cleanup:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass

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
            with utils.ReferenceSolutionContext(dtype=dtype):
                expected_output = problem.reference_solution(*input_tensors).cpu()

            # Create actual_output with the same shape as expected_output
            actual_output = torch.zeros_like(expected_output, device="cuda").contiguous()

            # Load fresh solution for each test case to prevent static variable cheats
            current_solution_func = get_fresh_solution_func()

            if param_func is None:
                parameters = utils.make_parameters(
                    language, current_solution_func, input_tensors, actual_output, problem, test_case
                )
            else:
                parameters = param_func(
                    language, current_solution_func, input_tensors, actual_output, problem, test_case
                )
            current_solution_func(*parameters)

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
    
    finally:
        # Always clean up temp files
        cleanup_temp_paths()


@utils.subproc_generator(timeout=60)
def run_sample_case(
    problem_name, problem_def, solution_code, compiled_lib, dtype, language, param_func=None
):
    """
    Run the sample test case of a problem and return result + output.
    """
    try:
        dtype = utils.DTYPE_MAP[dtype]
        problem = utils.load_problem_module(problem_name, problem_def)
        solution_func = utils.make_solution_func(language, solution_code, compiled_lib, problem)

        sample = problem.generate_sample(dtype)
        input_tensors = sample["create_inputs"]()
        expected_output = problem.reference_solution(*input_tensors).cpu()
        actual_output = torch.zeros_like(expected_output, device="cuda").contiguous()
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
            "input": utils.to_lossless_jsonable(
                [t if not isinstance(t, torch.Tensor) else t for t in input_tensors]
            ),
            "output": utils.to_lossless_jsonable(actual_output),
            "expected_output": utils.to_lossless_jsonable(expected_output),
            "debug_info": utils.to_lossless_jsonable(debug_info),
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
            with utils.ReferenceSolutionContext(dtype=dtype):
                expected_output = problem.reference_solution(*input_tensors).cpu()

            # Create actual_output with the same shape as expected_output
            actual_output = torch.zeros_like(expected_output, device="cuda").contiguous()

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
    problem_name: str,
    problem_def: str,
    solution_func,
    dtype: str,
    language: str,
    compiled_lib: bytes = None,
    solution_code: str = None,
    param_func=None,
):
    """
    Run benchmark on compiled CUDA solution

    Args:
        problem_name: Name of the problem
        problem_def: Problem instance
        solution_func: Callable function for the submitted solution
        dtype: Data type for the problem
        language: Programming language of the solution ("cuda", "python", or "mojo")
        compiled_lib: Compiled library bytes for CUDA/Mojo (enables anti-cheat fresh loading)
        solution_code: Source code for Python/Triton/CuTe (enables anti-cheat fresh loading)
        param_func: Optional custom parameter function (for baseline benchmarks)

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
            try:
                # Create inputs and reference output
                input_tensors = test_case["create_inputs"]()
                expected_output = problem.reference_solution(*input_tensors).cpu()
                actual_output = torch.zeros_like(expected_output, device="cuda").contiguous()

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
                    compiled_lib_bytes=compiled_lib,
                    solution_code=solution_code,
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
            # Average runtime in milliseconds
            avg_runtime_ms = statistics.mean([r["runtime_ms"] for r in test_results])

            # Average GFLOPS only if present (compute problems)
            has_gflops = any(("gflops" in r) and (r["gflops"] is not None) for r in test_results)
            if has_gflops:
                avg_gflops = statistics.mean(
                    [
                        r["gflops"]
                        for r in test_results
                        if ("gflops" in r) and (r["gflops"] is not None)
                    ]
                )
            else:
                avg_gflops = None
        else:
            avg_runtime_ms = 0
            avg_gflops = None

        if language == "python":
            try:
                temp_dir = os.path.dirname(solution_func.__code__.co_filename)
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        # Return final summary with additional metrics
        summary = {
            "status": "BENCHMARKED",
            "test_results": test_results,
            "avg_runtime_ms": avg_runtime_ms,
            "total_tests": test_count,
        }
        if avg_gflops is not None:
            summary["avg_gflops"] = avg_gflops

        yield summary

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
    Run sandbox on a compiled solution with real-time output streaming
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

        def cleanup_compiled_binary():
            try:
                os.unlink(compiled_lib_path)
            except OSError:
                pass

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
            output_bytes = 0
            output_limit_hit = False

            while len(streams_finished) < 2:
                if time.time() - start_time > timeout:
                    process.kill()
                    yield {
                        "status": "SANDBOX_TIMEOUT",
                        "message": "Binary execution timed out",
                        "details": f"Execution exceeded {timeout} second timeout",
                    }
                    cleanup_compiled_binary()
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

                    if stream_name in ("stdout", "stderr"):
                        prospective_total = output_bytes + len(line.encode("utf-8")) + 1
                        if prospective_total > MAX_SANDBOX_OUTPUT_BYTES:
                            output_limit_hit = True
                            process.kill()
                            break
                        output_bytes = prospective_total

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
            cleanup_compiled_binary()

            # Yield final result
            full_stdout = "\n".join(all_stdout)
            full_stderr = "\n".join(all_stderr)

            if output_limit_hit:
                yield {
                    "status": "SANDBOX_OUTPUT_LIMIT",
                    "message": f"Sandbox output exceeded limit of {MAX_SANDBOX_OUTPUT_BYTES // 1024} KB",
                    "stdout": full_stdout,
                    "stderr": full_stderr,
                    "details": "Execution stopped because the program produced too much console output.",
                }
                return

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
            cleanup_compiled_binary()

            yield {
                "status": "SANDBOX_ERROR",
                "message": "Permission denied executing binary",
                "details": "Unable to execute the compiled binary due to permission restrictions",
            }

        except Exception as e:
            cleanup_compiled_binary()

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
