import json
import sys
from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import modal
from pathlib import Path
import utils
import runner
from problem import Problem
import os

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
RUNTIME_IMAGE_NAME = "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
CURR_DIR = Path(__file__).parent


PIP_PACKAGES = ["torch", "numpy", "fastapi[standard]", "triton"]
LOCAL_SOURCE = ["utils", "runner", "problem"]

devel_image = (
    modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.11")
    .apt_install(["build-essential", "gcc", "g++"])
    .env({"CC": "gcc"})
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_SOURCE)
)

runtime_image = (
    modal.Image.from_registry(RUNTIME_IMAGE_NAME, add_python="3.11")
    .apt_install(["build-essential", "gcc", "g++"])
    .env({"CC": "gcc"})
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_SOURCE)
)

<<<<<<< HEAD
<<<<<<< HEAD
app = modal.App("tensara", image=devel_image)
web_app = FastAPI()

def binary_runner(type: str, compiled_lib: bytes, solution_code: str, problem_name: str, problem_def: str, dtype: str, language: str):
    gen = None
    if type == "checker":
        gen = runner.run_checker(problem_name, problem_def, compiled_lib, solution_code, dtype, language)
    elif type == "benchmark":
        gen = runner.run_benchmark(problem_name, problem_def, compiled_lib, solution_code, dtype, language)
    else:
        raise ValueError(f"Unknown binary type: {type}")

    for event in gen:
        yield event

gpu_runners = {
    gpu: app.function(
        image=runtime_image,
        name=f"runner_{gpu}",
        gpu=gpu,
        enable_memory_snapshot=True,
    )(binary_runner)
    for gpu in utils.GPU_COMPUTE_CAPABILITIES.keys()
}

for gpu in gpu_runners:
    globals()[f"runner_{gpu}"] = gpu_runners[gpu]

def gen_wrapper(gen):
    for event in gen:
        yield "data: " + json.dumps(event, allow_nan=False) + "\n\n"
=======
async def generic_checker(item: dict):
    """Common implementation for all checker endpoints."""
=======
@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def benchmark_t4(item: dict):
    async def generate_benchmark_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                solution_path = Path(tmpdir) / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = Path(tmpdir) / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                os.system("cp /root/benchmark.cu /root/core.hpp /root/Makefile " + tmpdir)            

                os.chdir(tmpdir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed", 
                        "details": os.popen("make 2>&1").read()
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                import subprocess
                result = subprocess.run(["./benchmark"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr
                    }) + "\n\n"
                    return
                
                try:
                    lines = result.stdout.strip().split('\n')
                    test_results = []
                    
                    for line in lines[:-1]:
                        test_id, name, runtime_ms, gflops = line.split(',')
                        test_result = {
                            "test_id": int(test_id),
                            "name": name,
                            "runtime_ms": float(runtime_ms),
                            "gflops": float(gflops)
                        }
                        test_results.append(test_result)
                        yield "data: " + json.dumps({
                            "status": "test_result",
                            "result": test_result
                        }) + "\n\n"
                    
                    avg_gflops = float(lines[-1])
                    
                    yield "data: " + json.dumps({
                        "status": "success",
                        "test_results": test_results,
                        "average_gflops": avg_gflops
                    }) + "\n\n"
                    
                except Exception as e:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Failed to parse benchmark output",
                        "details": str(e)
                    }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e)
            }) + "\n\n"

    return StreamingResponse(generate_benchmark_results(), media_type="text/event-stream")

@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def checker_t4(item: dict):
>>>>>>> parent of 24b4de1 (refactored for duplicate code, now can make changes to the testing and benchmark wihtout having to change it everywhere)
    async def generate_checker_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                tmpdir_path = Path(tmpdir)
                
                checker_dir = tmpdir_path / "checker"
                checker_dir.mkdir()
                os.system(f"cp /root/checker/core.hpp /root/checker/Makefile {str(checker_dir)}")
                
                solution_path = checker_dir / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = checker_dir / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                reference_path = checker_dir / "reference.cu"
                reference_path.write_text(item["reference_code"])
                
                os.system(f"cp /root/checker.cu {str(checker_dir)}")
                
                os.chdir(checker_dir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed",
                        "details": os.popen("make 2>&1").read(),
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                import subprocess
                result = subprocess.run(["./checker"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr,
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                lines = result.stdout.strip().split('\n')
                test_results = []
                passed_tests = 0
                
                for line in lines[:-1]:
                    test_id, name, status = line.split(',')
                    test_result = {
                        "test_id": int(test_id),
                        "name": name,
                        "status": status.strip()
                    }
                    test_results.append(test_result)
                    if status.strip() == "PASSED":
                        passed_tests += 1
                    yield "data: " + json.dumps({
                        "status": "test_result",
                        "result": test_result
                    }) + "\n\n"
                
                overall_status = lines[-1].strip()
                total_tests = len(test_results)
                
                yield "data: " + json.dumps({
                    "status": "complete",
                    "passed": overall_status == "PASSED",
                    "test_results": test_results,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests
                }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e),
                "test_results": []
            }) + "\n\n"
>>>>>>> parent of bdd95c4 (update how the test case data is sent to render the total cases correctly, revamped api.py completely)

        
@web_app.post("/checker-{gpu}")
async def checker(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return 404

<<<<<<< HEAD
<<<<<<< HEAD
    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    dtype = req["dtype"]
    language = req["language"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])
=======
async def generic_benchmark(item: dict):
    """Common implementation for all benchmark endpoints."""
=======

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def benchmark_h100(item: dict):
>>>>>>> parent of 24b4de1 (refactored for duplicate code, now can make changes to the testing and benchmark wihtout having to change it everywhere)
    async def generate_benchmark_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                solution_path = Path(tmpdir) / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = Path(tmpdir) / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                os.system("cp /root/benchmark.cu /root/core.hpp /root/Makefile " + tmpdir)            
>>>>>>> parent of 8e13d2b (import fix)

<<<<<<< HEAD
<<<<<<< HEAD
    def create_stream():
        yield {"status": "compiling"}
=======
                # Compile
=======
>>>>>>> parent of 24b4de1 (refactored for duplicate code, now can make changes to the testing and benchmark wihtout having to change it everywhere)
                os.chdir(tmpdir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed", 
                        "details": os.popen("make 2>&1").read()
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
<<<<<<< HEAD
                # Run benchmark
<<<<<<< HEAD
                process = subprocess.Popen(["./benchmark"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                test_results = []
                total_tests = 0
                test_count = 0
>>>>>>> parent of c05d800 (small change)

        def compile_benchmark():
            try:
                utils.run_nvcc_and_return_bytes(gpu, solution_code, "solution")
            except Exception:
                pass
        

        if language == "cuda":
            bench_thr = Thread(target=compile_benchmark)
            bench_thr.start()

            try:
                checker_compiled = utils.run_nvcc_and_return_bytes(gpu, solution_code, "checker")
            except utils.NVCCError as e:
                yield {
                    "status": "error",
                    "error": "Compilation failed",
                    "details": e.args[0],
                    "test_results": [],
                    "passed_tests": 0,
                    "total_tests": 0,
                }
                return

            bench_thr.join()
        else:
            checker_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen("checker", checker_compiled, solution_code, problem_name, problem_def, dtype, language)
        for event in stream:
            yield event

    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")


@web_app.post("/benchmark-{gpu}")
async def benchmark(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return 404

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    dtype = req["dtype"]

    language = req["language"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        yield {"status": "compiling"}

        if language == "cuda":
            try:
                benchmark_compiled = utils.run_nvcc_and_return_bytes(gpu, solution_code, "benchmark")
            except utils.NVCCError as e:
                yield { 
                    "status": "error",
                    "error": "Compilation failed",
                    "details": e.args[0],
                }
                return
        else:
            benchmark_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen("benchmark", benchmark_compiled, solution_code, problem_name, problem_def, dtype, language)
        for event in stream:
            yield event
    
    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
=======
=======
                import subprocess
>>>>>>> parent of 24b4de1 (refactored for duplicate code, now can make changes to the testing and benchmark wihtout having to change it everywhere)
                result = subprocess.run(["./benchmark"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr
                    }) + "\n\n"
                    return
                
                try:
                    lines = result.stdout.strip().split('\n')
                    test_results = []
                    
                    for line in lines[:-1]:
                        test_id, name, runtime_ms, gflops = line.split(',')
                        test_result = {
                            "test_id": int(test_id),
                            "name": name,
                            "runtime_ms": float(runtime_ms),
                            "gflops": float(gflops)
                        }
                        test_results.append(test_result)
                        yield "data: " + json.dumps({
                            "status": "test_result",
                            "result": test_result
                        }) + "\n\n"
                    
                    avg_gflops = float(lines[-1])
                    
                    yield "data: " + json.dumps({
                        "status": "success",
                        "test_results": test_results,
                        "average_gflops": avg_gflops
                    }) + "\n\n"
                    
                except Exception as e:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Failed to parse benchmark output",
                        "details": str(e)
                    }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e)
            }) + "\n\n"

    return StreamingResponse(generate_benchmark_results(), media_type="text/event-stream")

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def checker_h100(item: dict):
    async def generate_checker_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                tmpdir_path = Path(tmpdir)
                
                checker_dir = tmpdir_path / "checker"
                checker_dir.mkdir()
                os.system(f"cp /root/checker/core.hpp /root/checker/Makefile {str(checker_dir)}")
                
                solution_path = checker_dir / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = checker_dir / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                reference_path = checker_dir / "reference.cu"
                reference_path.write_text(item["reference_code"])
                
                os.system(f"cp /root/checker.cu {str(checker_dir)}")
                
                os.chdir(checker_dir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed",
                        "details": os.popen("make 2>&1").read(),
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                import subprocess
                result = subprocess.run(["./checker"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr,
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                lines = result.stdout.strip().split('\n')
                test_results = []
                passed_tests = 0
                
                for line in lines[:-1]:
                    test_id, name, status = line.split(',')
                    test_result = {
                        "test_id": int(test_id),
                        "name": name,
                        "status": status.strip()
                    }
                    test_results.append(test_result)
                    if status.strip() == "PASSED":
                        passed_tests += 1
                    yield "data: " + json.dumps({
                        "status": "test_result",
                        "result": test_result
                    }) + "\n\n"
                
                overall_status = lines[-1].strip()
                total_tests = len(test_results)
                
                yield "data: " + json.dumps({
                    "status": "complete",
                    "passed": overall_status == "PASSED",
                    "test_results": test_results,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests
                }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e),
                "test_results": []
            }) + "\n\n"

    return StreamingResponse(generate_checker_results(), media_type="text/event-stream")


@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def benchmark_a100_80gb(item: dict):
    async def generate_benchmark_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                solution_path = Path(tmpdir) / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = Path(tmpdir) / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                os.system("cp /root/benchmark.cu /root/core.hpp /root/Makefile " + tmpdir)            

                os.chdir(tmpdir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed", 
                        "details": os.popen("make 2>&1").read()
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                import subprocess
                result = subprocess.run(["./benchmark"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr
                    }) + "\n\n"
                    return
                
                try:
                    lines = result.stdout.strip().split('\n')
                    test_results = []
                    
                    for line in lines[:-1]:
                        test_id, name, runtime_ms, gflops = line.split(',')
                        test_result = {
                            "test_id": int(test_id),
                            "name": name,
                            "runtime_ms": float(runtime_ms),
                            "gflops": float(gflops)
                        }
                        test_results.append(test_result)
                        yield "data: " + json.dumps({
                            "status": "test_result",
                            "result": test_result
                        }) + "\n\n"
                    
                    avg_gflops = float(lines[-1])
                    
                    yield "data: " + json.dumps({
                        "status": "success",
                        "test_results": test_results,
                        "average_gflops": avg_gflops
                    }) + "\n\n"
                    
                except Exception as e:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Failed to parse benchmark output",
                        "details": str(e)
                    }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e)
            }) + "\n\n"

    return StreamingResponse(generate_benchmark_results(), media_type="text/event-stream")

@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def checker_a100_80gb(item: dict):
    async def generate_checker_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                tmpdir_path = Path(tmpdir)
                
                checker_dir = tmpdir_path / "checker"
                checker_dir.mkdir()
                os.system(f"cp /root/checker/core.hpp /root/checker/Makefile {str(checker_dir)}")
                
                solution_path = checker_dir / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = checker_dir / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                reference_path = checker_dir / "reference.cu"
                reference_path.write_text(item["reference_code"])
                
                os.system(f"cp /root/checker.cu {str(checker_dir)}")
                
                os.chdir(checker_dir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed",
                        "details": os.popen("make 2>&1").read(),
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                import subprocess
                result = subprocess.run(["./checker"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr,
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                lines = result.stdout.strip().split('\n')
                test_results = []
                passed_tests = 0
                
                for line in lines[:-1]:
                    test_id, name, status = line.split(',')
                    test_result = {
                        "test_id": int(test_id),
                        "name": name,
                        "status": status.strip()
                    }
                    test_results.append(test_result)
                    if status.strip() == "PASSED":
                        passed_tests += 1
                    yield "data: " + json.dumps({
                        "status": "test_result",
                        "result": test_result
                    }) + "\n\n"
                
                overall_status = lines[-1].strip()
                total_tests = len(test_results)
                
                yield "data: " + json.dumps({
                    "status": "complete",
                    "passed": overall_status == "PASSED",
                    "test_results": test_results,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests
                }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e),
                "test_results": []
            }) + "\n\n"

    return StreamingResponse(generate_checker_results(), media_type="text/event-stream")

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def benchmark_a10g(item: dict):
<<<<<<< HEAD
    return await generic_benchmark(item)
>>>>>>> parent of bdd95c4 (update how the test case data is sent to render the total cases correctly, revamped api.py completely)
=======
    async def generate_benchmark_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                solution_path = Path(tmpdir) / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = Path(tmpdir) / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                os.system("cp /root/benchmark.cu /root/core.hpp /root/Makefile " + tmpdir)            

                os.chdir(tmpdir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed", 
                        "details": os.popen("make 2>&1").read()
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                import subprocess
                result = subprocess.run(["./benchmark"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr
                    }) + "\n\n"
                    return
                
                try:
                    lines = result.stdout.strip().split('\n')
                    test_results = []
                    
                    for line in lines[:-1]:
                        test_id, name, runtime_ms, gflops = line.split(',')
                        test_result = {
                            "test_id": int(test_id),
                            "name": name,
                            "runtime_ms": float(runtime_ms),
                            "gflops": float(gflops)
                        }
                        test_results.append(test_result)
                        yield "data: " + json.dumps({
                            "status": "test_result",
                            "result": test_result
                        }) + "\n\n"
                    
                    avg_gflops = float(lines[-1])
                    
                    yield "data: " + json.dumps({
                        "status": "success",
                        "test_results": test_results,
                        "average_gflops": avg_gflops
                    }) + "\n\n"
                    
                except Exception as e:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Failed to parse benchmark output",
                        "details": str(e)
                    }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e)
            }) + "\n\n"

    return StreamingResponse(generate_benchmark_results(), media_type="text/event-stream")

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def checker_a10g(item: dict):
    async def generate_checker_results():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                tmpdir_path = Path(tmpdir)
                
                checker_dir = tmpdir_path / "checker"
                checker_dir.mkdir()
                os.system(f"cp /root/checker/core.hpp /root/checker/Makefile {str(checker_dir)}")
                
                solution_path = checker_dir / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = checker_dir / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                reference_path = checker_dir / "reference.cu"
                reference_path.write_text(item["reference_code"])
                
                os.system(f"cp /root/checker.cu {str(checker_dir)}")
                
                os.chdir(checker_dir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed",
                        "details": os.popen("make 2>&1").read(),
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                import subprocess
                result = subprocess.run(["./checker"], capture_output=True, text=True)
                
                if result.stderr:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr,
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                lines = result.stdout.strip().split('\n')
                test_results = []
                passed_tests = 0
                
                for line in lines[:-1]:
                    test_id, name, status = line.split(',')
                    test_result = {
                        "test_id": int(test_id),
                        "name": name,
                        "status": status.strip()
                    }
                    test_results.append(test_result)
                    if status.strip() == "PASSED":
                        passed_tests += 1
                    yield "data: " + json.dumps({
                        "status": "test_result",
                        "result": test_result
                    }) + "\n\n"
                
                overall_status = lines[-1].strip()
                total_tests = len(test_results)
                
                yield "data: " + json.dumps({
                    "status": "complete",
                    "passed": overall_status == "PASSED",
                    "test_results": test_results,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests
                }) + "\n\n"
                
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e),
                "test_results": []
            }) + "\n\n"

    return StreamingResponse(generate_checker_results(), media_type="text/event-stream")
>>>>>>> parent of 24b4de1 (refactored for duplicate code, now can make changes to the testing and benchmark wihtout having to change it everywhere)
