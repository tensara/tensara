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

        
@web_app.post("/checker-{gpu}")
async def checker(gpu: str, request: Request):
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

    stream = utils.async_wrap_iter(gen_wrapper(create_stream()))
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
    
    stream = utils.async_wrap_iter(gen_wrapper(create_stream()))
    return StreamingResponse(stream, media_type="text/event-stream")


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app