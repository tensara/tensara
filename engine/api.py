import modal

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from util import GPU_COMPUTE_CAPABILITIES, SKELETON_FILES
from util import run_nvcc_bytes, NVCCError, async_wrap_iter
from runner import run_checker, run_benchmark

SKELETON_DIR = Path(__file__).parent / "skeleton"

DEVEL_IMG_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
RUNTIME_IMG_NAME = "nvidia/cuda:12.8.0-runtime-ubuntu22.04"

PIP_PACKAGES = ["fastapi[standard]"]
LOCAL_PACKAGES = ["util", "runner"]

devel_image = (
    modal.Image.from_registry(DEVEL_IMG_NAME, add_python="3.11")
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_PACKAGES)
)

for path in SKELETON_FILES:
    devel_image = devel_image.add_local_file(SKELETON_DIR / path, "/skeleton/" + path)

runtime_image = (
    modal.Image.from_registry(RUNTIME_IMG_NAME, add_python="3.11")
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_PACKAGES)
)


app = modal.App("tensara-public", image=devel_image)
web_app = FastAPI()


def gen_wrapper(gen):
    for event in gen:
        yield "data: " + json.dumps(event, allow_nan=False) + "\n\n"


"""
Explanation: modal gets mad if we do what we're about to do to a
function in a different module (file). Of course these need
to be generators because modal doesn't seem to look at the actual
return value of the functions to check if it was a generator??

Like simply doing return run_checker(compiled) doesn't work
"""


def my_run_checker(compiled):
    for event in run_checker(compiled):
        yield event


def my_run_benchmark(compiled):
    for event in run_benchmark(compiled):
        yield event


gpu_checkers = {
    gpu: app.function(image=runtime_image, name=f"checker_{gpu}", gpu=gpu)(my_run_checker)
    for gpu in GPU_COMPUTE_CAPABILITIES.keys()
}
gpu_benchmarks = {
    gpu: app.function(image=runtime_image, name=f"benchmark_{gpu}", gpu=gpu)(my_run_benchmark)
    for gpu in GPU_COMPUTE_CAPABILITIES.keys()
}

"""
Explanation: modal gets mad if we create a app.function but that
function with that name isn't a global of this module. so, we
loop through those functions and add them to the globals dict
"""


for gpu in gpu_checkers:
    globals()[f"checker_{gpu}"] = gpu_checkers[gpu]

for gpu in gpu_benchmarks:
    globals()[f"benchmark_{gpu}"] = gpu_benchmarks[gpu]


@web_app.post("/checker-{gpu}")
async def checker(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_checkers:
        return 404

    files = {
        "reference.cu": req["reference_code"],
        "solution.cu": req["solution_code"],
        "tests.hpp": req["tests_code"],
    }

    def my_stream():
        yield {"status": "compiling"}
        try:
            checker_compiled = run_nvcc_bytes(gpu, files, "checker")
        except NVCCError as e:
            yield {
                "status": "error",
                "error": "Compilation failed",
                "details": e.args[0],
                "test_results": [],
                "passed_tests": 0,
                "total_tests": 0,
            }
            return

        gpu_checker = gpu_checkers[gpu]
        stream = gpu_checker.remote_gen(checker_compiled)
        for event in stream:
            yield event

    stream = async_wrap_iter(gen_wrapper(my_stream()))
    return StreamingResponse(stream, media_type="text/event-stream")


@web_app.post("/benchmark-{gpu}")
async def benchmark(gpu: str, request: Request):
    item = await request.json()
    if gpu not in gpu_benchmarks:
        return 404

    files = {
        "solution.cu": item["solution_code"],
        "tests.hpp": item["tests_code"],
    }

    def my_stream():
        yield {"status": "compiling"}
        try:
            benchmark_compiled = run_nvcc_bytes(gpu, files, "benchmark")
        except NVCCError as e:
            yield {"status": "error", "error": "Compilation failed", "details": e.args[0]}
            return

        gpu_benchmark = gpu_benchmarks[gpu]
        stream = gpu_benchmark.remote_gen(benchmark_compiled)
        for event in stream:
            yield event

    stream = async_wrap_iter(gen_wrapper(my_stream()))
    return StreamingResponse(stream, media_type="text/event-stream")


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
