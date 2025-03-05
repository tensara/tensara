import modal

import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from util import GPU_COMPUTE_CAPABILITIES, SKELETON_FILES
from util import into_async, run_nvcc_bytes
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


def generic_checker(binary: bytes):
    print("running checker, binary size", len(binary))
    for event in run_checker(binary):
        print("event", event)
        yield "data: " + json.dumps(event, allow_nan=False) + "\n\n"


def generic_benchmark(binary: bytes):
    print("running benchmark, binary size", len(binary))
    for event in run_benchmark(binary):
        print("event", event)
        yield "data: " + json.dumps(event, allow_nan=False) + "\n\n"


gpu_checkers = {
    gpu: app.function(image=runtime_image, name=f"checker_{gpu}", gpu=gpu)(generic_checker)
    for gpu in GPU_COMPUTE_CAPABILITIES.keys()
}
gpu_benchmarks = {
    gpu: app.function(image=runtime_image, name=f"benchmark_{gpu}", gpu=gpu)(generic_benchmark)
    for gpu in GPU_COMPUTE_CAPABILITIES.keys()
}


for gpu in gpu_checkers:
    globals()[f"checker_{gpu}"] = gpu_checkers[gpu]

for gpu in gpu_benchmarks:
    globals()[f"benchmark_{gpu}"] = gpu_benchmarks[gpu]


@web_app.post("/checker-{gpu}")
async def checker(gpu: str, request: Request):
    req = await request.json()
    print("checker with gpu", gpu)
    if gpu not in gpu_checkers:
        return 404

    files = {
        "reference.cu": req["reference_code"],
        "solution.cu": req["solution_code"],
        "tests.hpp": req["tests_code"],
    }

    checker_compiled = await into_async(run_nvcc_bytes)(gpu, files, "checker")

    print("checker compiled")
    gpu_checker = gpu_checkers[gpu]
    stream = gpu_checker.remote_gen(checker_compiled)

    return StreamingResponse(stream, media_type="text/event-stream")


@web_app.post("/benchmark-{gpu}")
async def benchmark(gpu: str, request: Request):
    item = await request.json()
    print("benchmark with gpu", gpu)
    if gpu not in gpu_benchmarks:
        return 404

    files = {
        "solution.cu": item["solution_code"],
        "tests.hpp": item["tests_code"],
    }

    benchmark_compiled = await into_async(run_nvcc_bytes)(gpu, files, "benchmark")

    print("benchmark compiled")
    gpu_benchmark = gpu_benchmarks[gpu]
    stream = gpu_benchmark.remote_gen(benchmark_compiled)

    return StreamingResponse(stream, media_type="text/event-stream")


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
