import modal
from pathlib import Path
import utils
import runner

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
RUNTIME_IMAGE_NAME = "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
CURR_DIR = Path(__file__).parent

PIP_PACKAGES = ["torch", "numpy", "fastapi[standard]"]
LOCAL_SOURCE = ["problems", "utils", "runner", "problem"]

devel_image = (
    modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.11")
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_SOURCE)
)

runtime_image = (
    modal.Image.from_registry(RUNTIME_IMAGE_NAME, add_python="3.11")
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_SOURCE)
)

app = modal.App("tensara-python-engine", image=devel_image)

# Generic checker function that contains the common logic
async def generic_checker(item: dict):
    """Generic function for checking CUDA solutions on GPU"""
    solution_code = item["solution_code"]
    problem_name = item["problem"]
     # Use the GPU type from the request if provided, otherwise it will default
    # to the smallest GPU (T4)
    gpu = item.get("gpu") or "T4"

    problem = utils.load_problem_module(problem_name)

    def json_stream():
        for result in runner.run_checker(problem, solution_code, gpu=gpu):
            yield utils.format_for_sse(result)

    return utils.create_streaming_response(json_stream)

async def generic_benchmark(item: dict):
    """Generic function for benchmarking CUDA solutions on GPU"""
    solution_code = item["solution_code"]
    problem_name = item["problem"]
    # Use the GPU type from the request if provided, otherwise it will default
    # to the smallest GPU (T4)
    gpu = item.get("gpu") or "T4"

    problem = utils.load_problem_module(problem_name)

    def json_stream():
        for result in runner.run_benchmark(problem, solution_code, gpu=gpu):
            yield utils.format_for_sse(result)

    return utils.create_streaming_response(json_stream)

# GPU-specific endpoints
@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def checker_t4(item: dict):
    return await generic_checker(item)

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def checker_a10g(item: dict):
    return await generic_checker(item)

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def checker_h100(item: dict):
    return await generic_checker(item)

@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def checker_a100_80gb(item: dict):
    return await generic_checker(item)

@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def benchmark_t4(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def benchmark_a10g(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def benchmark_h100(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def benchmark_a100_80gb(item: dict):
    return await generic_benchmark(item)
    
