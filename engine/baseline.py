import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import modal
from pathlib import Path
import utils
import runner
import tempfile
import os
import importlib
import ctypes
from tinygrad.tensor import Tensor
from tinygrad.dtype import _from_torch_dtype
import torch

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
RUNTIME_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
CURR_DIR = Path(__file__).parent

PIP_PACKAGES = ["torch", "numpy", "fastapi[standard]", "tinygrad", "simplejson"]
LOCAL_SOURCE = ["utils", "runner", "problem", "baseline"]
APT_PACKAGES = ["build-essential", "gcc", "g++"]

devel_image = (
    modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.11")
    .apt_install(APT_PACKAGES)
    .env({"CC": "gcc"})
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_SOURCE)
)

runtime_image = (
    modal.Image.from_registry(RUNTIME_IMAGE_NAME, add_python="3.11")
    .apt_install(APT_PACKAGES)
    .env({"CC": "gcc"})
    .pip_install(PIP_PACKAGES)
    .add_local_python_source(*LOCAL_SOURCE)
)

app = modal.App("baselines", image=devel_image)
web_app = FastAPI()

def tinygrad_param_func(language, solution_func, input_tensors, actual_output, problem, test_case):
    tinygrad_inputs = []
    solution_func.reset()
    for tensor in input_tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = Tensor.from_blob(
                tensor.data_ptr(), 
                tensor.shape, 
                dtype=_from_torch_dtype(tensor.dtype), 
                device='CUDA'
            )
            tinygrad_inputs.append(tensor)
        else:
            tinygrad_inputs.append(tensor)

    tinygrad_output = Tensor.from_blob(
        actual_output.data_ptr(),
        actual_output.shape,
        dtype=_from_torch_dtype(actual_output.dtype),
        device='CUDA'
    )
    
    extra_params = problem.get_extra_params(test_case)
    return tinygrad_inputs + [tinygrad_output] + list(extra_params)

def baseline_runner(solution_code: str, problem_name: str, problem_def: str, dtype: str, baseline: str, check: bool = False):
    problem = utils.load_problem_module(problem_name, problem_def)

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "baseline_solution.py")
    
    with open(temp_path, 'w') as f:
        f.write(solution_code)

    spec = importlib.util.spec_from_file_location("baseline_solution", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    param_func = None
    if baseline == "tinygrad":
        solution_func = module.solution
        param_func = tinygrad_param_func
    elif baseline == "torch_compile":
        solution_func = torch.compile(module.solution)
    elif baseline == "torch_vanilla":
        solution_func = module.solution
    
    if check:
        gen = runner.run_checker(problem_name, problem_def, solution_func, dtype, "python", param_func=param_func)
    else:
        gen = runner.run_benchmark(problem_name, problem_def, solution_func, dtype, "python", param_func=param_func)
    last_event = None
    for event in gen:
        print(event)
        last_event = event
    yield last_event


gpu_runners = {
    gpu: app.function(
        image=runtime_image,
        name=f"runner_{gpu}",
        gpu=gpu,
        enable_memory_snapshot=True,
        serialized=True,
    )(baseline_runner)
    for gpu in utils.GPU_COMPUTE_CAPABILITIES.keys()
}

for gpu in gpu_runners:
    globals()[f"runner_{gpu}"] = gpu_runners[gpu]

def gen_wrapper(gen):
    for event in gen:
        data = simplejson.dumps(event, ignore_nan=True)
        yield "data: " + data + "\n\n"

async def baseline_handler(request: Request, baseline: str):
    req = await request.json()
    gpu = req["gpu"]
    if gpu not in gpu_runners:
        return 404
    
    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    dtype = req["dtype"]
    check = req.get("check", False)
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        runner = gpu_runners[gpu]
        stream = runner.remote_gen(solution_code, problem_name, problem_def, dtype, baseline, check)
        for event in stream:
            yield event
    
    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")

@web_app.post("/tinygrad")
async def tinygrad(request: Request):
    return await baseline_handler(request, "tinygrad")

@web_app.post("/torch_compile")
async def torch_compile(request: Request):
    return await baseline_handler(request, "torch_compile")

@web_app.post("/torch_vanilla")
async def torch_vanilla(request: Request):
    return await baseline_handler(request, "torch_vanilla")

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app