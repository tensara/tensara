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

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
RUNTIME_IMAGE_NAME = "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
CURR_DIR = Path(__file__).parent

PIP_PACKAGES = ["torch", "numpy", "fastapi[standard]", "tinygrad"]
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

@utils.subproc_generator(timeout=60)
def tinygrad_runner(solution_code: str, problem_name: str, problem_def: str, dtype: str):
    problem = utils.load_problem_module(problem_name, problem_def)

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "tinygrad_baseline.py")
    
    # This is needed because @jit has to read the source code
    with open(temp_path, 'w') as f:
        f.write(solution_code)
        
    spec = importlib.util.spec_from_file_location("tinygrad_baseline", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    solution_func = module.solution
    gen = runner.run_benchmark(problem_name, problem_def, solution_func, dtype, "python")
    for event in gen:
        yield event

gpu_runners = {
    gpu: app.function(
        image=runtime_image,
        name=f"tinygrad_runner_{gpu}",
        gpu=gpu,
        enable_memory_snapshot=True,
        serialized=True,
    )(tinygrad_runner)
    for gpu in utils.GPU_COMPUTE_CAPABILITIES.keys()
}

for gpu in gpu_runners:
    globals()[f"tinygrad_runner_{gpu}"] = gpu_runners[gpu]

def gen_wrapper(gen):
    for event in gen:
        yield "data: " + json.dumps(event) + "\n\n"

@web_app.post("/tinygrad")
async def tinygrad(request: Request):
    req = await request.json()
    gpu = req["gpu"]
    if gpu not in gpu_runners:
        return 404

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    dtype = req["dtype"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        runner = gpu_runners[gpu]
        stream = runner.remote_gen(solution_code, problem_name, problem_def, dtype)
        for event in stream:
            yield event
    
    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
