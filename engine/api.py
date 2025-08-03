from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import modal
from pathlib import Path
import utils
import runner

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
RUNTIME_IMAGE_NAME = "nvidia/cuda:12.8.0-runtime-ubuntu22.04"
CURR_DIR = Path(__file__).parent


PIP_PACKAGES = ["torch", "numpy", "fastapi", "triton", "simplejson"]
UV_PREFIX = "uv pip install --system "
LOCAL_SOURCE = ["utils", "runner", "problem", "api"]
APT_PACKAGES = ["build-essential", "gcc", "g++", "curl"]

devel_image = (
    modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.11")
    .apt_install(APT_PACKAGES)
    .env({"CC": "gcc"})
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands(UV_PREFIX + " ".join(PIP_PACKAGES))
    .add_local_python_source(*LOCAL_SOURCE)
)


def build_runtime_image(gpu: str):
    if gpu == "B200":
        return (
            modal.Image.from_registry(RUNTIME_IMAGE_NAME, add_python="3.11")
            .apt_install(APT_PACKAGES + ["libedit-dev", "zlib1g-dev"])
            .env({"CC": "gcc"})
            .env({"PATH": "/root/.local/bin:$PATH"})
            .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
            .run_commands(UV_PREFIX + " ".join([p for p in PIP_PACKAGES if p != "torch"]))
            .run_commands(
                "uv pip install --system --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
            )
            .add_local_python_source(*LOCAL_SOURCE)
        )
    else:
        return (
            modal.Image.from_registry(RUNTIME_IMAGE_NAME, add_python="3.11")
            .apt_install(APT_PACKAGES + ["libedit-dev", "zlib1g-dev"])
            .env({"CC": "gcc"})
            .env({"PATH": "/root/.local/bin:$PATH"})
            .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
            .run_commands(UV_PREFIX + " ".join(PIP_PACKAGES))
            .run_commands(
                "uv pip install --system modular --extra-index-url https://dl.modular.com/public/nightly/python/simple/ --index-strategy unsafe-best-match"
            )
            .add_local_python_source(*LOCAL_SOURCE)
        )


app = modal.App("tensara", image=devel_image)
web_app = FastAPI()


@utils.subproc_generator(timeout=60)
def binary_runner(
    type: str,
    compiled_lib: bytes,
    solution_code: str,
    problem_name: str,
    problem_def: str,
    dtype: str,
    language: str,
):
    gen = None

    if language == "mojo" and compiled_lib is None:
        try:
            compiled_lib = utils.run_mojo_and_return_bytes(solution_code, type)
        except utils.MojoError as e:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Compilation Failed",
                "details": e.args[0],
            }
            return
    problem = utils.load_problem_module(problem_name, problem_def)
    solution_func = utils.make_solution_func(language, solution_code, compiled_lib, problem)

    if type == "sample":
        gen = runner.run_sample_case(problem_name, problem_def, solution_func, dtype, language)
    elif type == "checker":
        gen = runner.run_checker(problem_name, problem_def, solution_func, dtype, language)
    elif type == "benchmark":
        gen = runner.run_benchmark(problem_name, problem_def, solution_func, dtype, language)
    elif type == "sanity_check":
        gen = runner.run_sanity_check(problem_name, problem_def, solution_func, dtype, language)
    else:
        raise ValueError(f"Unknown binary type: {type}")

    for event in gen:
        yield event


gpu_runners = {
    gpu: app.function(
        image=build_runtime_image(gpu),
        name=f"runner_{gpu}",
        gpu=gpu,
        enable_memory_snapshot=False if gpu == "B200" else True,
        serialized=True,
    )(binary_runner)
    for gpu in utils.GPU_COMPUTE_CAPABILITIES.keys()
}

for gpu in gpu_runners:
    globals()[f"runner_{gpu}"] = gpu_runners[gpu]


def gen_wrapper(gen):
    import simplejson

    for event in gen:
        data = simplejson.dumps(event, ignore_nan=True)
        yield "data: " + data + "\n\n"


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
        yield {"status": "COMPILING"}

        def compile_benchmark():
            try:
                utils.run_nvcc_and_return_bytes(gpu, solution_code, "benchmark")
            except Exception:
                pass

        if language == "cuda":
            bench_thr = Thread(target=compile_benchmark)
            bench_thr.start()

            try:
                checker_compiled = utils.run_nvcc_and_return_bytes(gpu, solution_code, "checker")
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Compilation Failed",
                    "details": e.args[0],
                }
                return

            bench_thr.join()
        else:
            checker_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen(
            "checker", checker_compiled, solution_code, problem_name, problem_def, dtype, language
        )
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
        if language == "cuda":
            try:
                benchmark_compiled = utils.run_nvcc_and_return_bytes(
                    gpu, solution_code, "benchmark"
                )
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Compilation Failed",
                    "details": e.args[0],
                }
                return
        else:
            benchmark_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen(
            "benchmark",
            benchmark_compiled,
            solution_code,
            problem_name,
            problem_def,
            dtype,
            language,
        )
        for event in stream:
            yield event

    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")


@web_app.post("/sample-{gpu}")
async def sample_runner(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return 404

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    dtype = req["dtype"]
    language = req["language"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        yield {"status": "COMPILING"}

        if language == "cuda":
            try:
                sample_compiled = utils.run_nvcc_and_return_bytes(gpu, solution_code, "sample")
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Compilation Failed",
                    "details": e.args[0],
                }
                return
        else:
            sample_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen(
            "sample", sample_compiled, solution_code, problem_name, problem_def, dtype, language
        )
        for event in stream:
            yield event

    return StreamingResponse(gen_wrapper(create_stream()), media_type="text/event-stream")


@web_app.post("/benchmark_cli-{gpu}")
async def benchmark_cli(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return 404

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    dtype = req["dtype"]

    language = req["language"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        yield {"status": "COMPILING"}

        if language == "cuda":
            try:
                benchmark_compiled = utils.run_nvcc_and_return_bytes(
                    gpu, solution_code, "benchmark"
                )
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Compilation Failed",
                    "details": e.args[0],
                }
                return
        else:
            benchmark_compiled = None

        runner = gpu_runners[gpu]
        sanity_check_stream = runner.remote_gen(
            "sanity_check",
            benchmark_compiled,
            solution_code,
            problem_name,
            problem_def,
            dtype,
            language,
        )
        for event in sanity_check_stream:
            yield event
            if event["status"] == "SANITY_CHECK_PASSED":
                break
            elif (
                event["status"] == "RUNTIME_ERROR"
                or event["status"] == "ERROR"
                or event["status"] == "COMPILE_ERROR"
                or event["status"] == "WRONG_ANSWER"
            ):
                return

        stream = runner.remote_gen(
            "benchmark",
            benchmark_compiled,
            solution_code,
            problem_name,
            problem_def,
            dtype,
            language,
        )
        for event in stream:
            yield event

    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
