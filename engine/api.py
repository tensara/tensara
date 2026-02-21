from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import modal
from pathlib import Path
import utils
import runner

DEVEL_IMAGE_NAME = "nvidia/cuda:13.1.0-devel-ubuntu24.04"
RUNTIME_IMAGE_NAME = "nvidia/cuda:13.1.0-runtime-ubuntu24.04"
CURR_DIR = Path(__file__).parent
MODULAR_INDEX = "https://modular.gateway.scarf.sh/simple/modular/modular-26.1.0-py3-none-any.whl"

PIP_PACKAGES = ["numpy", "fastapi", "triton", "simplejson", "nvidia-cutlass-dsl", "nvidia-ml-py"]
UV_PREFIX = "uv pip install --system "
LOCAL_SOURCE = ["utils", "runner", "problem", "api", "gpu_monitor"]
APT_PACKAGES = ["build-essential", "gcc", "g++", "curl"]
CUBLAS_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

devel_image = (
    modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.13")
    .apt_install(APT_PACKAGES)
    .env({"CC": "gcc"})
    .env({"PATH": "/root/.local/bin:$PATH"})
    .env(CUBLAS_ENV)
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands(UV_PREFIX + " ".join(PIP_PACKAGES))
    .run_commands("uv pip install --system torch==2.9.0")
    .add_local_python_source(*LOCAL_SOURCE)
)


runtime_image = (
    modal.Image.from_registry(RUNTIME_IMAGE_NAME, add_python="3.13")
    .apt_install(APT_PACKAGES + ["libedit-dev", "zlib1g-dev"])
    .env({"CC": "gcc"})
    .env({"PATH": "/root/.local/bin:$PATH"})
    .env(CUBLAS_ENV)
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands(UV_PREFIX + " ".join(PIP_PACKAGES))
    .run_commands(f"uv pip install --system mojo --extra-index-url {MODULAR_INDEX}")
    # install torch separately with CUDA 12.8
    .run_commands(
        "uv pip install --system torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128"
    )
    .add_local_python_source(*LOCAL_SOURCE)
)


def b200_image():
    return (
        modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.13")
        .apt_install(APT_PACKAGES + ["libedit-dev", "zlib1g-dev"])
        .env({"CC": "gcc"})
        .env({"PATH": "/root/.local/bin:$PATH"})
        .env(CUBLAS_ENV)
        .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
        .run_commands(UV_PREFIX + " ".join(PIP_PACKAGES + ["cuda-tile", "cupy-cuda13x"]))
        .run_commands(f"uv pip install --system mojo --extra-index-url {MODULAR_INDEX}")
        # install torch separately with CUDA 12.8
        .run_commands(
            "uv pip install --system torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128"
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
    language: str,
):
    gen = None

    if language == "mojo" and compiled_lib is None:
        if type != "sandbox":
            try:
                problem = utils.load_problem_module(problem_name, problem_def)
                expected = len(problem.get_function_signature().get("argtypes") or [])
                utils.validate_mojo_solution_signature_from_source(solution_code, expected)
            except utils.SolutionSignatureError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Invalid `solution` signature",
                    "details": str(e),
                }
                return
        try:
            if type == "sandbox":
                compiled_lib = utils.run_mojo_and_return_executable(solution_code, type)
            else:
                compiled_lib = utils.run_mojo_and_return_bytes(solution_code, type)
        except utils.MojoError as e:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Compilation Failed",
                "details": e.args[0],
            }
            return

    if type == "sandbox":
        gen = runner.run_sandbox(compiled_lib, solution_code)
    elif type == "sample":
        gen = runner.run_sample_case(
            problem_name, problem_def, solution_code, compiled_lib, language
        )
    else:
        try:
            problem = utils.load_problem_module(problem_name, problem_def)
            solution_func = utils.make_solution_func(language, solution_code, compiled_lib, problem)
        except utils.SolutionSignatureError as e:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Invalid `solution` signature",
                "details": str(e),
            }
            return
        except Exception as e:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Compilation Failed",
                "details": str(e),
            }
            return


          if type == "sample":
            # this should not be reached
            raise ValueError("This code path should not be reached")
        elif type == "checker":
            gen = runner.run_checker(problem_name, problem_def, solution_func, language)
        elif type == "benchmark":
            gen = runner.run_benchmark(problem_name, problem_def, solution_func, language)
        elif type == "sanity_check":
            gen = runner.run_sanity_check(problem_name, problem_def, solution_func, language)
        else:
            raise ValueError(f"Unknown binary type: {type}")

    for event in gen:
        yield event


gpu_runners = {
    gpu: app.function(
        image=b200_image() if gpu == "B200" else runtime_image,
        name=f"runner_{gpu}",
        gpu=gpu + "!" if gpu == "H100" else gpu,
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
        if event is None or event == {}:
            continue
        data = simplejson.dumps(event, ignore_nan=True)
        yield "data: " + data + "\n\n"


@web_app.post("/checker-{gpu}")
async def checker(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return JSONResponse(status_code=404, content={"error": f"GPU '{gpu}' not supported"})

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
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
            try:
                problem = utils.load_problem_module(problem_name, problem_def)
                expected = len(problem.get_function_signature().get("argtypes") or [])
                utils.validate_cuda_solution_signature_from_source(solution_code, expected)
            except utils.SolutionSignatureError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Invalid `solution` signature",
                    "details": str(e),
                }
                return
            bench_thr = Thread(target=compile_benchmark)
            bench_thr.start()

            try:
                checker_compiled = utils.run_nvcc_and_return_bytes(gpu, solution_code, "checker")
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "NVCC Compilation Failed",
                    "details": e.args[0],
                }
                return
            except Exception as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Unexpected Compilation Error",
                    "details": str(e),
                }
                return

            bench_thr.join()

            for event in utils.yield_ptx_sass(gpu, solution_code):
                yield event
        else:
            checker_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen(
            "checker", checker_compiled, solution_code, problem_name, problem_def, language
        )
        for event in stream:
            yield event

    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")


@web_app.post("/benchmark-{gpu}")
async def benchmark(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return JSONResponse(status_code=404, content={"error": f"GPU '{gpu}' not supported"})

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    language = req["language"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        if language == "cuda":
            try:
                problem = utils.load_problem_module(problem_name, problem_def)
                expected = len(problem.get_function_signature().get("argtypes") or [])
                utils.validate_cuda_solution_signature_from_source(solution_code, expected)
            except utils.SolutionSignatureError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Invalid `solution` signature",
                    "details": str(e),
                }
                return
            try:
                benchmark_compiled = utils.run_nvcc_and_return_bytes(
                    gpu, solution_code, "benchmark"
                )
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "NVCC Compilation Failed",
                    "details": e.args[0],
                }
                return
            except Exception as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Unexpected Compilation Error",
                    "details": str(e),
                }
                return

            for event in utils.yield_ptx_sass(gpu, solution_code):
                yield event
        else:
            benchmark_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen(
            "benchmark",
            benchmark_compiled,
            solution_code,
            problem_name,
            problem_def,
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
        return JSONResponse(status_code=404, content={"error": f"GPU '{gpu}' not supported"})

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    language = req["language"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        yield {"status": "COMPILING"}

        if language == "cuda":
            try:
                problem = utils.load_problem_module(problem_name, problem_def)
                expected = len(problem.get_function_signature().get("argtypes") or [])
                utils.validate_cuda_solution_signature_from_source(solution_code, expected)
            except utils.SolutionSignatureError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Invalid `solution` signature",
                    "details": str(e),
                }
                return
            try:
                sample_compiled = utils.run_nvcc_and_return_bytes(gpu, solution_code, "sample")
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "NVCC Compilation Failed",
                    "details": e.args[0],
                }
                return
            except Exception as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Unexpected Compilation Error",
                    "details": str(e),
                }
                return

            for event in utils.yield_ptx_sass(gpu, solution_code):
                yield event
        else:
            sample_compiled = None

        runner = gpu_runners[gpu]
        stream = runner.remote_gen(
            "sample", sample_compiled, solution_code, problem_name, problem_def, language
        )
        for event in stream:
            yield event

    return StreamingResponse(gen_wrapper(create_stream()), media_type="text/event-stream")


@web_app.post("/sandbox-{gpu}")
async def sandbox(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return JSONResponse(status_code=404, content={"error": f"GPU '{gpu}' not supported"})

    solution_code = req["code"]
    language = req.get("language", "cuda")

    def create_stream():
        yield {"status": "COMPILING"}

        try:
            if language == "cuda":
                compiled_lib = utils.run_nvcc_and_return_executable(gpu, solution_code)
                for event in utils.yield_ptx_sass(gpu, solution_code):
                    yield event
            elif language == "mojo":
                # Attempt a local compile if Mojo is available; if not, fall back
                # to letting the remote worker perform compilation by passing
                # `compiled_lib=None` (binary_runner will compile inside the
                # modal runtime when it sees a Mojo submission with no compiled bytes).
                try:
                    compiled_lib = utils.run_mojo_and_return_executable(solution_code, "sandbox")
                except utils.MojoError:
                    compiled_lib = None
            else:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Compilation Failed",
                    "details": f"Unsupported language for sandbox: {language}",
                }
                return
        except utils.NVCCError as e:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Compilation Failed",
                "details": e.args[0],
            }
            return
        except utils.MojoError as e:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Compilation Failed",
                "details": e.args[0],
            }
            return
        except Exception as e:
            yield {
                "status": "COMPILE_ERROR",
                "message": "Unexpected Compilation Error",
                "details": str(e),
            }
            return

        runner = gpu_runners[gpu]
        stream = runner.remote_gen(
            "sandbox", compiled_lib, solution_code, "sandbox", "sandbox", language
        )
        for event in stream:
            if not event:
                continue
            status = event.get("status") if isinstance(event, dict) else None
            if status == "TIME_LIMIT_EXCEEDED":
                yield {
                    "status": "SANDBOX_TIMEOUT",
                    "message": event.get("message", "Sandbox time limit exceeded"),
                    "details": event.get("details", ""),
                }
                return
            yield event

    stream = gen_wrapper(create_stream())
    return StreamingResponse(stream, media_type="text/event-stream")


@web_app.post("/benchmark_cli-{gpu}")
async def benchmark_cli(gpu: str, request: Request):
    req = await request.json()
    if gpu not in gpu_runners:
        return JSONResponse(status_code=404, content={"error": f"GPU '{gpu}' not supported"})

    solution_code = req["solution_code"]
    problem_def = req["problem_def"]
    language = req["language"]
    problem_name = utils.convert_slug_to_module_name(req["problem"])

    def create_stream():
        yield {"status": "COMPILING"}

        if language == "cuda":
            try:
                problem = utils.load_problem_module(problem_name, problem_def)
                expected = len(problem.get_function_signature().get("argtypes") or [])
                utils.validate_cuda_solution_signature_from_source(solution_code, expected)
            except utils.SolutionSignatureError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Invalid `solution` signature",
                    "details": str(e),
                }
                return
            try:
                benchmark_compiled = utils.run_nvcc_and_return_bytes(
                    gpu, solution_code, "benchmark"
                )
            except utils.NVCCError as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "NVCC Compilation Failed",
                    "details": e.args[0],
                }
                return
            except Exception as e:
                yield {
                    "status": "COMPILE_ERROR",
                    "message": "Unexpected Compilation Error",
                    "details": str(e),
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
