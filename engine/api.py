import os
import tempfile
import subprocess
from pathlib import Path

import asyncio
import threading

import json
import modal
from fastapi.responses import StreamingResponse


GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90",
    "A100-80GB": "80",
    "A10G": "86",
}

SKELETON_DIR = Path(__file__).parent / "skeleton"
SKELETON_FILES = ["benchmark.cu", "checker.cu", "core.hpp"]

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
image = modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.11")
for path in SKELETON_FILES:
    image = image.add_local_file(SKELETON_DIR / path, "/skeleton/" + path)

app = modal.App("tensara", image=image)


class NVCCError(Exception):
    pass


def nvcc_command(gpu: str, srcs: list[Path | str], out: Path | str):
    """Get nvcc command for the given GPU, source files, and output file"""

    srcs = map(str, srcs)
    out = str(out)

    sm = GPU_COMPUTE_CAPABILITIES[gpu]
    cmd = ["nvcc", "-O2", f"-arch=compute_{sm}", f"-code=sm_{sm}", "-o", out] + srcs

    return cmd


def run_nvcc(gpu: str, files: dict[str, str], binary_name: str) -> Path:
    """Compile checker code

    Args:
        gpu (str): GPU type
        files (dict[str, str]): Code files (file name -> content)
        binary_name (str): Binary name ("checker" or "benchmark")

    Returns:
        Path: Path to the compiled binary

    Raises:
        ValueError: If the binary name is not "checker" or "benchmark"
        NVCCError: If compilation fails
    """

    binary_sources = {
        "checker": ["checker.cu"],
        "benchmark": ["benchmark.cu"],
    }

    if binary_name not in binary_sources:
        raise ValueError(f"Unknown binary name: {binary_name}")

    binary_file = tempfile.NamedTemporaryFile(delete=False)
    binary_file.close()

    out_path = Path(binary_file.name)
    out_path.unlink()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td)

        # symlink skeleton files
        for name in SKELETON_FILES:
            os.symlink(f"/skeleton/{name}", path / name)

        # write solution files
        for name, code in files.items():
            (path / name).write_text(code)

        # compile
        srcs = [path / src for src in binary_sources[binary_name]]
        nvcc = subprocess.Popen(
            nvcc_command(gpu, srcs, out_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        out, err = nvcc.communicate(timeout=60)
        if nvcc.returncode != 0:
            raise NVCCError(err)

    return out_path


def generic_checker(gpu: str, item: dict):
    """Common implementation for all checker endpoints."""

    try:
        yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"

        files = {
            "reference.cu": item["reference_code"],
            "solution.cu": item["solution_code"],
            "tests.hpp": item["tests_code"],
        }
        try:
            checker_path = run_nvcc(gpu, files, "checker")
        except NVCCError as e:
            obj = {
                "status": "error",
                "error": "Compilation failed",
                "details": e.args[0],
                "test_results": [],
                "passed_tests": 0,
                "total_tests": 0,
            }
            yield "data: " + json.dumps(obj) + "\n\n"
            return

        yield "data: " + json.dumps({"status": "running"}) + "\n\n"

        # Stream the output line by line
        process = subprocess.Popen(
            [str(checker_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        test_results = []
        passed_tests = 0
        total_tests = 0
        has_failed = False

        # Process each line as it comes in the stream
        for line in iter(process.stdout.readline, ""):
            if not line.strip():
                continue

            # This would indicate the end of the test results
            if line.strip() in ["PASSED", "FAILED"]:
                continue

            test_case, name, status = line.split(",")
            status = status.strip()
            test_id = int(test_case.split("/")[0])
            total_tests = int(test_case.split("/")[1])

            test_result = {"test_id": test_id, "name": name, "status": status}
            test_results.append(test_result)
            obj = {
                "status": "test_result",
                "result": test_result,
                "totalTests": total_tests,
            }
            yield "data: " + json.dumps(obj) + "\n\n"

            if status == "PASSED":
                passed_tests += 1
            else:
                has_failed = True

        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()

        stderr_output = process.stderr.read()
        if stderr_output:
            obj = {
                "status": "error",
                "error": "Runtime error",
                "details": stderr_output,
                "test_results": [],
                "passed_tests": 0,
                "total_tests": 0,
            }
            yield "data: " + json.dumps(obj) + "\n\n"
            return

        # Finally, at the very end, we can send the overall status
        obj = {
            "status": "complete",
            "passed": not has_failed,
            "test_results": test_results,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
        }
        yield "data: " + json.dumps(obj) + "\n\n"

    except Exception as e:
        obj = {"status": "error", "error": str(e), "test_results": []}
        yield "data: " + json.dumps(obj) + "\n\n"


def generic_benchmark(gpu: str, item: dict):
    """Common implementation for all benchmark endpoints."""

    try:
        yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"

        # compile
        files = {
            "solution.cu": item["solution_code"],
            "tests.hpp": item["tests_code"],
        }
        try:
            benchmark_path = run_nvcc(gpu, files, "benchmark")
        except NVCCError as e:
            obj = {"status": "error", "error": "Compilation failed", "details": e.args[0]}
            yield "data: " + json.dumps(obj) + "\n\n"
            return

        yield "data: " + json.dumps({"status": "running"}) + "\n\n"

        # Run benchmark
        process = subprocess.Popen(
            [str(benchmark_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        test_results = []
        test_count = 0
        avg_gflops = None

        # Process each line as it comes in the stream
        for line in iter(process.stdout.readline, ""):
            if not line.strip():
                continue

            # Check if it's the last line with average GFLOPS
            try:
                avg_gflops = float(line.strip())
                continue
            except ValueError:
                pass

            try:
                test_id, name, runtime_ms, gflops = line.split(",")
                test_result = {
                    "test_id": int(test_id),
                    "name": name,
                    "runtime_ms": float(runtime_ms),
                    "gflops": float(gflops),
                }
                obj = {
                    "status": "test_result",
                    "result": test_result,
                    "totalTests": test_count,
                }
                test_results.append(test_result)
                test_count += 1

                yield "data: " + json.dumps(obj) + "\n\n"
            except Exception as e:
                obj = {
                    "status": "error",
                    "error": "Failed to parse benchmark line",
                    "details": str(e),
                    "line": line,
                }
                yield "data: " + json.dumps(obj) + "\n\n"

        stderr_output = process.stderr.read()

        if stderr_output:
            obj = {"status": "error", "error": "Runtime error", "details": stderr_output}
            yield "data: " + json.dumps(obj) + "\n\n"
            return

        # Finally, at the very end, we can send the overall status
        # Calculate average GFLOPS if not already calculated
        if not avg_gflops:
            avg_gflops = (
                sum(result["gflops"] for result in test_results) / len(test_results)
                if test_results
                else 0
            )

        obj = {
            "status": "success",
            "test_results": test_results,
            "average_gflops": avg_gflops,
            "total_tests": test_count,
        }
        yield "data: " + json.dumps(obj) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"status": "error", "error": str(e)}) + "\n\n"


def async_wrap_iter(it):
    """Wrap a sync iterator to make it async"""
    loop = asyncio.get_event_loop()
    exc = None
    END_MARKER = object()
    q = asyncio.Queue(1)

    def iter_to_queue():
        try:
            for item in it:
                pass
        except Exception as e:
            nonlocal exc
            exc = e
        finally:
            asyncio.run_coroutine_threadsafe(q.put(END_MARKER), loop).result()

    async def yield_items():
        while True:
            next_item = await q.get()
            if next_item is END_MARKER:
                break
            yield next_item
        if exc:
            raise exc

    threading.Thread(target=iter_to_queue).start()
    return yield_items()


def make_endpoints(gpu):
    @app.function(gpu=gpu)
    @modal.web_endpoint(method="POST")
    async def checker(item: dict):
        stream = async_wrap_iter(generic_checker(gpu, item))
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.function(gpu=gpu)
    @modal.web_endpoint(method="POST")
    async def benchmark(item: dict):
        stream = async_wrap_iter(generic_benchmark(gpu, item))
        return StreamingResponse(stream, media_type="text/event-stream")


# create endpoints for each GPU
for gpu in GPU_COMPUTE_CAPABILITIES.keys():
    make_endpoints(gpu)
