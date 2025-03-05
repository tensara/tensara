import os
import asyncio
import tempfile
import subprocess
from pathlib import Path
from functools import wraps

import modal

SKELETON_DIR = Path(__file__).parent / "skeleton"
SKELETON_FILES = ["benchmark.cu", "checker.cu", "core.hpp"]

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

GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90",
    "A100-80GB": "80",
    "A10G": "86",
}


class NVCCError(Exception):
    pass


def nvcc_command(gpu: str, srcs: list[Path | str], out: Path | str):
    """Get nvcc command for the given GPU, source files, and output file"""

    srcs = [str(src) for src in srcs]
    out = str(out)

    sm = GPU_COMPUTE_CAPABILITIES[gpu]
    cmd = ["nvcc", "-std=c++20", "-O2", f"-arch=compute_{sm}", f"-code=sm_{sm}", "-o", out] + srcs

    return cmd


def run_nvcc_bytes(gpu: str, files: dict[str, str], binary_name: str) -> bytes:
    path = run_nvcc(gpu, files, binary_name)
    data = path.read_bytes()
    path.unlink()

    return data


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


def into_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.to_thread(func, *args, **kwargs)

    return wrapper
