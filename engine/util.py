import os
import tempfile
import subprocess
from pathlib import Path
from functools import lru_cache, wraps

import threading
import asyncio

SKELETON_FILES = ["benchmark.cu", "checker.cu", "core.hpp"]

GPU_COMPUTE_CAPABILITIES = {
    "T4": "75",
    "H100": "90",
    "A100-80GB": "80",
    "A10G": "86",
    "L40S": "89",
    "L4": "89"
}


class NVCCError(Exception):
    pass


def hash_dict(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    class HDict(dict):
        def __hash__(self):
            return hash(frozenset(self.items()))

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([HDict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: HDict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapped


def nvcc_command(gpu: str, srcs: list[Path | str], out: Path | str):
    """Get nvcc command for the given GPU, source files, and output file"""

    srcs = [str(src) for src in srcs]
    out = str(out)

    sm = GPU_COMPUTE_CAPABILITIES[gpu]
    cmd = ["nvcc", "-std=c++20", "-O2", f"-arch=compute_{sm}", f"-code=sm_{sm}", "-o", out] + srcs

    return cmd


@hash_dict
@lru_cache(maxsize=512)  # each binary is ~1MB, so 512MB cache
def run_nvcc_bytes(gpu: str, files: dict[str, str], binary_name: str) -> bytes:
    """Compile checker code

    Args:
        gpu (str): GPU type
        files (dict[str, str]): Code files (file name -> content)
        binary_name (str): Binary name ("checker" or "benchmark")

    Returns:
        bytes: Compiled binary

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

    data = out_path.read_bytes()
    out_path.unlink()

    return data


def async_wrap_iter(it):
    """
    Wrap blocking iterator into an asynchronous one

    From: https://stackoverflow.com/questions/62294385/synchronous-generator-in-asyncio
    """
    loop = asyncio.get_event_loop()
    q = asyncio.Queue(1)
    exception = None
    _END = object()

    async def yield_queue_items():
        while True:
            next_item = await q.get()
            if next_item is _END:
                break
            yield next_item
        if exception is not None:
            # the iterator has raised, propagate the exception
            raise exception

    def iter_to_queue():
        nonlocal exception
        try:
            for item in it:
                # This runs outside the event loop thread, so we
                # must use thread-safe API to talk to the queue.
                asyncio.run_coroutine_threadsafe(q.put(item), loop).result()
        except Exception as e:
            exception = e
        finally:
            asyncio.run_coroutine_threadsafe(q.put(_END), loop).result()

    threading.Thread(target=iter_to_queue).start()
    return yield_queue_items()
