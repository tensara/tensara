import modal
import tempfile
import os
from pathlib import Path

stub_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install([
        "build-essential",
        "make"
    ])
    .pip_install(["fastapi", "uvicorn"])
    .add_local_file(stub_dir / "benchmark/benchmark.cu", "/root/benchmark.cu")
    .add_local_file(stub_dir / "benchmark/Makefile", "/root/Makefile")
)

app = modal.App("tensara", image=image)

@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
def benchmark(item: dict):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            solution_path = Path(tmpdir) / "solution.cuh"
            solution_path.write_text(item["code"])
            
            os.system("cp /root/benchmark.cu /root/Makefile " + tmpdir)            

            os.chdir(tmpdir)
            compile_result = os.system("make 2>&1")
            if compile_result != 0:
                return {"error": "Compilation failed", "details": os.popen("make 2>&1").read()}
            
            import subprocess
            result = subprocess.run(["./benchmark"], capture_output=True, text=True)
            
            if result.stderr:
                return {"error": "Runtime error", "details": result.stderr}
            
            try:
                avg_runtime = float(result.stdout.strip())
                return {
                    "status": "success",
                    "average_runtime_ms": avg_runtime
                }
            except ValueError:
                return {"error": "Failed to parse benchmark output", "details": result.stdout}
            
    except Exception as e:
        return {"error": str(e)}

