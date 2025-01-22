import modal
import tempfile
import os
from pathlib import Path

# Create image with required dependencies
stub_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install([
        "build-essential",
        "make"
    ])
    .pip_install(["fastapi", "uvicorn"])
    .copy_local_file(stub_dir / "benchmark.cu", "/root/benchmark.cu")
    .copy_local_file(stub_dir / "Makefile", "/root/Makefile")
)

# Create Modal app with the image
app = modal.App("cudaforces", image=image)

@app.function(gpu="any")
@modal.web_endpoint(method="POST")
def benchmark_solution(item: dict):
    """
    Accepts CUDA solution code and runs benchmarks
    Expected input format: {"code": "your CUDA code here"}
    """
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
            
            return {
                "status": "success",
                "benchmark_results": result.stdout,
                "errors": result.stderr if result.stderr else None
            }
            
    except Exception as e:
        return {"error": str(e)}
