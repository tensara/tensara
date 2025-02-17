import modal
import tempfile
import os
from pathlib import Path

stub_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install([
        "build-essential",
        "make",
        "python3-dev",
        "python3-pip"
    ])
    .pip_install([
        "fastapi", 
        "uvicorn",
        "torch",
        "ninja"  # Required for PyTorch CUDA extensions
    ])
    .add_local_file(stub_dir / "benchmark/benchmark.cu", "/root/benchmark.cu")
    .add_local_file(stub_dir / "benchmark/Makefile", "/root/Makefile")
    .add_local_file(stub_dir / "checker/setup.py", "/root/checker_setup.py")
    .add_local_file(stub_dir / "checker/test.py", "/root/checker_test.py")
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


@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
def checker(item: dict):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            solution_cu = tmpdir_path / "solution.cu"
            cuda_bindings = tmpdir_path / "cuda_bindings.cpp"
            reference_py = tmpdir_path / "reference.py"
            
            solution_cu.write_text(item["solution_cu"])
            cuda_bindings.write_text(item["cuda_bindings"])
            reference_py.write_text(item["reference_py"])
            
            os.system(f"cp {stub_dir}/checker/setup.py {stub_dir}/checker/test.py " + tmpdir)
            
            os.chdir(tmpdir)
            build_result = os.system("python3 setup.py build_ext --inplace 2>&1")
            if build_result != 0:
                return {"error": "Compilation failed", "details": os.popen("python3 setup.py build_ext --inplace 2>&1").read()}
            
            import subprocess
            result = subprocess.run(
                ["python3", "test.py", "cuda_solution", "reference"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                return {"error": "Test failed", "details": result.stderr or result.stdout}
            
            return {
                "status": "success",
                "message": result.stdout.strip()
            }
            
    except Exception as e:
        return {"error": str(e)}

