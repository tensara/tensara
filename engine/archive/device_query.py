import modal
import subprocess
import os

DEVEL_IMG_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"

PIP_PACKAGES = ["torch", "numpy", "fastapi[standard]", "triton"]

image = (
    modal.Image.from_registry(DEVEL_IMG_NAME, add_python="3.11")
    .apt_install(["git", "cmake", "build-essential", "gcc", "g++"])
    .env({"CC": "gcc"})
    .pip_install(PIP_PACKAGES)
)

app = modal.App("cuda-device-query", image=image)

@app.function(gpu="A100-80GB", image=image)
def run_device_query():
    subprocess.run(["git", "clone", "https://github.com/NVIDIA/cuda-samples.git"], check=True)
    
    os.chdir("cuda-samples/Samples/1_Utilities/deviceQuery")
    
    os.makedirs("build", exist_ok=True)
    os.chdir("build")

    subprocess.run(["cmake", ".."], check=True)
    subprocess.run(["make"], check=True)
    
    print("\n=== Running deviceQuery ===\n")
    result = subprocess.run(["./deviceQuery"],
                          capture_output=True, text=True, check=True)
    
    print(result.stdout)
    return result.stdout

@app.local_entrypoint()
def main():
    run_device_query.remote()