import modal
import tempfile
import os
from pathlib import Path

# Define the Modal stub
stub = modal.Stub("cuda-benchmark")

# Create a GPU image with CUDA tools
cuda_image = (
    modal.Image.debian_slim()
    .apt_install(["nvidia-cuda-toolkit", "build-essential"])
    .pip_install(["fastapi", "uvicorn"])
)

# FastAPI app
from fastapi import FastAPI, HTTPException
app = FastAPI()

@stub.function(gpu="any", image=cuda_image)
@modal.web_endpoint(method="POST")
async def benchmark_solution(solution_code: str):
    """
    Accepts CUDA solution code and runs benchmarks
    """
    try:
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write solution code
            solution_path = Path(tmpdir) / "solution.cuh"
            solution_path.write_text(solution_code)
            
            # Write benchmark code
            benchmark_path = Path(tmpdir) / "benchmark.cu"
            benchmark_path.write_text(BENCHMARK_CODE)
            
            # Write Makefile
            makefile_path = Path(tmpdir) / "Makefile"
            makefile_path.write_text(MAKEFILE)
            
            # Compile and run
            os.chdir(tmpdir)
            compile_result = os.system("make")
            if compile_result != 0:
                raise HTTPException(status_code=400, detail="Compilation failed")
            
            # Run benchmark and capture output
            import subprocess
            result = subprocess.run(["./benchmark"], capture_output=True, text=True)
            
            return {
                "status": "success",
                "compile_status": "success",
                "benchmark_results": result.stdout,
                "errors": result.stderr if result.stderr else None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Benchmark code template
BENCHMARK_CODE = """
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "solution.cuh"

class BenchmarkRunner {
    // ... (same as before)
};

int main() {
    std::vector<size_t> sizes = {1<<20, 1<<22, 1<<24, 1<<26};
    for (size_t size : sizes) {
        BenchmarkRunner runner(size);
        runner.run_benchmark();
    }
    return 0;
}
"""

# Makefile template
MAKEFILE = """
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75

TARGET = benchmark
SRC = benchmark.cu
HEADERS = solution.cuh

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

.PHONY: clean run
clean:
	rm -f $(TARGET)
run: $(TARGET)
	./$(TARGET)
"""

if __name__ == "__main__":
    stub.serve()
