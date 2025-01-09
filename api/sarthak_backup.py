import modal
from modal import Image, App, web_endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Define request model
class SolutionRequest(BaseModel):
    solution_code: str

# Define the Modal app
app = App("cuda-benchmark")

# Create a GPU image with CUDA tools
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install([
        "build-essential",
        "make"
    ])
    .pip_install(["fastapi", "uvicorn"])
)

# FastAPI app
fastapi_app = FastAPI(title="CUDA Benchmark API")

@app.function(gpu="any", image=cuda_image)
@modal.web_endpoint(method="POST")
async def benchmark_solution(request: SolutionRequest):
    """
    Accepts CUDA solution code and runs benchmarks
    Returns benchmark results including runtime and correctness
    """
    try:
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write solution code
            solution_path = Path(tmpdir) / "solution.cuh"
            solution_path.write_text(request.solution_code)
            
            # Write benchmark code
            benchmark_path = Path(tmpdir) / "benchmark.cu"
            benchmark_path.write_text("""
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "solution.cuh"

class BenchmarkRunner {
private:
    cudaEvent_t start, stop;
    float *d_input1, *d_input2, *d_output;
    float *h_input1, *h_input2, *h_output;
    size_t size;
    size_t num_runs;

public:
    BenchmarkRunner(size_t n, size_t runs = 100) 
        : size(n), num_runs(runs) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaMalloc(&d_input1, size * sizeof(float));
        cudaMalloc(&d_input2, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));
        
        h_input1 = new float[size];
        h_input2 = new float[size];
        h_output = new float[size];
        
        for (size_t i = 0; i < size; i++) {
            h_input1[i] = static_cast<float>(i);
            h_input2[i] = static_cast<float>(i * 2);
        }
        
        cudaMemcpy(d_input1, h_input1, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, h_input2, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~BenchmarkRunner() {
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output);
        delete[] h_input1;
        delete[] h_input2;
        delete[] h_output;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    bool verify_result() {
        cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < std::min(size_t(5), size); i++) {
            float expected = h_input1[i] + h_input2[i];
            if (abs(h_output[i] - expected) > 1e-5) {
                std::cout << "Verification failed at " << i << ": "
                          << h_output[i] << " != " << expected << "\\n";
                return false;
            }
        }
        if (size > 5) {
            for (size_t i = size - 5; i < size; i++) {
                float expected = h_input1[i] + h_input2[i];
                if (abs(h_output[i] - expected) > 1e-5) {
                    std::cout << "Verification failed at " << i << ": "
                              << h_output[i] << " != " << expected << "\\n";
                    return false;
                }
            }
        }
        return true;
    }

    void run_benchmark() {
        float total_ms = 0.0f;
        float min_ms = 1e9;
        float max_ms = 0.0f;

        solution(d_input1, d_input2, d_output, size);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < num_runs; i++) {
            cudaEventRecord(start);
            solution(d_input1, d_input2, d_output, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }

        bool correct = verify_result();

        float avg_ms = total_ms / num_runs;
        float gb_per_sec = (size * 3 * sizeof(float)) / (avg_ms * 1e-3) / 1e9;
        
        std::cout << "\\nBenchmark Results for size " << size << ":\\n";
        std::cout << "----------------------------------------\\n";
        std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Average Runtime: " << avg_ms << " ms\\n";
        std::cout << "Min Runtime: " << min_ms << " ms\\n";
        std::cout << "Max Runtime: " << max_ms << " ms\\n";
        std::cout << "Memory Bandwidth: " << gb_per_sec << " GB/s\\n";
        std::cout << "Throughput: " << (size / (avg_ms * 1e-3)) / 1e9 
                  << " billion elements/second\\n";
    }
};

int main() {
    std::vector<size_t> sizes = {1<<20, 1<<22, 1<<24, 1<<26};
    
    for (size_t size : sizes) {
        BenchmarkRunner runner(size);
        runner.run_benchmark();
    }
    
    return 0;
}
""")
            
            # Write Makefile
            makefile_path = Path(tmpdir) / "Makefile"
            makefile_path.write_text("""
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

all: $(TARGET)
""")
            
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

# Entry point for Modal
if __name__ == "__main__":
    app.serve()
