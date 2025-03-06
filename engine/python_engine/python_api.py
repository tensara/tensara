import modal
from examples import matmul, vecadd
from checker import Checker
from pathlib import Path

DEVEL_IMAGE_NAME = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
CURR_DIR = Path(__file__).parent

image = modal.Image.from_registry(DEVEL_IMAGE_NAME, add_python="3.11").pip_install(
    "torch",
    "numpy"
)

image = image.add_local_file(CURR_DIR / "Makefile", "/Makefile")

app = modal.App("tensara-python-engine", image=image)


@app.function(gpu="T4")
def checker():
    sample_solution = """
    __global__ void vectorAddKernel(float* a, float* b, float* c, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input1,   // Input vector a
        d_input2,   // Input vector b
        d_output,   // Output vector c
        n           // Length of vectors
    );
    
    cudaDeviceSynchronize();
}
    """
    
    problem = vecadd.VectorAdditionProblem()
    checker = Checker(problem)
    return checker.check_solution(sample_solution)