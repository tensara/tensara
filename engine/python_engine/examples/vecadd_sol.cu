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