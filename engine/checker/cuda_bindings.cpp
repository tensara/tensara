#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n);

void cuda_solution(torch::Tensor input1, torch::Tensor input2, torch::Tensor output) {
    TORCH_CHECK(input1.device().is_cuda(), "input1 must be a CUDA tensor");
    TORCH_CHECK(input2.device().is_cuda(), "input2 must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    
    TORCH_CHECK(input1.scalar_type() == torch::kFloat32, "input1 must be float32");
    TORCH_CHECK(input2.scalar_type() == torch::kFloat32, "input2 must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    
    const size_t n = input1.numel();
    
    solution(
        input1.data_ptr<float>(),
        input2.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

PYBIND11_MODULE(cuda_solution, m) {
    m.doc() = "CUDA vector addition with PyTorch"; 
    m.def("cuda_solution", &cuda_solution, "CUDA vector addition",
          py::arg("input1"), py::arg("input2"), py::arg("output"));
} 