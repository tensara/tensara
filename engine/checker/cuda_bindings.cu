#include <torch/extension.h>
#include "solution.cuh"

void cuda_vector_add(torch::Tensor input1, torch::Tensor input2, torch::Tensor output) {
    // Ensure inputs are on GPU and are float32
    TORCH_CHECK(input1.device().is_cuda(), "input1 must be a CUDA tensor");
    TORCH_CHECK(input2.device().is_cuda(), "input2 must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    
    TORCH_CHECK(input1.scalar_type() == torch::kFloat32, "input1 must be float32");
    TORCH_CHECK(input2.scalar_type() == torch::kFloat32, "input2 must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    
    const size_t n = input1.numel();
    
    // Call your existing solution function
    solution(
        input1.data_ptr<float>(),
        input2.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA vector addition with PyTorch"; 
    m.def("vector_add", &cuda_vector_add, "CUDA vector addition",
          py::arg("input1"), py::arg("input2"), py::arg("output"));
} 