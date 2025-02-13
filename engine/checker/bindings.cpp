#include <pybind11/pybind11.h>
#include "solution.cuh"

namespace py = pybind11;

// Wrapper function that takes PyTorch tensors
void solution_wrapper(py::buffer input1, py::buffer input2, py::buffer output) {
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    py::buffer_info buf_out = output.request();
    
    float* ptr1 = static_cast<float*>(buf1.ptr);
    float* ptr2 = static_cast<float*>(buf2.ptr);
    float* ptr_out = static_cast<float*>(buf_out.ptr);
    
    size_t n = buf1.size;
    solution(ptr1, ptr2, ptr_out, n);
}

PYBIND11_MODULE(cuda_solution, m) {
    m.doc() = "CUDA vector addition kernel bindings"; 
    m.def("solution", &solution_wrapper, "Vector addition CUDA implementation",
          py::arg("input1"), py::arg("input2"), py::arg("output"));
} 