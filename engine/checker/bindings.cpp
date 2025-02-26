#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "python_reference.hpp"
#include "core.hpp"
#include <functional>

namespace py = pybind11;

// Wrapper function for Python to call
void run_checker() {
    // This will invoke the main function in checker.cu
    int result = system("./checker");
    if (result != 0) {
        throw std::runtime_error("Checker failed with exit code " + std::to_string(result));
    }
}

// Callback from Python to C++
void python_reference_callback_wrapper(
    py::function py_callback,
    float* input_a, float* input_b, float* output_c,
    size_t m, size_t n, size_t k) {
    
    // Convert C++ pointers to numpy arrays to pass to Python
    const size_t a_size = m * k;
    const size_t b_size = k * n;
    const size_t c_size = m * n;
    
    // Create numpy array views without copying the data
    py::array_t<float> py_input_a({m, k}, input_a);
    py::array_t<float> py_input_b({k, n}, input_b);
    
    // Call the Python function
    py::object result = py_callback(py_input_a, py_input_b, m, n, k);
    
    // Get numpy array from result
    py::array_t<float> py_output = result.cast<py::array_t<float>>();
    auto r = py_output.unchecked<2>();
    
    // Copy results back to C++ array
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            output_c[i * n + j] = r(i, j);
        }
    }
}

// Register a Python function as the reference implementation
void register_python_reference_impl(py::function py_callback) {
    // Create a C++ callback that wraps the Python function
    auto cpp_callback = [py_callback](
        float* input_a, float* input_b, float* output_c,
        size_t m, size_t n, size_t k) {
        python_reference_callback_wrapper(
            py_callback, input_a, input_b, output_c, m, n, k);
    };
    
    // Register the C++ callback
    register_python_reference(cpp_callback);
}

PYBIND11_MODULE(checker_bindings, m) {
    m.doc() = "PyTorch to CUDA checker bindings";
    
    // Expose function to register Python reference implementation
    m.def("register_reference", &register_python_reference_impl, 
          "Register a PyTorch reference implementation");
    
    // Expose function to run checker
    m.def("run_checker", &run_checker, 
          "Run the checker to compare PyTorch reference with CUDA solution");
} 