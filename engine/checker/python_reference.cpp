#include "python_reference.hpp"
#include <iostream>

// Initialize the global callback pointer to nullptr
PythonReferenceCallback g_python_reference_callback = nullptr;

// Function to register the Python callback
extern "C" void register_python_reference(PythonReferenceCallback callback) {
    g_python_reference_callback = callback;
    std::cout << "Python reference callback registered" << std::endl;
}

// Implementation of the reference_solution that calls the Python callback
extern "C" void reference_solution(float* input_a, float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    if (g_python_reference_callback == nullptr) {
        std::cerr << "Error: Python reference callback not registered" << std::endl;
        // Fill output with zeros to avoid undefined behavior
        for (size_t i = 0; i < m * n; i++) {
            output_c[i] = 0.0f;
        }
        return;
    }
    
    // Call the Python implementation
    g_python_reference_callback(input_a, input_b, output_c, m, n, k);
} 