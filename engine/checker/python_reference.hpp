#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

// Function pointer type for Python reference solution callback
typedef void (*PythonReferenceCallback)(
    float* input_a, float* input_b, float* output_c,
    size_t m, size_t n, size_t k);

// Global function pointer that will hold the Python callback
extern PythonReferenceCallback g_python_reference_callback;

// Initialize function to register the Python callback
extern "C" void register_python_reference(PythonReferenceCallback callback);

// Reference solution that calls into Python
extern "C" void reference_solution(float* input_a, float* input_b, float* output_c, size_t m, size_t n, size_t k); 