#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tests.hpp"
#include "core.hpp"

extern "C" void reference_solution(float* d_input1, float* d_input2, float* d_output, size_t n);

bool check_results(float* output1, float* output2, size_t size, float tolerance = 1e-5) {
    for (size_t i = 0; i < size; i++) {
        if (fabs(output1[i] - output2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

template<typename T>
void launch_reference_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, const std::vector<size_t>& sizes) {
    reference_solution(inputs[0], inputs[1], outputs[0], sizes[0], sizes[1], sizes[2]);
}

template<typename T>
bool run_test(TestCase<T>& test_case) {
    const auto& input_shapes = test_case.input_shapes();
    const auto& output_shapes = test_case.output_shapes();
    
    std::vector<T*> h_inputs(input_shapes.size());
    std::vector<T*> h_outputs(output_shapes.size());
    std::vector<T*> h_reference_outputs(output_shapes.size());
    
    for (size_t i = 0; i < input_shapes.size(); i++) {
        h_inputs[i] = new T[input_shapes[i]->size()];
    }
    for (size_t i = 0; i < output_shapes.size(); i++) {
        h_outputs[i] = new T[output_shapes[i]->size()];
        h_reference_outputs[i] = new T[output_shapes[i]->size()];
    }
    
    test_case.prepare_data(h_inputs.data(), h_outputs.data());
    
    std::vector<T*> d_inputs(input_shapes.size());
    std::vector<T*> d_outputs(output_shapes.size());
    std::vector<T*> d_reference_outputs(output_shapes.size());
    
    for (size_t i = 0; i < input_shapes.size(); i++) {
        cudaMalloc(&d_inputs[i], input_shapes[i]->size() * sizeof(T));
        cudaMemcpy(d_inputs[i], h_inputs[i], input_shapes[i]->size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    for (size_t i = 0; i < output_shapes.size(); i++) {
        cudaMalloc(&d_outputs[i], output_shapes[i]->size() * sizeof(T));
        cudaMalloc(&d_reference_outputs[i], output_shapes[i]->size() * sizeof(T));
    }
    
    std::vector<size_t> sizes = test_case.get_sizes();
    launch_kernel(d_inputs, d_outputs, sizes);
    launch_reference_kernel(d_inputs, d_reference_outputs, sizes);
    
    for (size_t i = 0; i < output_shapes.size(); i++) {
        cudaMemcpy(h_outputs[i], d_outputs[i], output_shapes[i]->size() * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_reference_outputs[i], d_reference_outputs[i], output_shapes[i]->size() * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    bool passed = true;
    for (size_t i = 0; i < output_shapes.size(); i++) {
        if (!check_results(h_outputs[i], h_reference_outputs[i], output_shapes[i]->size())) {
            passed = false;
            break;
        }
    }
    
    for (size_t i = 0; i < input_shapes.size(); i++) {
        delete[] h_inputs[i];
        cudaFree(d_inputs[i]);
    }
    for (size_t i = 0; i < output_shapes.size(); i++) {
        delete[] h_outputs[i];
        delete[] h_reference_outputs[i];
        cudaFree(d_outputs[i]);
        cudaFree(d_reference_outputs[i]);
    }
    
    return passed;
}

int main() {
    auto test_cases = create_test_cases();
    bool all_passed = true;
    
    for (size_t i = 0; i < test_cases.size(); i++) {
        if (!run_test(*test_cases[i])) {
            std::cout << (i + 1) << "," << "FAILED" << std::endl;
            all_passed = false;
            break;
        }
        std::cout << (i + 1) << "," << "PASSED" << std::endl;
    }
    
    if (all_passed) {
        std::cout << "PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
} 