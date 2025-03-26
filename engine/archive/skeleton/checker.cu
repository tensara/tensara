#include "core.hpp"
#include "reference.cu"
#include "solution.cu"
#include "tests.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

bool check_results(float *output1, float *output2, size_t size, float tolerance = 1e-7) {
    for (size_t i = 0; i < size; i++) {
        if (fabs(output1[i] - output2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool run_test(TestCase<T> &test_case) {
    const auto &input_shapes = test_case.input_shapes();
    const auto &output_shapes = test_case.output_shapes();

    std::vector<T *> h_inputs(input_shapes.size());
    std::vector<T *> h_outputs(output_shapes.size());
    std::vector<T *> h_reference_outputs(output_shapes.size());

    for (size_t i = 0; i < input_shapes.size(); i++) {
        h_inputs[i] = new T[input_shapes[i]->size()];
    }
    for (size_t i = 0; i < output_shapes.size(); i++) {
        h_outputs[i] = new T[output_shapes[i]->size()];
        h_reference_outputs[i] = new T[output_shapes[i]->size()];
    }

    test_case.prepare_data(h_inputs.data(), h_outputs.data());

    std::vector<const T*> d_inputs(input_shapes.size());
    std::vector<T *> d_outputs(output_shapes.size());
    std::vector<T *> d_reference_outputs(output_shapes.size());

    for (size_t i = 0; i < input_shapes.size(); i++) {
        cudaMalloc(const_cast<T**>(&d_inputs[i]), input_shapes[i]->size() * sizeof(T));
        cudaMemcpy(const_cast<T*>(d_inputs[i]), h_inputs[i], input_shapes[i]->size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    for (size_t i = 0; i < output_shapes.size(); i++) {
        cudaMalloc(&d_outputs[i], output_shapes[i]->size() * sizeof(T));
        cudaMalloc(&d_reference_outputs[i], output_shapes[i]->size() * sizeof(T));
    }

    std::vector<size_t> sizes = test_case.get_sizes();
    test_case.launch_kernel(d_inputs, d_outputs, sizes, reinterpret_cast<void *>(solution));
    test_case.launch_kernel(d_inputs, d_reference_outputs, sizes, reinterpret_cast<void *>(reference_solution));

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
        cudaFree(const_cast<T*>(d_inputs[i]));
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
    int total_cases = test_cases.size();

    for (size_t i = 0; i < total_cases; i++) {
        if (!run_test(*test_cases[i])) {
            std::cout << (i + 1) << "/" << total_cases << "," << test_cases[i]->get_name() << ","
                      << "FAILED" << std::endl;
            all_passed = false;
            break;
        }
        std::cout << (i + 1) << "/" << total_cases << "," << test_cases[i]->get_name() << ","
                  << "PASSED" << std::endl;
    }

    if (all_passed) {
        std::cout << "PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
}