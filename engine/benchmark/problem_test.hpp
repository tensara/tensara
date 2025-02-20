#pragma once
#include <vector>
#include "test_cases.hpp"
#include "solution.cuh"

template<typename T>
class VectorAddTestCase : public TestCase<T> {
public:
    VectorAddTestCase(size_t n) {
        this->size = n;
        this->num_outputs = 1;
        
        this->host_inputs.resize(2);
        for (size_t i = 0; i < this->host_inputs.size(); i++) {
            this->host_inputs[i] = new T[n];
        }
        
        for (size_t i = 0; i < n; i++) {
            this->host_inputs[0][i] = static_cast<T>(i);
            this->host_inputs[1][i] = static_cast<T>(i * 2);
        }
        
        for (const auto& input : this->host_inputs) {
            this->const_inputs.push_back(input);
        }
    }
};

void launch_kernel(const std::vector<float*>& inputs, 
                  const std::vector<float*>& outputs, 
                  size_t n) {
    solution(inputs[0], inputs[1], outputs[0], n);
}

std::vector<TestCase<float>*> create_test_cases() {
    std::vector<TestCase<float>*> test_cases;
    test_cases.push_back(new VectorAddTestCase<float>(10));
    test_cases.push_back(new VectorAddTestCase<float>(100));
    test_cases.push_back(new VectorAddTestCase<float>(1 << 24));
    return test_cases;
} 