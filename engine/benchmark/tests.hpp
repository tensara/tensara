#include "core.hpp"
#include "solution.cuh"

template<typename T>
class VectorAddTest: public TestCase<T> {
public:
    explicit VectorAddTest(size_t n) {
        this->problem_size_ = n;
        
        auto vec_shape = std::vector<size_t>{n};
        this->inputs_ = {
            std::make_shared<Tensor<T>>(vec_shape),
            std::make_shared<Tensor<T>>(vec_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(vec_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t n = this->problem_size_;
        
        for (size_t i = 0; i < n; i++) {
            host_inputs[0][i] = static_cast<T>(i);
            host_inputs[1][i] = static_cast<T>(i * 2);
        }
    }
    
    size_t calculate_flops() const override {
        return this->problem_size_;
    }
};

template<typename T>
void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, const std::vector<size_t>& sizes) {
    solution(inputs[0], inputs[1], outputs[0], sizes[0]);
}

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(1000000));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(5000000));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(10000000));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(50000000));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(100000000));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(500000000));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(1000000000));
    return test_cases;
} 
