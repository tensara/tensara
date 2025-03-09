#include "core.hpp"
#include <random>

template<typename T>
class MatrixVectorTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(const T*, const T*, T*, size_t, size_t);
    
    MatrixVectorTest(size_t m, size_t k, unsigned int seed = 42) : rng_(seed) {
        this->problem_size_ = m * k;
        this->name_ = std::to_string(m) + "x" + std::to_string(k) + " x " + std::to_string(k) + "x1";
        
        auto matrix_shape = std::vector<size_t>{m, k};

        this->inputs_ = {
            std::make_shared<Tensor<const T>>(matrix_shape),
            std::make_shared<Tensor<const T>>(std::vector<size_t>{k})
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(std::vector<size_t>{m})
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        
        std::uniform_real_distribution<T> dist(-10000.0, 10000.0);

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < k; j++) {
                host_inputs[0][i * k + j] = dist(rng_);
            }
        }
        
        for (size_t i = 0; i < k; i++) {
            host_inputs[1][i] = dist(rng_);
        }
    }

    size_t calculate_flops() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        return m * k * 2;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        return {m, k};
    }

    void launch_kernel(const std::vector<const T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], outputs[0], sizes[0], sizes[1]);
    }

private:
    std::mt19937 rng_;
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    test_cases.push_back(std::make_unique<MatrixVectorTest<float>>(4096, 4096, 4891));
    test_cases.push_back(std::make_unique<MatrixVectorTest<float>>(6144, 4096, 3928));
    test_cases.push_back(std::make_unique<MatrixVectorTest<float>>(7168, 4096, 48923));
    test_cases.push_back(std::make_unique<MatrixVectorTest<float>>(8192, 4096, 382));
    test_cases.push_back(std::make_unique<MatrixVectorTest<float>>(9216, 4096, 48));
    return test_cases;
}
