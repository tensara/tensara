#include "core.hpp"
#include <random>

template<typename T>
class Conv2DTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(const T*, const T*, T*, size_t, size_t, size_t, size_t);
    
    Conv2DTest(size_t h, size_t w, size_t kh, size_t kw, unsigned int seed = 42) : rng_(seed) {
        this->problem_size_ = h * w;
        this->name_ = "H" + std::to_string(h) + "_W" + std::to_string(w) + 
                     "_Kh" + std::to_string(kh) + "_Kw" + std::to_string(kw);
        
        auto input_shape = std::vector<size_t>{h, w};
        auto kernel_shape = std::vector<size_t>{kh, kw};
        auto output_shape = std::vector<size_t>{h, w};

        this->inputs_ = {
            std::make_shared<Tensor<const T>>(input_shape),
            std::make_shared<Tensor<const T>>(kernel_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(output_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t H = this->inputs_[0]->shape()[0];
        const size_t W = this->inputs_[0]->shape()[1];
        const size_t Kh = this->inputs_[1]->shape()[0];
        const size_t Kw = this->inputs_[1]->shape()[1];
        
        std::uniform_real_distribution<T> input_dist(-10.0, 10.0);
        std::normal_distribution<T> kernel_dist(0.0, 1.0);
        
        for (size_t i = 0; i < H * W; i++) {
            host_inputs[0][i] = input_dist(rng_);
        }

        for (size_t i = 0; i < Kh * Kw; i++) {
            host_inputs[1][i] = kernel_dist(rng_);
        }
    }

    size_t calculate_flops() const override {
        const size_t H = this->inputs_[0]->shape()[0];
        const size_t W = this->inputs_[0]->shape()[1];
        const size_t Kh = this->inputs_[1]->shape()[0];
        const size_t Kw = this->inputs_[1]->shape()[1];
        return 2 * H * W * Kh * Kw;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t H = this->inputs_[0]->shape()[0];
        const size_t W = this->inputs_[0]->shape()[1];
        const size_t Kh = this->inputs_[1]->shape()[0];
        const size_t Kw = this->inputs_[1]->shape()[1];
        return {H, W, Kh, Kw};
    }

    void launch_kernel(const std::vector<const T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], outputs[0], sizes[0], sizes[1], sizes[2], sizes[3]);
    }

private:
    std::mt19937 rng_;
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> sizes = {
        {512, 512, 3, 3},
        {1024, 1024, 5, 5}, 
        {2048, 2048, 7, 7},
        {4096, 4096, 9, 9},
        {8192, 8192, 11, 11},
        {16384, 16384, 13, 13},
        {1024, 1024, 32, 32},
        {2048, 2048, 64, 64},
        {4096, 4096, 128, 128}
    };

    unsigned int base_seed = 42;
    for (const auto& [H, W, Kh, Kw] : sizes) {
        test_cases.push_back(std::make_unique<Conv2DTest<float>>(H, W, Kh, Kw, base_seed++));
    }
    
    return test_cases;
}
