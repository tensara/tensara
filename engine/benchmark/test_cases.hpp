#pragma once
#include <vector>
#include <memory>

template<typename T>
class TestCase {
public:
    virtual ~TestCase() {
        for (auto ptr : host_inputs) {
            delete[] ptr;
        }
    }

    const std::vector<const T*>& get_inputs() const {
        return const_inputs;
    }

    size_t get_size() const {
        return size;
    }

    size_t get_num_inputs() const {
        return host_inputs.size();
    }

    size_t get_num_outputs() const {
        return num_outputs;
    }

protected:
    std::vector<T*> host_inputs;
    std::vector<const T*> const_inputs;
    size_t size;
    size_t num_outputs;
}; 