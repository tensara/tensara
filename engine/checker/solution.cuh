#pragma once
#include <cuda_runtime.h>

__global__ void vector_add(const float* input1, const float* input2, float* output, size_t n);
void solution(float* d_input1, float* d_input2, float* d_output, size_t n);
