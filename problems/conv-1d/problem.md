---
slug: "conv-1d"
title: "1D Convolution"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "signal-processing"]
---

Perform 1D convolution between an input signal and a kernel:
$$
\text{C}[i] = \sum_{j=0}^{K-1} \text{A}[i + j] \cdot \text{B}[j]
$$

The convolution operation slides the kernel over the input signal, computing the sum of element-wise multiplications at each position. Zero padding is used at the boundaries.

## Input:
- Vector $\text{A}$ of size $\text{N}$ (input signal)
- Vector $\text{B}$ of size $\text{K}$ (convolution kernel)
- $\text{K}$ is odd and smaller than $\text{N}$

## Output:
- Vector $\text{C}$ of size $\text{N}$ (convolved signal)

## Notes:
- Use zero padding at the boundaries where the kernel extends beyond the input signal
- The kernel is centered at each position, with $(K-1)/2$ elements on each side
