---
slug: "leaky-relu"
title: "Leaky ReLU"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "neural-networks"]
---

## Problem Statement
Implement a CUDA kernel to perform the Leaky ReLU (Leaky Rectified Linear Unit) activation function on an input matrix:
$\text{output}_{i,j} = \max(\alpha \cdot \text{input}_{i,j}, \text{input}_{i,j})$ where $\alpha$ is a small positive constant (e.g. 0.01)

The Leaky ReLU function is defined as:
$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$

## Input:
- Matrix $\text{input}$ of size $M \times N$ 
- $\alpha$ value (slope for negative values)

## Output:
- Matrix $\text{output}$ of size $M \times N$