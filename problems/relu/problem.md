---
slug: "relu"
title: "ReLU"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "neural-networks"]
---

Implement a CUDA kernel to perform the ReLU (Rectified Linear Unit) activation function on an input matrix:
$C_{i,j} = \max(0, A_{i,j})$

The ReLU function is defined as:
$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0 
\end{cases}
$$

## Input:
- Matrix $A$ of size $M \times N$ containing floating-point values

## Output:
- Matrix $C$ of size $M \times N$ containing the ReLU activation values