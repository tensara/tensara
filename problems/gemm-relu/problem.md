---
slug: "gemm-relu"
title: "GEMM with Bias and ReLU"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "neural-networks"]
parameters:
  - name: "A"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "W" 
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "b" 
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "C" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "B"
    type: "size_t"
    pointer: "false"
    constant: "false"

  - name: "N"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "M" 
    type: "size_t"
    pointer: "false"
    constant: "false"
  
---

Perform a matrix multiplication followed by bias addition and ReLU activation:
$$
\text{C} = \text{ReLU}(\text{A} \cdot \text{W}^T + \text{b})
$$

The ReLU (Rectified Linear Unit) activation function is defined as:
$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0 
\end{cases}
$$

## Input:
- Matrix $\text{A}$ of size $\text{B} \times \text{N}$ corresponding to `batch_size x input_features`
- Matrix $\text{W}$ of size $\text{M} \times \text{N}$ (weights)
- Vector $\text{b}$ of size $\text{M}$ (bias)

## Output:
- Matrix $\text{C}$ of size $\text{B} \times \text{M}$

## Notes:
- All matrices $\text{A}$, $\text{W}$, and $\text{C}$ are stored in row-major order