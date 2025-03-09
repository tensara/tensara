---
slug: "relu"
title: "ReLU"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "neural-networks"]
parameters:
  - name: "input"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "output"
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "m"
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform the ReLU (Rectified Linear Unit) activation function on an input matrix:
$$
C[i][j] = \max(0, A[i][j])
$$

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

## Notes:
- Both matrices $\text{A}$ and $\text{C}$ are stored in row-major order