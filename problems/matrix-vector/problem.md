---
slug: "matrix-vector"
title: "Matrix Vector Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
---

Implement a CUDA kernel to perform multiplication of a matrix and a vector:
$C_i = \sum_{k=0}^{K-1} A_{i,k} \cdot B_k$

## Input:
- Matrix $A$ of size $M \times K$
- Vector $B$ of size $K \times 1$

## Output:
- Vector $C = AB$ of size $M \times 1$