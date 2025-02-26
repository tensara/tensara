---
slug: "matrix-vector"
title: "Matrix Vector Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
---

Perform multiplication of a matrix and a vector:
$$
C[i] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k]
$$

## Input:
- Matrix $A$ of size $M \times K$
- Vector $B$ of size $K \times 1$

## Output:
- Vector $C = AB$ of size $M \times 1$