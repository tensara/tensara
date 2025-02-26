---
slug: "matrix-multiplication"
title: "Matrix Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
---

Perform matrix multiplication of two matrices:
$$
C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]
$$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $K \times N$

## Output
- Matrix $C$ of size $M \times N$