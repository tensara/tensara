---
slug: "square-matmul"
title: "Square Matrix Multiplication"
difficulty: "EASY"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
---

Perform multiplication of two square matrices:
$$
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]
$$

## Input
- Matrix $A$ of size $N \times N$
- Matrix $B$ of size $N \times N$ 

## Output
- Matrix $C = AB$ of size $N \times N$