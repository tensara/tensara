---
slug: "matrix-multiplication"
title: "Matrix Multiplication"
difficulty: "MEDIUM"
author: "sarthak"
tags: ["cuda-basics", "parallel-computing"]
---

## Problem Statement
Implement a CUDA kernel to perform matrix multiplication of two matrices:
$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$

## Input
- Matrix $A$ of size $M \times K$
- Matrix $B$ of size $K \times N$

## Output
- Matrix $C$ of size $M \times N$