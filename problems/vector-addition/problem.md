---
slug: "vector-addition"
title: "Vector Addition"
difficulty: "EASY"
author: "someshkar"
tags: ["cuda-basics", "parallel-computing"]
parameters:
  - name: "d_input1"
    type: "[VAR]"
    pointer: "true"
    const: "true"
  
  - name: "d_input2"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "d_output" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "n" 
    type: "size_t"
    pointer: "false"
    constant: "false"
    
---

Perform element-wise addition of two vectors:
$$
c_i = a_i + b_i
$$

## Input
- Vectors $a$ and $b$ of length $N$

## Output
- Vector $c$ of length $N$ containing the element-wise sum
