---
slug: "conv-2d"
title: "2D Convolution"
difficulty: "EASY" 
author: "sarthak"
tags: ["cuda-basics", "parallel-computing", "image-processing"]
parameters:
  - name: "A"
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "B" 
    type: "[VAR]"
    pointer: "true"
    const: "true"

  - name: "C" 
    type: "[VAR]"
    pointer: "true"
    const: "false"

  - name: "H"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "W" 
    type: "size_t"
    pointer: "false"
    constant: "false"
  
  - name: "Kh"
    type: "size_t"
    pointer: "false"
    constant: "false"
    
  - name: "Kw" 
    type: "size_t"
    pointer: "false"
    constant: "false"
---

Perform 2D convolution between an input image and a kernel:
$$
\text{C}[i,j] = \sum_{k=0}^{K_h-1}\sum_{l=0}^{K_w-1} \text{A}[i+k,j+l] \cdot \text{B}[k,l]
$$

The convolution operation slides the 2D kernel over the input image, computing the sum of element-wise multiplications at each position. Zero padding is used at the boundaries.

## Input:
- Matrix $\text{A}$ of size $\text{H} \times \text{W}$ (input image)
- Matrix $\text{B}$ of size $K_h \times K_w$ (convolution kernel)
- Both $K_h$ and $K_w$ are odd and smaller than $H$ and $W$ respectively

## Output:
- Matrix $\text{C}$ of size $\text{H} \times \text{W}$ (convolved image)

## Notes:
- All matrices $\text{A}$, $\text{B}$, and $\text{C}$ are stored in row-major order
- Use zero padding at the boundaries where the kernel extends beyond the input image
- The kernel is centered at each position, with $(K_h-1)/2$ rows and $(K_w-1)/2$ columns on each side
