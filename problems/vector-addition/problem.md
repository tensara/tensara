---
slug: "vector-addition"
title: "Vector Addition"
difficulty: "MEDIUM"
author: "someshkar"
tags: ["cuda-basics", "parallel-computing"]
---

## Problem Statement
Implement a CUDA kernel to perform element-wise addition of two vectors:
`c[i] = a[i] + b[i]`

## Input Specifications
- Vectors `a` and `b` will contain `N` floating-point elements
- `N` will be between 2^10 (1024) and 2^28 (268,435,456) elements

## Output Requirements
- Vector `c` must contain the element-wise sum
- Must handle non-uniform grid sizes efficiently

## Examples

### Example 1
**Input:**
```json
{ "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0] }
```

**Output:**
```json
[5.0, 7.0, 9.0]
```

## Starter Code
```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Your implementation here
}
```

## Solution Setup
```json
{
  "kernel_name": "vector_add",
  "block_size": 256
}
```

## Hints
1. Use thread blocks effectively
2. Consider memory coalescing
3. Handle edge cases for grid dimensions
