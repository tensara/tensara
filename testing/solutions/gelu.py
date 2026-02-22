import triton
import triton.language as tl


# Note: input, output are device tensors
def solution(input, output, n: int, m: int):
    num_elements = m * n
    if num_elements <= 0:
        return

    BLOCK_SIZE = 256

    @triton.jit
    def gelu_kernel(A_ptr, C_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_elements

        x = tl.load(A_ptr + offs, mask=mask)

        # GELU approximation:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        k_alpha = 0.044715
        k_sqrt2_over_pi = 0.7978845608028654  # sqrt(2/pi)

        x3 = x * x * x
        inner = k_sqrt2_over_pi * (x + k_alpha * x3)
        t = tl.tanh(inner)

        half = 0.5 * x
        one = tl.zeros([BLOCK_SIZE], dtype=x.dtype) + 1.0
        y = half * (one + t)

        tl.store(C_ptr + offs, y, mask=mask)

    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    gelu_kernel[grid](input, output, num_elements, BLOCK_SIZE=BLOCK_SIZE)

