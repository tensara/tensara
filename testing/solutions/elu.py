import triton
import triton.language as tl


# Note: input, output are device tensors
def solution(input, output, n: int, m: int, alpha: float):
    num_elements = m * n
    if num_elements <= 0:
        return

    BLOCK_SIZE = 256

    @triton.jit
    def elu_kernel(A_ptr, C_ptr, num_elements, alpha, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_elements

        a = tl.load(A_ptr + offs, mask=mask)
        zero = tl.zeros([BLOCK_SIZE], dtype=a.dtype)
        one = zero + 1.0

        pos = a
        neg = alpha * (tl.exp(a) - one)
        c = tl.where(a > zero, pos, neg)
        tl.store(C_ptr + offs, c, mask=mask)

    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    elu_kernel[grid](input, output, num_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
