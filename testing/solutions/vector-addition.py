import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(input1_ptr, input2_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    input1 = tl.load(input1_ptr + offsets, mask=mask)
    input2 = tl.load(input2_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input1 + input2, mask=mask)

def solution(d_input1, d_input2, d_output, n: int):
    BLOCK_SIZE = 256
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # print("Hoi")

    vector_add_kernel[(num_blocks,)](d_input1, d_input2, d_output, n, BLOCK_SIZE=BLOCK_SIZE)
  