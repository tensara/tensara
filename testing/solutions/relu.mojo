from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from memory import UnsafePointer

comptime dtype = DType.float32


fn relu_kernel(
    input: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int32,
    m: Int32,
):
    idx = block_idx.x * block_dim.x + thread_idx.x
    total = n * m
    if Int32(idx) < total:
        x = input[idx]
        zero = x - x
        if x > zero:
            output[idx] = x
        else:
            output[idx] = zero


@export
fn solution(input_addr: Int, output_addr: Int, n: Int32, m: Int32) raises:
    input = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=input_addr)
    output = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=output_addr)

    if n <= 0 or m <= 0:
        return

    total = n * m

    ctx = DeviceContext()
    block_size = (256, 1)
    grid_size = ((total + block_size[0] - 1) // block_size[0], 1)

    ctx.enqueue_function[relu_kernel, relu_kernel](
        input,
        output,
        n,
        m,
        grid_dim=grid_size,
        block_dim=block_size,
    )

    ctx.synchronize()
