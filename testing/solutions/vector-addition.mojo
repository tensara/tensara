from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from memory import UnsafePointer
from layout import Layout, LayoutTensor


comptime SIZE = 2
comptime layout = Layout.row_major(SIZE, SIZE)
comptime dtype = DType.float32


fn add_kernel(
    input1: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    input2: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int32,
):
    idx = block_idx.x * block_dim.x + thread_idx.x
    if Int32(idx) < n:
        output[idx] = input1[idx] + input2[idx]


@export
fn solution(input1_addr: Int, input2_addr: Int, output_addr: Int, n: Int32) raises:
    input1 = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=input1_addr)
    input2 = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=input2_addr)
    output = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=output_addr)

    ctx = DeviceContext()
    block_size = (256, 1)
    grid_size = ((n + block_size[0] - 1) // block_size[0], 1)
    # print(input1.get_device_type_name())

    ctx.enqueue_function[add_kernel, add_kernel](
        input1,
        input2,
        output,
        n,
        grid_dim=grid_size,
        block_dim=block_size,
    )

    ctx.synchronize()
