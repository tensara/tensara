from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from memory import UnsafePointer

comptime dtype = DType.float32
comptime BLOCK = 256


fn matvec_kernel(
    input_a: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    input_b: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    output_c: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    m: Int32,
    k: Int32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var mm = Int(m)
    var kk_max = Int(k)
    if row >= mm or kk_max <= 0:
        return

    var b0 = input_b[0]
    var zero = b0 - b0

    var acc = zero
    var a_base = row * kk_max

    var kk = tid
    while kk < kk_max:
        acc += input_a[a_base + kk] * input_b[kk]
        kk += BLOCK

    var smem = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    smem[tid] = acc
    barrier()

    var stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        output_c[row] = smem[0][0]


@export
fn solution(
    input_a_addr: Int,
    input_b_addr: Int,
    output_c_addr: Int,
    m: Int32,
    k: Int32,
) raises:
    input_a = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=input_a_addr)
    input_b = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=input_b_addr)
    output_c = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=output_c_addr)

    if m <= 0 or k <= 0:
        return

    ctx = DeviceContext()
    block_size = (BLOCK, 1)
    grid_size = (m, 1)

    ctx.enqueue_function[matvec_kernel, matvec_kernel](
        input_a,
        input_b,
        output_c,
        m,
        k,
        grid_dim=grid_size,
        block_dim=block_size,
    )

    ctx.synchronize()
