from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from memory import UnsafePointer

comptime dtype = DType.float32
comptime TILE_M = 16
comptime TILE_N = 16
comptime TILE_K = 32
comptime BLOCK = TILE_M * TILE_N


fn matmul_kernel(
    input_a: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    input_b: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    output_c: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
):
    var mm = Int(m)
    var nn = Int(n)
    var kk_max = Int(k)
    if mm <= 0 or nn <= 0 or kk_max <= 0:
        return

    var tid = Int(thread_idx.x)
    var tx = tid % TILE_N
    var ty = tid // TILE_N

    var blocks_per_row = (nn + TILE_N - 1) // TILE_N
    var block_linear = Int(block_idx.x)
    var block_row = block_linear // blocks_per_row
    var block_col = block_linear - block_row * blocks_per_row

    var row = block_row * TILE_M + ty
    var col = block_col * TILE_N + tx

    var valid = row < mm and col < nn

    var smem_a = LayoutTensor[
        dtype,
        Layout.row_major(TILE_M * TILE_K),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var smem_b = LayoutTensor[
        dtype,
        Layout.row_major(TILE_K * TILE_N),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Tile k=0 load (2 elements per thread for TILE_K=32).
    var a_col0 = tx
    var a_col1 = tx + TILE_N
    if row < mm and a_col0 < kk_max:
        smem_a[ty * TILE_K + a_col0] = input_a[row * kk_max + a_col0]
    else:
        smem_a[ty * TILE_K + a_col0] = 0.0
    if row < mm and a_col1 < kk_max:
        smem_a[ty * TILE_K + a_col1] = input_a[row * kk_max + a_col1]
    else:
        smem_a[ty * TILE_K + a_col1] = 0.0

    var b_row0 = ty
    var b_row1 = ty + TILE_M
    if b_row0 < kk_max and col < nn:
        smem_b[b_row0 * TILE_N + tx] = input_b[b_row0 * nn + col]
    else:
        smem_b[b_row0 * TILE_N + tx] = 0.0
    if b_row1 < kk_max and col < nn:
        smem_b[b_row1 * TILE_N + tx] = input_b[b_row1 * nn + col]
    else:
        smem_b[b_row1 * TILE_N + tx] = 0.0

    barrier()

    # Initialize accumulator without Float64 literals.
    # Note: LayoutTensor scalar indexing yields a SIMD "element_type", so extract lane 0.
    var acc = smem_a[ty * TILE_K + 0][0] * smem_b[0 * TILE_N + tx][0]
    @parameter
    for t in range(1, TILE_K):
        acc += smem_a[ty * TILE_K + t][0] * smem_b[t * TILE_N + tx][0]

    barrier()

    # Remaining k tiles
    var base_k = TILE_K
    while base_k < kk_max:
        var a_col = base_k + tx
        var a_col_2 = base_k + tx + TILE_N
        if row < mm and a_col < kk_max:
            smem_a[ty * TILE_K + tx] = input_a[row * kk_max + a_col]
        else:
            smem_a[ty * TILE_K + tx] = 0.0
        if row < mm and a_col_2 < kk_max:
            smem_a[ty * TILE_K + tx + TILE_N] = input_a[row * kk_max + a_col_2]
        else:
            smem_a[ty * TILE_K + tx + TILE_N] = 0.0

        var b_row = base_k + ty
        var b_row_2 = base_k + ty + TILE_M
        if b_row < kk_max and col < nn:
            smem_b[ty * TILE_N + tx] = input_b[b_row * nn + col]
        else:
            smem_b[ty * TILE_N + tx] = 0.0
        if b_row_2 < kk_max and col < nn:
            smem_b[(ty + TILE_M) * TILE_N + tx] = input_b[b_row_2 * nn + col]
        else:
            smem_b[(ty + TILE_M) * TILE_N + tx] = 0.0

        barrier()

        @parameter
        for t in range(TILE_K):
            acc += smem_a[ty * TILE_K + t][0] * smem_b[t * TILE_N + tx][0]

        barrier()
        base_k += TILE_K

    if valid:
        output_c[row * nn + col] = acc


@export
fn solution(
    input_a_addr: Int,
    input_b_addr: Int,
    output_c_addr: Int,
    m: Int32,
    n: Int32,
    k: Int32,
) raises:
    input_a = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=input_a_addr)
    input_b = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=input_b_addr)
    output_c = UnsafePointer[Scalar[dtype], MutExternalOrigin](unsafe_from_address=output_c_addr)

    var mm = Int(m)
    var nn = Int(n)
    var kk_max = Int(k)
    if mm <= 0 or nn <= 0 or kk_max <= 0:
        return

    var blocks_per_row = (nn + TILE_N - 1) // TILE_N
    var blocks_per_col = (mm + TILE_M - 1) // TILE_M
    var num_blocks = blocks_per_row * blocks_per_col

    ctx = DeviceContext()
    block_size = (BLOCK, 1)
    grid_size = (num_blocks, 1)

    ctx.enqueue_function[matmul_kernel, matmul_kernel](
        input_a,
        input_b,
        output_c,
        m,
        n,
        k,
        grid_dim=grid_size,
        block_dim=block_size,
    )

    ctx.synchronize()
