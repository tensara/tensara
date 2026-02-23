import triton
import triton.language as tl


# Note: input, output are all float32 device tensors
def solution(input, alpha: float, output, n: int, m: int):
    # Set BLOCK_SIZE for parallelism. A good default for 1D problems like this
    # is often 1024 or 2048, but 128 is a common starting point for small kernels.
    BLOCK_SIZE = 128

    # The total number of elements in the matrix (M * N)
    num_elements = m * n

    # We use a 1D launch grid, where each program will handle BLOCK_SIZE elements.
    # The grid size is calculated as ceil(num_elements / BLOCK_SIZE).
    @triton.jit
    def leaky_relu_kernel(
        A_ptr,
        C_ptr,  # Pointers to the input (A) and output (C) matrices
        num_elements,  # Total number of elements (M * N)
        alpha,  # The alpha constant (slope for negative values)
        BLOCK_SIZE: tl.constexpr,  # Block size for parallel access
    ):
        # Get the program ID (pid). Since we're using a 1D grid, this is just pid_x.
        pid = tl.program_id(axis=0)

        # Create a range of indices for the current block.
        # For example, if BLOCK_SIZE=128 and pid=0, offs = [0, 1, ..., 127]
        # If pid=1, offs = [128, 129, ..., 255]
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Create a mask to ensure we don't read/write past the end of the matrix
        mask = offs < num_elements

        # Load the input values A[i][j] from memory
        a = tl.load(A_ptr + offs, mask=mask)

        # --- Leaky ReLU Calculation ---

        # 1. Compare 'a' with zero.
        is_positive = a > 0.0

        # 2. Calculate the value for the negative/zero case: alpha * a
        a_neg = alpha * a

        # 3. Use tl.where to select between the two cases:
        #    - If a > 0 (is_positive is True), the result is 'a'.
        #    - If a <= 0 (is_positive is False), the result is 'a_neg' (alpha * a).
        c = tl.where(is_positive, a, a_neg)

        # Alternative (more direct) implementation using max:
        # c = tl.maximum(alpha * a, a)

        # Store the result c into the output matrix C[i][j]
        tl.store(C_ptr + offs, c, mask=mask)

    # Calculate the total number of elements
    grid_size = triton.cdiv(num_elements, BLOCK_SIZE)

    # Launch the kernel
    leaky_relu_kernel[(grid_size,)](
        input,  # A_ptr
        output,  # C_ptr
        num_elements,
        alpha,
        BLOCK_SIZE=BLOCK_SIZE,
    )
