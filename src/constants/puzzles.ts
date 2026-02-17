export interface GpuPuzzle {
  id: number;
  slug: string;
  title: string;
  difficulty: "intro" | "easy" | "medium" | "hard";
  description: string;
  hint?: string;
  starterCode: string;
  setupCode: string;
  spec: string;
  expectedOutput: string;
}

export const GPU_PUZZLES: GpuPuzzle[] = [
  {
    id: 1,
    slug: "map",
    title: "Map",
    difficulty: "intro",
    description:
      "Implement a kernel that adds 10 to each position of vector `a` and stores it in `out`. You have 1 thread per position.",
    hint: "Think of the function as being run 1 time for each thread. `cuda.threadIdx.x` changes each time.",
    starterCode: `def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 line)

    return call`,
    setupCode: `SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem("Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec)`,
    spec: `def map_spec(a):
    return a + 10`,
    expectedOutput: "[10 11 12 13]",
  },
  {
    id: 2,
    slug: "zip",
    title: "Zip",
    difficulty: "intro",
    description:
      "Implement a kernel that adds together each position of `a` and `b` and stores it in `out`. You have 1 thread per position.",
    starterCode: `def zip_test(cuda):
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 1 line)

    return call`,
    setupCode: `SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem("Zip", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec)`,
    spec: `def zip_spec(a, b):
    return a + b`,
    expectedOutput: "[0 2 4 6]",
  },
  {
    id: 3,
    slug: "guards",
    title: "Guards",
    difficulty: "intro",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have more threads than positions.",
    hint: "Use a guard to check if the thread index is within bounds.",
    starterCode: `def map_guard_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 2 lines)

    return call`,
    setupCode: `SIZE = 4
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem("Guard", map_guard_test, [a], out, [SIZE], threadsperblock=Coord(8, 1), spec=map_spec)`,
    spec: `def map_spec(a):
    return a + 10`,
    expectedOutput: "[10 11 12 13]",
  },
  {
    id: 4,
    slug: "map-2d",
    title: "Map 2D",
    difficulty: "easy",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out`. Input `a` is 2D and square. You have more threads than positions.",
    starterCode: `def map_2D_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # FILL ME IN (roughly 2 lines)

    return call`,
    setupCode: `SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
problem = CudaProblem("Map 2D", map_2D_test, [a], out, [SIZE], threadsperblock=Coord(3, 3), spec=map_spec)`,
    spec: `def map_spec(a):
    return a + 10`,
    expectedOutput: "[[10 11]\n [12 13]]",
  },
  {
    id: 5,
    slug: "broadcast",
    title: "Broadcast",
    difficulty: "easy",
    description:
      "Implement a kernel that adds `a` and `b` and stores it in `out`. Inputs `a` and `b` are vectors. You have more threads than positions.",
    starterCode: `def broadcast_test(cuda):
    def call(out, a, b, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # FILL ME IN (roughly 2 lines)

    return call`,
    setupCode: `SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE).reshape(SIZE, 1)
b = np.arange(SIZE).reshape(1, SIZE)
problem = CudaProblem("Broadcast", broadcast_test, [a, b], out, [SIZE], threadsperblock=Coord(3, 3), spec=zip_spec)`,
    spec: `def zip_spec(a, b):
    return a + b`,
    expectedOutput: "[[0 1]\n [1 2]]",
  },
  {
    id: 6,
    slug: "blocks",
    title: "Blocks",
    difficulty: "easy",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have fewer threads per block than the size of `a`.",
    hint: "A block is a group of threads. `cuda.blockIdx` tells us what block we are in.",
    starterCode: `def map_block_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # FILL ME IN (roughly 2 lines)

    return call`,
    setupCode: `SIZE = 9
out = np.zeros((SIZE,))
a = np.arange(SIZE)
problem = CudaProblem("Blocks", map_block_test, [a], out, [SIZE], threadsperblock=Coord(4, 1), blockspergrid=Coord(3, 1), spec=map_spec)`,
    spec: `def map_spec(a):
    return a + 10`,
    expectedOutput: "[10 11 12 13 14 15 16 17 18]",
  },
  {
    id: 7,
    slug: "blocks-2d",
    title: "Blocks 2D",
    difficulty: "medium",
    description:
      "Implement the same kernel in 2D. You have fewer threads per block than the size of `a` in both directions.",
    starterCode: `def map_block2D_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # FILL ME IN (roughly 4 lines)

    return call`,
    setupCode: `SIZE = 5
out = np.zeros((SIZE, SIZE))
a = np.ones((SIZE, SIZE))
problem = CudaProblem("Blocks 2D", map_block2D_test, [a], out, [SIZE], threadsperblock=Coord(3, 3), blockspergrid=Coord(2, 2), spec=map_spec)`,
    spec: `def map_spec(a):
    return a + 10`,
    expectedOutput: "[[11. 11. ...]\n ...]",
  },
  {
    id: 8,
    slug: "shared-memory",
    title: "Shared Memory",
    difficulty: "medium",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out` using shared memory. Each block can only have a constant amount of shared memory. After writing to shared memory you need to call `cuda.syncthreads`.",
    hint: "This example doesn't really need shared memory, but it's a demo of the pattern.",
    starterCode: `TPB = 4
def shared_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        if i < size:
            shared[local_i] = a[i]
            cuda.syncthreads()

        # FILL ME IN (roughly 2 lines)

    return call`,
    setupCode: `SIZE = 8
out = np.zeros(SIZE)
a = np.ones(SIZE)
problem = CudaProblem("Shared", shared_test, [a], out, [SIZE], threadsperblock=Coord(TPB, 1), blockspergrid=Coord(2, 1), spec=map_spec)`,
    spec: `def map_spec(a):
    return a + 10`,
    expectedOutput: "[11. 11. 11. 11. 11. 11. 11. 11.]",
  },
  {
    id: 9,
    slug: "pooling",
    title: "Pooling",
    difficulty: "medium",
    description:
      "Implement a kernel that sums together the last 3 positions of `a` and stores it in `out`. You have 1 thread per position. You only need 1 global read and 1 global write per thread.",
    hint: "Remember to be careful about syncing.",
    starterCode: `TPB = 8
def pool_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 8 lines)

    return call`,
    setupCode: `SIZE = 8
out = np.zeros(SIZE)
a = np.arange(SIZE)
problem = CudaProblem("Pooling", pool_test, [a], out, [SIZE], threadsperblock=Coord(TPB, 1), blockspergrid=Coord(1, 1), spec=pool_spec)`,
    spec: `def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out`,
    expectedOutput: "[ 0.  1.  3.  6.  9. 12. 15. 18.]",
  },
  {
    id: 10,
    slug: "dot-product",
    title: "Dot Product",
    difficulty: "medium",
    description:
      "Implement a kernel that computes the dot-product of `a` and `b` and stores it in `out`. You have 1 thread per position. You only need 2 global reads and 1 global write per thread.",
    starterCode: `TPB = 8
def dot_test(cuda):
    def call(out, a, b, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 9 lines)

    return call`,
    setupCode: `SIZE = 8
out = np.zeros(1)
a = np.arange(SIZE)
b = np.arange(SIZE)
problem = CudaProblem("Dot", dot_test, [a, b], out, [SIZE], threadsperblock=Coord(SIZE, 1), blockspergrid=Coord(1, 1), spec=dot_spec)`,
    spec: `def dot_spec(a, b):
    return a @ b`,
    expectedOutput: "140",
  },
  {
    id: 11,
    slug: "1d-convolution",
    title: "1D Convolution",
    difficulty: "hard",
    description:
      "Implement a kernel that computes a 1D convolution between `a` and `b` and stores it in `out`. You need to handle the general case. You only need 2 global reads and 1 global write per thread.",
    starterCode: `MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 17 lines)

    return call`,
    setupCode: `SIZE = 6; CONV = 3
out = np.zeros(SIZE)
a = np.arange(SIZE)
b = np.arange(CONV)
problem = CudaProblem("1D Conv", conv_test, [a, b], out, [SIZE, CONV], Coord(1, 1), Coord(TPB, 1), spec=conv_spec)`,
    spec: `def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out`,
    expectedOutput: "[ 5.  8. 11. 14.  5.  0.]",
  },
  {
    id: 12,
    slug: "prefix-sum",
    title: "Prefix Sum",
    difficulty: "hard",
    description:
      "Implement a kernel that computes a sum over `a` and stores it in `out`. If the size of `a` is greater than the block size, only store the sum of each block. Use the parallel prefix sum algorithm in shared memory.",
    hint: "Each step of the algorithm should sum together half the remaining numbers.",
    starterCode: `TPB = 8
def sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        # FILL ME IN (roughly 12 lines)

    return call`,
    setupCode: `SIZE = 8
out = np.zeros(1)
inp = np.arange(SIZE)
problem = CudaProblem("Sum", sum_test, [inp], out, [SIZE], Coord(1, 1), Coord(TPB, 1), spec=sum_spec)`,
    spec: `TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0] + TPB - 1) // TPB)
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[j] = a[i : i + TPB].sum()
    return out`,
    expectedOutput: "[28.]",
  },
  {
    id: 13,
    slug: "axis-sum",
    title: "Axis Sum",
    difficulty: "hard",
    description:
      "Implement a kernel that computes a sum over each column of `a` and stores it in `out`.",
    starterCode: `TPB = 8
def axis_sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        batch = cuda.blockIdx.y
        # FILL ME IN (roughly 12 lines)

    return call`,
    setupCode: `BATCH = 4; SIZE = 6
out = np.zeros((BATCH, 1))
inp = np.arange(BATCH * SIZE).reshape((BATCH, SIZE))
problem = CudaProblem("Axis Sum", axis_sum_test, [inp], out, [SIZE], Coord(1, BATCH), Coord(TPB, 1), spec=sum_spec)`,
    spec: `TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[..., j] = a[..., i : i + TPB].sum(-1)
    return out`,
    expectedOutput: "[[ 15.]\n [ 51.]\n [ 87.]\n [123.]]",
  },
  {
    id: 14,
    slug: "matrix-multiply",
    title: "Matrix Multiply",
    difficulty: "hard",
    description:
      "Implement a kernel that multiplies square matrices `a` and `b` and stores the result in `out`.",
    hint: "The most efficient algorithm copies a block into shared memory before computing individual row-column dot products. Do the simple case first, then handle partial dot-products iteratively.",
    starterCode: `TPB = 3
def mm_oneblock_test(cuda):
    def call(out, a, b, size: int) -> None:
        a_shared = cuda.shared.array((TPB, TPB), numba.float32)
        b_shared = cuda.shared.array((TPB, TPB), numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        # FILL ME IN (roughly 14 lines)

    return call`,
    setupCode: `SIZE = 2
out = np.zeros((SIZE, SIZE))
inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T
problem = CudaProblem("Matmul", mm_oneblock_test, [inp1, inp2], out, [SIZE], Coord(1, 1), Coord(TPB, TPB), spec=matmul_spec)`,
    spec: `def matmul_spec(a, b):
    return a @ b`,
    expectedOutput: "[[ 1  3]\n [ 3 13]]",
  },
];
