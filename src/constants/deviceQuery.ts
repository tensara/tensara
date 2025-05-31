export type DeviceQueryGpu = {
  name: string;
  cudaCapability: {
    major: number;
    minor: number;
  };
  globalMemory: number; // assume bytes
  multiprocessors: number;
  cudaCoresPerMP: number;
  totalCUDACores: number;
  gpuMaxClockRate: number; // MHz
  memoryClockRate: number; // MHz
  memoryBusWidth: number; // bits
  l2CacheSize: number; // bytes
  textureDimensions: {
    max1D: number;
    max2D: number[];
    max3D: number[];
  };
  layeredTextureDimensions: {
    max1D: number;
    layers1D: number;
    max2D: number[];
    layers2D: number;
  };
  memory: {
    constantMemory: number; // bytes
    sharedMemoryPerBlock: number; // bytes
    sharedMemoryPerMP: number; // bytes
  };
  registersPerBlock: number;
  warpSize: number;
  threads: {
    maxPerMP: number;
    maxPerBlock: number;
    maxBlockDim: number[];
    maxGridDim: number[];
  };
  textureAlignment: number; // bytes
};

export const CUDA_DRIVER_VERSION = 12.9;
export const CUDA_RUNTIME_VERSION = 12.9;
export const PYTHON_VERSION = "3.11.5";
export const NVCC_CMD = "nvcc -std=c++20 -O2 -Xcompiler -fPIC";
export const MOJO_CMD = "mojo build --optimization-level=3";
export const TRITON_VERSION = "3.2.0";

export const DEVICE_QUERY_GPU_MAP: Record<string, DeviceQueryGpu> = {
  T4: {
    name: "Tesla T4",
    cudaCapability: {
      major: 7,
      minor: 5,
    },
    globalMemory: 15638134784, // bytes
    multiprocessors: 40,
    cudaCoresPerMP: 64,
    totalCUDACores: 2560,
    gpuMaxClockRate: 1590, // MHz
    memoryClockRate: 5001, // MHz
    memoryBusWidth: 256, // bits
    l2CacheSize: 4194304, // bytes
    textureDimensions: {
      max1D: 131072,
      max2D: [131072, 65536],
      max3D: [16384, 16384, 16384],
    },
    layeredTextureDimensions: {
      max1D: 32768,
      layers1D: 2048,
      max2D: [32768, 32768],
      layers2D: 2048,
    },
    memory: {
      constantMemory: 65536, // bytes
      sharedMemoryPerBlock: 49152, // bytes
      sharedMemoryPerMP: 65536, // bytes
    },
    registersPerBlock: 65536,
    warpSize: 32,
    threads: {
      maxPerMP: 1024,
      maxPerBlock: 1024,
      maxBlockDim: [1024, 1024, 64],
      maxGridDim: [2147483647, 65535, 65535],
    },
    textureAlignment: 512, // bytes
  },
  A10G: {
    name: "NVIDIA A10G",
    cudaCapability: {
      major: 8,
      minor: 6,
    },
    globalMemory: 23696375808, // bytes
    multiprocessors: 80,
    cudaCoresPerMP: 128,
    totalCUDACores: 10240,
    gpuMaxClockRate: 1710, // MHz
    memoryClockRate: 6251, // MHz
    memoryBusWidth: 384, // bits
    l2CacheSize: 6291456, // bytes
    textureDimensions: {
      max1D: 131072,
      max2D: [131072, 65536],
      max3D: [16384, 16384, 16384],
    },
    layeredTextureDimensions: {
      max1D: 32768,
      layers1D: 2048,
      max2D: [32768, 32768],
      layers2D: 2048,
    },
    memory: {
      constantMemory: 65536, // bytes
      sharedMemoryPerBlock: 49152, // bytes
      sharedMemoryPerMP: 102400, // bytes
    },
    registersPerBlock: 65536,
    warpSize: 32,
    threads: {
      maxPerMP: 1536,
      maxPerBlock: 1024,
      maxBlockDim: [1024, 1024, 64],
      maxGridDim: [2147483647, 65535, 65535],
    },
    textureAlignment: 512, // bytes
  },
  "A100-80GB": {
    name: "NVIDIA A100 (80GB)",
    cudaCapability: {
      major: 8,
      minor: 0,
    },
    globalMemory: 85095874560, // bytes
    multiprocessors: 108,
    cudaCoresPerMP: 64,
    totalCUDACores: 6912,
    gpuMaxClockRate: 1410, // MHz
    memoryClockRate: 1512, // MHz
    memoryBusWidth: 5120, // bits
    l2CacheSize: 41943040, // bytes
    textureDimensions: {
      max1D: 131072,
      max2D: [131072, 65536],
      max3D: [16384, 16384, 16384],
    },
    layeredTextureDimensions: {
      max1D: 32768,
      layers1D: 2048,
      max2D: [32768, 32768],
      layers2D: 2048,
    },
    memory: {
      constantMemory: 65536, // bytes
      sharedMemoryPerBlock: 49152, // bytes
      sharedMemoryPerMP: 167936, // bytes
    },
    registersPerBlock: 65536,
    warpSize: 32,
    threads: {
      maxPerMP: 2048,
      maxPerBlock: 1024,
      maxBlockDim: [1024, 1024, 64],
      maxGridDim: [2147483647, 65535, 65535],
    },
    textureAlignment: 512, // bytes
  },
  L40S: {
    name: "NVIDIA L40S",
    cudaCapability: {
      major: 8,
      minor: 9,
    },
    globalMemory: 47677177856, // bytes
    multiprocessors: 142,
    cudaCoresPerMP: 128,
    totalCUDACores: 18176,
    gpuMaxClockRate: 2520, // MHz
    memoryClockRate: 9001, // MHz
    memoryBusWidth: 384, // bits
    l2CacheSize: 100663296, // bytes
    textureDimensions: {
      max1D: 131072,
      max2D: [131072, 65536],
      max3D: [16384, 16384, 16384],
    },
    layeredTextureDimensions: {
      max1D: 32768,
      layers1D: 2048,
      max2D: [32768, 32768],
      layers2D: 2048,
    },
    memory: {
      constantMemory: 65536, // bytes
      sharedMemoryPerBlock: 49152, // bytes
      sharedMemoryPerMP: 102400, // bytes
    },
    registersPerBlock: 65536,
    warpSize: 32,
    threads: {
      maxPerMP: 1536,
      maxPerBlock: 1024,
      maxBlockDim: [1024, 1024, 64],
      maxGridDim: [2147483647, 65535, 65535],
    },
    textureAlignment: 512, // bytes
  },
  L4: {
    name: "NVIDIA L4",
    cudaCapability: {
      major: 8,
      minor: 9,
    },
    globalMemory: 23670685696, // bytes
    multiprocessors: 58,
    cudaCoresPerMP: 129,
    totalCUDACores: 7424,
    gpuMaxClockRate: 2040, // MHz
    memoryClockRate: 6251, // MHz
    memoryBusWidth: 192, // bits
    l2CacheSize: 50331648, // bytes
    textureDimensions: {
      max1D: 131072,
      max2D: [131072, 65536],
      max3D: [16384, 16384, 16384],
    },
    layeredTextureDimensions: {
      max1D: 32768,
      layers1D: 2048,
      max2D: [32768, 32768],
      layers2D: 2048,
    },
    memory: {
      constantMemory: 65536, // bytes
      sharedMemoryPerBlock: 49152, // bytes
      sharedMemoryPerMP: 102400, // bytes
    },
    registersPerBlock: 65536,
    warpSize: 32,
    threads: {
      maxPerMP: 1536,
      maxPerBlock: 1024,
      maxBlockDim: [1024, 1024, 64],
      maxGridDim: [2147483647, 65535, 65535],
    },
    textureAlignment: 512, // bytes
  },
  H100: {
    name: "NVIDIA H100",
    cudaCapability: {
      major: 9,
      minor: 0,
    },
    globalMemory: 85029158912, // bytes
    multiprocessors: 132,
    cudaCoresPerMP: 128,
    totalCUDACores: 16896,
    gpuMaxClockRate: 1980, // MHz
    memoryClockRate: 2619, // MHz
    memoryBusWidth: 5120, // bits
    l2CacheSize: 52428800, // bytes
    textureDimensions: {
      max1D: 131072,
      max2D: [131072, 65536],
      max3D: [16384, 16384, 16384],
    },
    layeredTextureDimensions: {
      max1D: 32768,
      layers1D: 2048,
      max2D: [32768, 32768],
      layers2D: 2048,
    },
    memory: {
      constantMemory: 65536, // bytes
      sharedMemoryPerBlock: 49152, // bytes
      sharedMemoryPerMP: 233472, // bytes
    },
    registersPerBlock: 65536,
    warpSize: 32,
    threads: {
      maxPerMP: 2048,
      maxPerBlock: 1024,
      maxBlockDim: [1024, 1024, 64],
      maxGridDim: [2147483647, 65535, 65535],
    },
    textureAlignment: 512, // bytes
  },
} as const;
