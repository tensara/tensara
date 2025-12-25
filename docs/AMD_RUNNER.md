# AMD GPU Runner Architecture

## Overview

Tensara's AMD runner enables HIP kernel benchmarking on **AMD MI300X GPUs** via **dstack.ai** and **AMD DevCloud**. It provides:

- Serverless-style execution on AMD hardware
- Automatic fleet management with dstack 2.0+
- Real-time SSE streaming of compilation and execution results
- Benchmark harness generation for performance testing
- Cost-effective execution ($1.99/hour with auto-shutdown)

## Quick Start

### 1. Configure Environment

Create or update `engine/.env` with the following variables:

```bash
# dstack AMD DevCloud Configuration
AMD_BACKEND=amddevcloud
AMD_FLEET_NAME=amd-mi300x-fleet
AMD_GPU_TYPE=MI300X
AMD_GPU_MEMORY=192GB
AMD_FLEET_MAX_NODES=1
AMD_FLEET_IDLE_DURATION=10m
AMD_SPOT_POLICY=auto
AMD_FLEET_DRY_RUN=false

# dstack.ai Authentication (get from https://sky.dstack.ai)
DSTACK_TOKEN=your_token_here
```

### 2. Create Fleet (One-Time Setup)

Fleet creation is **required** in dstack 2.0+. This was optional in earlier versions but is now mandatory:

```bash
cd engine
python create_amd_fleet.py
```

Verify the fleet was created:

```bash
dstack fleet list
```

You should see `amd-mi300x-fleet` with backend `amddevcloud`.

### 3. Start Development Server

```bash
npm run dev
```

Navigate to any problem (e.g., http://localhost:3000/problems/leaky-relu), select **MI300X** as the GPU type, and submit your HIP kernel.

## Architecture

### Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                           │
│                    (Problem Editor UI)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST
                            │ /api/amd/submit
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              src/pages/api/amd/submit.ts                        │
│              - Authenticates user                               │
│              - Validates payload                                │
│              - Sets up SSE streaming                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Spawns Python process
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              src/server/amd/runner.ts                           │
│              - Spawns Python subprocess                         │
│              - Streams SSE events back to frontend              │
│              - Updates database with results                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Executes
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              engine/amd_task_runner.py                          │
│              - Entry point for AMD task execution               │
│              - Generates SSE events (IN_QUEUE, PROVISIONING,    │
│                COMPILING, CHECKING, BENCHMARKING, etc.)         │
│              - Parses kernel output for results                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ dstack_cli_  │  │  hip_harness.py  │  │ amd_fleet_       │
│ wrapper.py   │  │                  │  │ manager.py       │
│              │  │  - Generates     │  │                  │
│ - Wraps      │  │    complete C++  │  │ - Ensures fleet  │
│   dstack CLI │  │    benchmark     │  │   exists         │
│ - Submits    │  │    harness with  │  │ - Creates fleet  │
│   tasks      │  │    main()        │  │   if missing     │
│ - Polls      │  │  - Memory alloc  │  │ - Validates      │
│   status     │  │  - Timing code   │  │   connectivity   │
│ - Retrieves  │  │  - Output format │  │                  │
│   logs       │  │                  │  │                  │
└──────┬───────┘  └──────────────────┘  └────────┬─────────┘
       │                                          │
       └──────────────────┬───────────────────────┘
                          │ dstack CLI commands
                          ▼
              ┌───────────────────────┐
              │     dstack CLI        │
              │  (subprocess calls)   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    dstack.ai API      │
              │  (sky.dstack.ai)      │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    AMD DevCloud       │
              │   MI300X GPU Fleet    │
              │   - ROCm 6.0+         │
              │   - hipcc compiler    │
              │   - 192GB VRAM        │
              └───────────────────────┘
```

### Execution Timeline

**Cold Start (First Submission):**

1. **0-10s**: Request validation, SSE setup
2. **10s-3min**: VM provisioning on AMD DevCloud (IN_QUEUE → PROVISIONING)
3. **3-4min**: HIP kernel compilation with hipcc (COMPILING)
4. **4-5min**: Kernel execution and benchmarking (CHECKING/BENCHMARKING)
5. **5-6min**: Results parsed and streamed back (BENCHMARKED/ACCEPTED)

**Warm Reuse (Within 10-Minute Window):**

1. **0-10s**: Request validation, SSE setup
2. **10-30s**: Compilation + execution (VM already warm)
3. **30-45s**: Results streamed back

Fleet auto-terminates VMs after 10 minutes of idle time to minimize costs.

## How dstack Works

### What is dstack?

**dstack** is an infrastructure orchestration tool for running AI/ML workloads on cloud GPUs. It provides:

- Unified API across multiple cloud providers (AWS, GCP, Azure, Lambda Labs, RunPod, etc.)
- Task-based execution model (submit task → provision resources → execute → terminate)
- Fleet management for persistent GPU pools
- Cost optimization via spot instances and auto-shutdown

Tensara uses dstack to access **AMD DevCloud**, which provides AMD MI300X GPUs.

### Backend Fleets vs SSH Fleets

dstack supports two fleet types:

| Feature           | Backend Fleet                            | SSH Fleet                   |
| ----------------- | ---------------------------------------- | --------------------------- |
| **Use Case**      | Cloud providers (AWS, GCP, AMD DevCloud) | On-premises servers         |
| **Provisioning**  | API-based (dynamic)                      | SSH-based (static IPs)      |
| **Configuration** | `backends: [amddevcloud]`                | `ssh_hosts: [host1, host2]` |
| **Scaling**       | Auto-scale on demand                     | Fixed pool size             |

**Tensara uses Backend Fleets** because AMD DevCloud is a cloud provider with API-based provisioning.

### Why Fleets Are Required (dstack 2.0+ API Change)

In **dstack 0.18-0.19**, you could submit tasks directly without pre-creating a fleet:

```yaml
# Old approach (no longer works)
type: task
backends: [amddevcloud]
resources:
  gpu: MI300X
```

In **dstack 0.20+**, you MUST create a fleet first:

```yaml
# Step 1: Create fleet (one-time)
type: fleet
name: amd-mi300x-fleet
backends: [amddevcloud]
resources:
  gpu: MI300X

# Step 2: Submit tasks to fleet
type: task
fleets: [amd-mi300x-fleet]
```

This change improves reliability and enables better resource management. The `amd_fleet_manager.py` module automatically ensures the fleet exists before submitting tasks.

### Task Lifecycle

1. **Submit**: Call `dstack apply -f task.yaml` with task configuration
2. **Provision**: dstack requests GPU from AMD DevCloud (~2-5 min cold start)
3. **Run**: Task executes on provisioned VM
4. **Terminate**: VM shuts down after completion or idle timeout

## Why CLI Wrapper Instead of Python SDK

Tensara uses a **CLI wrapper** (`dstack_cli_wrapper.py`) that calls dstack commands via subprocess, rather than the official Python SDK.

### SDK Bugs (Why We Didn't Use It)

The dstack Python SDK had several critical bugs that broke task status detection:

#### Bug 1: Enum-to-String Conversion

```python
# SDK code (incorrect)
str(RunStatus.DONE)  # Returns "runstatus.done" (lowercase, wrong format)

# Expected
"DONE"  # Uppercase, matches CLI output
```

This caused status polling to never detect task completion.

#### Bug 2: Terminated Status Handling

```python
# SDK behavior (incorrect)
if run.status == "terminated":
    return TaskStatus.FAILED  # Always treats as failure

# Correct behavior
if run.status == "terminated":
    if run.exit_code == 0:
        return TaskStatus.SUCCEEDED
    else:
        return TaskStatus.FAILED
```

Tasks that completed successfully with exit code 0 were incorrectly marked as failed.

#### Bug 3: Stale Cached Run Objects

The SDK caches `Run` objects, so repeated calls to `client.runs.get(run_name)` return stale data. You must access the internal `run._run` object to get fresh metadata.

#### Bug 4: Missing Metadata

Critical fields like `exit_code` were only available in the internal `run._run` object, not in the public API.

### Why CLI Wrapper is More Reliable

The CLI wrapper:

- Calls `dstack` commands directly via subprocess (no SDK bugs)
- Parses JSON output from `dstack ps --all` (deterministic)
- Retrieves logs with `dstack logs` (with retry logic)
- Easier to debug (can run commands manually)
- More predictable behavior

### SDK Attempt Files (Preserved for Reference)

The SDK-based implementation is preserved in these files:

| File                                    | Purpose                           |
| --------------------------------------- | --------------------------------- |
| `dstack_runner.sdk_attempt.py`          | SDK-based dstack interface        |
| `vm_orchestrator.sdk_attempt.py`        | VM pool management (built on SDK) |
| `vm_orchestrator_client.sdk_attempt.py` | Orchestrator wrapper              |

These files document:

- What we tried with the SDK
- Why it didn't work
- What bugs we encountered
- How the orchestrator was designed (for future reference)

**Do not use these files in production.** They are kept for historical reference only.

## Production Code Reference

### Core Files

| File                           | Purpose                            | Lines |
| ------------------------------ | ---------------------------------- | ----- |
| `src/pages/api/amd/submit.ts`  | API endpoint for AMD submissions   | 125   |
| `src/server/amd/runner.ts`     | Spawns Python process, streams SSE | 266   |
| `engine/amd_task_runner.py`    | Entry point, SSE event generation  | ~400  |
| `engine/dstack_cli_wrapper.py` | CLI-based dstack interface         | ~350  |
| `engine/amd_fleet_manager.py`  | Fleet lifecycle management         | ~300  |
| `engine/hip_harness.py`        | Generates benchmark harness        | ~200  |
| `engine/create_amd_fleet.py`   | One-time fleet creation script     | ~80   |
| `engine/fleet-amd-mi300x.yml`  | Fleet configuration file           | ~20   |

### Key Functions

**`amd_task_runner.py`:**

- `execute_task(payload)` - Main execution function
  - Sends SSE events: IN_QUEUE, PROVISIONING, COMPILING, CHECKING, BENCHMARKING, etc.
  - Calls `dstack_cli_wrapper` to submit task
  - Polls for completion with status updates
  - Parses output for benchmark results

**`dstack_cli_wrapper.py`:**

- `submit_task(code, config)` - Submits task to dstack
  - Ensures fleet exists via `amd_fleet_manager`
  - Creates temp directory with source code
  - Generates `.dstack.yml` config
  - Calls `dstack apply -f .dstack.yml -y -d`
- `get_task_status(task_id)` - Polls task status
  - Calls `dstack ps --all` and parses JSON
  - Maps dstack statuses to internal enum
- `get_task_logs(task_id)` - Retrieves logs
  - Calls `dstack logs <task_id>`
  - Retry logic for log availability

**`amd_fleet_manager.py`:**

- `ensure_fleet_exists()` - Ensures fleet is ready
  - Checks if fleet exists via `dstack fleet list`
  - Creates fleet if missing via `dstack apply`
  - Verifies creation with retry logic

**`hip_harness.py`:**

- `generate_hip_benchmark_harness(problem, def, dtype)` - Generates C++ harness
  - Creates `main()` function
  - Allocates input/output memory
  - Launches user's kernel
  - Measures timing (warmup + 20 iterations)
  - Prints parseable output: `Runtime: X ms`, `GFLOPS: Y`

## Environment Variables

### Required Variables

| Variable         | Default            | Description                          |
| ---------------- | ------------------ | ------------------------------------ |
| `DSTACK_TOKEN`   | _(required)_       | API token from https://sky.dstack.ai |
| `AMD_BACKEND`    | `amddevcloud`      | dstack backend name                  |
| `AMD_FLEET_NAME` | `amd-mi300x-fleet` | Fleet identifier                     |

### Optional Variables (with Defaults)

| Variable                  | Default  | Description                          |
| ------------------------- | -------- | ------------------------------------ |
| `AMD_GPU_TYPE`            | `MI300X` | GPU type to provision                |
| `AMD_GPU_MEMORY`          | `192GB`  | Required GPU memory                  |
| `AMD_FLEET_MAX_NODES`     | `1`      | Max concurrent VMs (cost protection) |
| `AMD_FLEET_IDLE_DURATION` | `10m`    | Auto-shutdown after idle time        |
| `AMD_SPOT_POLICY`         | `auto`   | Use spot instances when available    |
| `AMD_FLEET_DRY_RUN`       | `false`  | Test fleet config without creating   |

### Environment Setup

**Option 1: Using `.env` file**

Create `engine/.env`:

```bash
DSTACK_TOKEN=dstack_xxxxxxxxxxxxx
AMD_BACKEND=amddevcloud
AMD_FLEET_NAME=amd-mi300x-fleet
AMD_GPU_TYPE=MI300X
AMD_GPU_MEMORY=192GB
AMD_FLEET_MAX_NODES=1
AMD_FLEET_IDLE_DURATION=10m
AMD_SPOT_POLICY=auto
```

**Option 2: Using shell environment**

```bash
export DSTACK_TOKEN=dstack_xxxxxxxxxxxxx
export AMD_BACKEND=amddevcloud
# ... etc
```

## Modal vs dstack Comparison

Tensara supports two GPU backends:

| Aspect              | Modal (NVIDIA)                                  | dstack (AMD)                                |
| ------------------- | ----------------------------------------------- | ------------------------------------------- |
| **Execution Model** | Serverless functions (`modal.App` decorator)    | Fleet-based tasks (YAML config)             |
| **Infrastructure**  | Modal managed (https://modal.com)               | dstack.ai + AMD DevCloud                    |
| **Cold Start**      | ~5-10 seconds (memory snapshots)                | ~2-6 minutes (VM provisioning)              |
| **Warm Reuse**      | Automatic (Modal handles internally)            | 10-minute idle window (dstack fleet config) |
| **Compiler**        | NVCC (CUDA)                                     | hipcc (HIP)                                 |
| **Languages**       | CUDA, Mojo                                      | HIP                                         |
| **GPU Options**     | T4, A10G, L4, L40S, A100-80GB, H100, H200, B200 | MI300X                                      |
| **Cost Model**      | Per-second billing                              | $1.99/hour (hourly rate)                    |
| **Invocation**      | `runner.remote_gen()` via Modal SDK             | CLI wrapper + subprocess                    |
| **Code Location**   | `engine/api.py`, `engine/runner.py`             | `engine/amd_*.py`, `src/server/amd/`        |
| **Compilation**     | Local (NVCC on Modal container)                 | Remote (hipcc on AMD VM)                    |
| **Log Streaming**   | Direct generator yield (synchronous)            | Buffered polling with retry (async)         |
| **Deployment**      | `modal deploy api.py` (GitHub Actions)          | dstack fleet (one-time setup)               |

### When to Use Each Backend

**Use Modal (NVIDIA) when:**

- Fast iteration is critical (5-10s cold start)
- Working with CUDA-specific features
- Need a wide variety of GPU options (T4 → H200)
- Want serverless billing (pay per second)

**Use dstack (AMD) when:**

- Need AMD-specific hardware (MI300X)
- Working with HIP code
- Budget-conscious (lower hourly rate)
- Can tolerate 2-6 minute cold starts

## Cost Analysis

### AMD DevCloud Pricing

- **MI300X**: $1.99/hour
- **Fleet configuration**: Auto-scale from 0 (no cost when idle)
- **Max nodes**: 1 (prevents runaway costs)
- **Idle timeout**: 10 minutes (auto-shutdown)

### Cost Per Task

**Cold Start Task (First Submission):**

- Provisioning: 2-5 min
- Execution: 1-3 min
- Total: ~3-8 min
- **Cost**: ~$0.10-0.27

**Warm Reuse Task (Within 10-Min Window):**

- Compilation: 10-20 sec
- Execution: 10-30 sec
- Total: ~20-50 sec
- **Cost**: ~$0.01-0.03

**Idle Cost (VM Waiting for Next Task):**

- 10 minutes max before auto-shutdown
- **Cost**: ~$0.33 per idle period

### Cost Optimization Strategies

1. **Batch submissions**: Submit multiple problems within 10-minute window to reuse warm VM
2. **Reduce idle duration**: Set `AMD_FLEET_IDLE_DURATION=5m` for faster shutdown (but less reuse)
3. **Increase idle duration**: Set `AMD_FLEET_IDLE_DURATION=15m` for more reuse (but higher idle costs)
4. **Monitor usage**: Check `dstack ps --all` to see running VMs

**Recommended for development**: 10-minute idle duration (balances reuse vs cost)  
**Recommended for production**: 5-minute idle duration (minimize idle costs)

## Test Files and Directories

### `testing/rocm-dstack-hotaisle/`

This directory contains HIP test kernels and dstack configuration examples for testing on AMD hardware:

| File                                | Purpose                                         |
| ----------------------------------- | ----------------------------------------------- |
| `leaky_relu_test.hip`               | Leaky ReLU kernel test (correctness)            |
| `matmul.hip`                        | Matrix multiplication kernel test (performance) |
| `.dstack.yml`                       | dstack task configuration example               |
| `test-mi300x.sh`                    | Shell script to test MI300X provisioning        |
| `vm_reuse_working_shell_output.txt` | Example output showing VM reuse working         |
| `working_shell_output.txt`          | Example output from successful execution        |

**Usage:**

```bash
cd testing/rocm-dstack-hotaisle
dstack apply -f .dstack.yml
```

This submits a test task to verify MI300X provisioning and execution.

### `testing/` (Root-Level Test Files)

| File                    | Purpose                                               |
| ----------------------- | ----------------------------------------------------- |
| `test_payload.json`     | Example JSON payload for testing `amd_task_runner.py` |
| `test_dstack_output.sh` | Shell script to test dstack output parsing            |
| `test-cpu-only.yml`     | dstack config for CPU-only testing                    |
| `test-minimal.yml`      | Minimal dstack config example                         |

## Troubleshooting

### Common Issues

#### "No offers" Error

**Symptom:**

```
Error: No offers found for fleet amd-mi300x-fleet
```

**Cause:** Fleet doesn't exist or wasn't created properly.

**Solution:**

```bash
cd engine
python create_amd_fleet.py
dstack fleet list  # Verify fleet appears
```

#### "No capacity" Error

**Symptom:**

```
Error: No available capacity for MI300X GPU
```

**Cause:** AMD DevCloud has limited MI300X availability.

**Solution:**

- Wait 5-10 minutes and retry
- Check AMD DevCloud status page
- Try during off-peak hours (late night/early morning PST)

#### 10-Minute Timeout (Task Hangs)

**Symptom:** Task stuck in PROVISIONING or RUNNING for > 10 minutes with no updates.

**Cause:** This was a bug in the SDK-based implementation (status polling broken).

**Solution:** Verify you're using `dstack_cli_wrapper.py`, not `dstack_runner.sdk_attempt.py`. The CLI wrapper has proper status polling.

#### Empty Benchmark Output

**Symptom:** Task succeeds but no runtime/GFLOPS printed.

**Cause:** User submitted only a kernel function, missing `main()`.

**Solution:** The `hip_harness.py` module automatically generates a complete benchmark harness. Verify it's being called in `amd_task_runner.py`.

#### SSE Connection Drops During Provisioning

**Symptom:** Frontend loses connection during 2-6 minute provisioning wait.

**Cause:** No heartbeat sent during long polling intervals.

**Solution:** The current implementation sends heartbeat SSE events every 15 seconds. Verify `send_sse()` is being called in the polling loop.

#### Authentication Error

**Symptom:**

```
Error: Unauthorized - dstack token invalid
```

**Cause:** Missing or expired `DSTACK_TOKEN`.

**Solution:**

1. Get token from https://sky.dstack.ai (Settings → API Tokens)
2. Add to `engine/.env`: `DSTACK_TOKEN=dstack_xxxxx`
3. Restart development server

#### Task Fails with "hipcc: command not found"

**Symptom:**

```
Error: /bin/sh: hipcc: command not found
```

**Cause:** ROCm not installed on VM, or incorrect Docker image.

**Solution:** This shouldn't happen with AMD DevCloud VMs (ROCm pre-installed). If it does, check the fleet configuration in `fleet-amd-mi300x.yml` to ensure it's using AMD DevCloud backend.

### Debug Commands

**Check fleet status:**

```bash
dstack fleet list
```

**View active tasks:**

```bash
dstack ps --all
```

**Get task logs:**

```bash
dstack logs <task-id>
```

**Stop a stuck task:**

```bash
dstack stop <task-id>
```

**Delete and recreate fleet:**

```bash
dstack fleet delete amd-mi300x-fleet
cd engine && python create_amd_fleet.py
```

## Future: Unified Backend Architecture

For discussion with the team - ideas for abstracting Modal + dstack into a common interface to easily add new backends in the future.

### Vision

Currently, Tensara has two separate execution paths:

1. **Modal** → NVIDIA GPUs (serverless functions)
2. **dstack** → AMD GPUs (fleet-based tasks)

**Goal:** Create a unified `Backend` abstraction that allows adding new GPU providers (e.g., RunPod, Lambda Labs, Vast.ai) without rewriting the frontend or API layer.

### Proposed Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

class TaskStatus(Enum):
    """Universal task status across all backends"""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    COMPILING = "compiling"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskConfig:
    """Universal task configuration"""
    source_code: str
    problem_id: str
    gpu_type: str
    language: str  # "cuda", "hip", "mojo"
    dtype: str
    timeout: int = 600

@dataclass
class TaskResult:
    """Universal task result"""
    task_id: str
    status: TaskStatus
    output: str
    runtime_ms: Optional[float] = None
    gflops: Optional[float] = None
    error: Optional[str] = None
    cost_usd: Optional[float] = None

class Backend(ABC):
    """Abstract backend interface"""

    @abstractmethod
    async def submit(self, config: TaskConfig) -> str:
        """
        Submit a task for execution.
        Returns: task_id (unique identifier for polling)
        """
        pass

    @abstractmethod
    async def poll(self, task_id: str) -> TaskStatus:
        """
        Get current task status.
        Returns: TaskStatus enum value
        """
        pass

    @abstractmethod
    async def get_result(self, task_id: str) -> TaskResult:
        """
        Get final result (blocks until complete or timeout).
        Returns: TaskResult with output and metrics
        """
        pass

    @abstractmethod
    async def stream_logs(self, task_id: str) -> AsyncIterator[str]:
        """
        Stream logs as they're produced (real-time).
        Yields: log lines as strings
        """
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a running task.
        Returns: True if cancelled, False if already complete
        """
        pass

    @abstractmethod
    def supports_gpu(self, gpu_type: str) -> bool:
        """
        Check if this backend supports a given GPU type.
        Returns: True if supported
        """
        pass
```

### Implementation Sketch

#### ModalBackend

```python
class ModalBackend(Backend):
    """
    Serverless execution on Modal (NVIDIA GPUs)

    Characteristics:
    - Fast cold start (~5-10s via memory snapshots)
    - Local compilation (NVCC)
    - Direct generator-based log streaming
    - Per-second billing
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.supported_gpus = ["T4", "A10G", "L4", "L40S", "A100-80GB",
                               "H100", "H200", "B200"]

    async def submit(self, config: TaskConfig) -> str:
        # Compile locally with NVCC
        binary = self._compile_cuda(config.source_code)

        # POST to Modal endpoint with binary
        response = await self._post_to_modal(
            f"{self.endpoint}/benchmark-{config.gpu_type}",
            {"binary": binary, "config": config}
        )

        return response["task_id"]

    async def stream_logs(self, task_id: str) -> AsyncIterator[str]:
        # Modal yields logs directly from generator
        async for line in self._stream_from_modal(task_id):
            yield line
```

#### DStackBackend

```python
class DStackBackend(Backend):
    """
    Fleet-based execution on dstack (AMD GPUs, potentially others)

    Characteristics:
    - Slower cold start (~2-6 min for VM provisioning)
    - Remote compilation (hipcc on VM)
    - Poll-based log retrieval with retry
    - Hourly billing with auto-shutdown
    """

    def __init__(self, fleet_name: str, backend: str):
        self.fleet_name = fleet_name
        self.backend = backend
        self.supported_gpus = ["MI300X", "MI250X", "MI210", "MI300A"]
        self.cli_wrapper = DStackCLIWrapper()

    async def submit(self, config: TaskConfig) -> str:
        # Ensure fleet exists (dstack 2.0+ requirement)
        await self._ensure_fleet()

        # Submit task to fleet (compilation happens remotely)
        result = await self.cli_wrapper.submit_task(
            code=config.source_code,
            config=config
        )

        return result.task_id

    async def stream_logs(self, task_id: str) -> AsyncIterator[str]:
        # dstack requires polling for logs
        while True:
            logs = await self.cli_wrapper.get_task_logs(task_id)
            if logs:
                yield logs
            await asyncio.sleep(2)  # Poll interval
```

### Backend Factory

```python
class BackendFactory:
    """Factory for creating backend instances"""

    @staticmethod
    def create(gpu_type: str) -> Backend:
        """
        Create appropriate backend for given GPU type.
        Uses environment config to determine routing.
        """
        if gpu_type in ["T4", "A10G", "L4", "L40S", "A100-80GB", "H100", "H200", "B200"]:
            return ModalBackend(endpoint=os.getenv("MODAL_ENDPOINT"))

        elif gpu_type in ["MI300X", "MI250X", "MI210", "MI300A"]:
            return DStackBackend(
                fleet_name=os.getenv("AMD_FLEET_NAME"),
                backend=os.getenv("AMD_BACKEND")
            )

        else:
            raise ValueError(f"Unsupported GPU type: {gpu_type}")
```

### Usage in API Endpoint

```typescript
// Simplified API endpoint using unified backend
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { code, problemSlug, gpuType } = req.body;

  // Create appropriate backend for GPU type
  const backend = BackendFactory.create(gpuType);

  // Submit task (backend handles compilation, execution, etc.)
  const taskId = await backend.submit({
    source_code: code,
    problem_id: problemSlug,
    gpu_type: gpuType,
    language: getLanguageForGpu(gpuType),
    dtype: "float32",
  });

  // Stream logs back to frontend
  for await (const log of backend.stream_logs(taskId)) {
    res.write(`data: ${log}\n\n`);
  }

  // Get final result
  const result = await backend.get_result(taskId);
  res.write(`data: ${JSON.stringify(result)}\n\n`);
  res.end();
}
```

### Key Challenges

#### 1. Different Paradigms

**Modal:** Stateless serverless functions  
**dstack:** Stateful fleet-based VMs

**Solution:** Abstract at the task level, not infrastructure level. Each backend manages its own lifecycle internally (fleet creation, VM pooling, etc.).

#### 2. Compilation Location

**Modal:** Compiles locally (NVCC on Modal container, binary transferred)  
**dstack:** Compiles remotely (hipcc on AMD VM, source transferred)

**Solution:** `submit()` method handles this internally. Caller doesn't need to know where compilation happens.

#### 3. Log Streaming

**Modal:** Direct generator yield (synchronous)  
**dstack:** Poll-based retrieval (async with retry)

**Solution:** `stream_logs()` returns an AsyncIterator in both cases. Modal yields immediately, dstack polls with backoff.

#### 4. Cold Start Behavior

**Modal:** 5-10 seconds (memory snapshots)  
**dstack:** 2-6 minutes (VM provisioning)

**Solution:** Expose `supports_gpu()` and document cold start expectations. Frontend can show different messaging based on backend.

#### 5. Error Handling

**Modal:** Exceptions from Python generator  
**dstack:** CLI exit codes + log parsing

**Solution:** Each backend translates errors to standard `TaskResult.error` field. Caller sees consistent error interface.

### Potential Approach

1. **Phase 1**: Create `Backend` interface and `TaskConfig`/`TaskResult` types
2. **Phase 2**: Implement `ModalBackend` (wrap existing Modal code)
3. **Phase 3**: Implement `DStackBackend` (wrap existing dstack code)
4. **Phase 4**: Update API endpoints to use `BackendFactory`
5. **Phase 5**: Add new backends (RunPod, Lambda Labs, etc.)

### Benefits

- **Easy to add new providers**: Implement `Backend` interface, add to factory
- **Consistent API**: Frontend doesn't need to know about infrastructure differences
- **Testable**: Mock `Backend` for unit tests
- **Maintainable**: Changes to one backend don't affect others
- **Type-safe**: TypeScript + Python type hints ensure correctness

### Trade-offs

- **Abstraction overhead**: Some backend-specific optimizations may be harder
- **Lowest common denominator**: Interface must work for all backends
- **Migration cost**: Need to refactor existing Modal and dstack code

### Next Steps

1. Prototype `Backend` interface with Modal and dstack implementations
2. Test with small subset of GPU types
3. Measure performance impact of abstraction
4. Discuss with team: is the flexibility worth the complexity?

---

## Questions?

For issues or questions about the AMD runner:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review dstack documentation: https://dstack.ai/docs
3. Check AMD DevCloud status: https://www.amd.com/en/developer/resources/amd-devcloud.html
4. Open an issue in the Tensara repository

---

**Last Updated:** December 2024  
**dstack Version:** 0.20+  
**AMD DevCloud:** ROCm 6.0+
