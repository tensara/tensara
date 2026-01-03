# AMD GPU Runner Documentation

Tensara's AMD runner enables HIP kernel benchmarking on **AMD MI300X GPUs** via **dstack.ai** and **AMD DevCloud**.

**Key features:**

- GPU-accurate benchmarking (identical methodology to NVIDIA)
- Real-time SSE streaming of results
- Automatic fleet management
- Submission cancellation support

---

## Local Development Setup

### Prerequisites

- **Node.js 18+** and **npm/pnpm**
- **Python 3.10+** with pip
- **dstack CLI** (installed below)

### Step 1: Install dstack CLI

```bash
pip install dstack
```

Verify installation:

```bash
dstack --version
```

### Step 2: Configure dstack

Create `~/.dstack/config.yml`:

```yaml
projects:
  - default: true
    name: Tensara
    token: insert-token-here
    url: https://sky.dstack.ai
repos: []
```

> **Note:** Get the token from a team member or create your own dstack account at https://sky.dstack.ai

Verify configuration:

```bash
dstack fleet list
```

### Step 3: Environment Variables

Add to your `.env` file (root or `engine/.env`):

```bash
# AMD Fleet Configuration (all have sensible defaults)
AMD_BACKEND=amddevcloud
AMD_FLEET_NAME=amd-mi300x-fleet
AMD_GPU_TYPE=MI300X
AMD_GPU_MEMORY=192GB
AMD_FLEET_MAX_NODES=1
AMD_FLEET_IDLE_DURATION=10m
AMD_SPOT_POLICY=auto
AMD_FLEET_DRY_RUN=false
```

### Step 4: Python Environment

```bash
cd engine
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .           # or: uv sync
```

### Step 5: Create Fleet (One-Time)

```bash
cd engine
source .venv/bin/activate
python create_amd_fleet.py
```

Verify:

```bash
dstack fleet list
# Should show: amd-mi300x-fleet with backend amddevcloud
```

### Step 6: Test It

```bash
# Start the app
npm run dev

# Navigate to any problem, select MI300X, submit HIP code
# Example: http://localhost:3000/problems/leaky-relu
```

### Setup Troubleshooting

| Issue                       | Solution                                          |
| --------------------------- | ------------------------------------------------- |
| `dstack: command not found` | Run `pip install dstack`                          |
| `No fleet found`            | Run `python create_amd_fleet.py` in `engine/`     |
| `Unauthorized`              | Check token in `~/.dstack/config.yml`             |
| `Python module not found`   | Activate venv: `source engine/.venv/bin/activate` |

---

## Architecture

### Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                           │
│                    (Problem Editor UI)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ POST /api/submissions/benchmark
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              src/pages/api/submissions/benchmark.ts             │
│              - Authenticates user                               │
│              - Routes to AMD or NVIDIA based on GPU type        │
│              - Sets up SSE streaming                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (if AMD GPU selected)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              src/server/amd/runner.ts                           │
│              - Spawns Python subprocess                         │
│              - Tracks active processes (for cancellation)       │
│              - Streams SSE events to frontend                   │
│              - Updates database with results                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Spawns
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              engine/amd_task_runner.py                          │
│              - Submits task to dstack                           │
│              - Polls for status updates                         │
│              - Parses JSON events from remote runner            │
│              - Emits SSE events (IN_QUEUE, PROVISIONING, etc.)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ via dstack CLI
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              engine/dstack_cli_wrapper.py                       │
│              - Wraps dstack CLI commands                        │
│              - submit_task(), get_task_status(), get_task_logs()│
│              - Ensures fleet exists via amd_fleet_manager.py    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ dstack apply/ps/logs
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    dstack.ai → AMD DevCloud                     │
│                    (provisions MI300X VM)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Runs on VM
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              engine/amd_remote_runner.py                        │
│              - Compiles HIP kernel with hipcc                   │
│              - Loads problem from tensara/problems repo         │
│              - Runs checker (correctness tests)                 │
│              - Runs benchmark (GPU event timing)                │
│              - Emits JSON events for parsing                    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Files

| File                                  | Status     | Purpose                                      |
| ------------------------------------- | ---------- | -------------------------------------------- |
| `src/server/amd/runner.ts`            | **ACTIVE** | Spawns Python, streams SSE, tracks processes |
| `src/pages/api/submissions/cancel.ts` | **ACTIVE** | Cancellation endpoint                        |
| `engine/amd_task_runner.py`           | **ACTIVE** | Entry point, submits to dstack, polls status |
| `engine/amd_remote_runner.py`         | **ACTIVE** | Runs on AMD VM, compiles & benchmarks        |
| `engine/dstack_cli_wrapper.py`        | **ACTIVE** | Wraps dstack CLI commands                    |
| `engine/amd_fleet_manager.py`         | **ACTIVE** | Ensures fleet exists before submission       |
| `engine/problem.py`                   | **ACTIVE** | Base Problem class (uploaded to VM)          |
| `engine/create_amd_fleet.py`          | **ACTIVE** | One-time fleet creation script               |
| `engine/fleet-amd-mi300x.yml`         | **DEAD**   | Old SSH fleet config (not used)              |
| `engine/*_sdk_attempt.py`             | **DEAD**   | Failed SDK experiments (kept for reference)  |

### Execution Timeline

**Cold Start (first submission or after 10min idle):**

1. **0-10s**: Request validation, SSE setup
2. **10s-3min**: VM provisioning (IN_QUEUE → PROVISIONING)
3. **3-4min**: HIP compilation (COMPILING)
4. **4-5min**: Checker + Benchmark (CHECKING → BENCHMARKING)
5. **5-6min**: Results streamed (BENCHMARKED)

**Warm Start (VM already running):**

1. **0-10s**: Request validation
2. **10-30s**: Compilation + execution
3. **30-45s**: Results streamed

---

## How Benchmarking Works

### AMD vs NVIDIA Parity

The AMD benchmarking implementation (`amd_remote_runner.py`) **mirrors the NVIDIA implementation** (`engine/runner.py` + `engine/utils.py`). Both use identical methodology:

| Aspect             | AMD                             | NVIDIA                          |
| ------------------ | ------------------------------- | ------------------------------- |
| **Timing**         | `torch.cuda.Event` (GPU events) | `torch.cuda.Event` (GPU events) |
| **Min iterations** | 5                               | 5                               |
| **Max iterations** | 20                              | 15                              |
| **Target CV**      | 2%                              | 2%                              |
| **FLOPS**          | `problem.get_flops(test_case)`  | `problem.get_flops(test_case)`  |
| **Verification**   | `problem.verify_result()`       | `problem.verify_result()`       |
| **Test cases**     | `problem.generate_test_cases()` | `problem.generate_test_cases()` |

> **Note:** PyTorch ROCm provides `torch.cuda.*` API compatibility, so the same code works on AMD GPUs.

### GPU Event Timing

Both AMD and NVIDIA use GPU events for accurate kernel timing:

```python
# From amd_remote_runner.py (lines 485-493)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
solution_func(*parameters)
end_event.record()
torch.cuda.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)
```

This measures actual GPU execution time, not CPU wall-clock time.

### CV Convergence

**CV = Coefficient of Variation** = `(std_dev / mean) * 100%`

The benchmark runs iterations until measurements stabilize:

1. Run at least 5 iterations
2. After each iteration, calculate CV of GFLOPS measurements
3. If CV < 2%, stop early (measurements are stable)
4. Otherwise, continue up to max iterations (20)

For long kernels (>1s), CV convergence is skipped and a fixed iteration count is used.

### FLOPS Calculation

Each problem defines its own FLOPS formula:

```python
# In problem definition (e.g., matrix_multiplication)
def get_flops(self, test_case):
    M, N, K = test_case["M"], test_case["N"], test_case["K"]
    return 2 * M * N * K  # Standard matmul FLOPS

# In benchmark
gflops = (problem.get_flops(test_case) / elapsed_seconds) / 1e9
```

Problems that don't override `get_flops()` won't report GFLOPS.

### Correctness Verification

1. Generate test inputs via `test_case["create_inputs"]()`
2. Run reference solution: `expected = problem.reference_solution(*inputs)`
3. Run user's kernel: `actual = solution_func(*parameters)`
4. Verify: `is_correct, debug_info = problem.verify_result(expected, actual, dtype)`

Each problem defines its own tolerance (e.g., relative error < 1e-5).

---

## Implementation Details

### SSE Event Flow

Events emitted during AMD submission:

```
IN_QUEUE        → Submission queued
PROVISIONING    → VM being provisioned (can take 2-5min)
COMPILING       → hipcc compiling kernel
CHECKING        → Running correctness tests
TEST_RESULT     → Individual test result (per test case)
CHECKED         → All tests passed
BENCHMARKING    → Running performance benchmarks
BENCHMARK_RESULT → Individual benchmark result (per test case)
BENCHMARKED     → Final results with avg runtime/GFLOPS

Error events:
COMPILE_ERROR   → hipcc compilation failed
WRONG_ANSWER    → Test case failed verification
RUNTIME_ERROR   → Kernel crashed during execution
ERROR           → General error
```

### Cancellation Support

Users can cancel AMD submissions during the long provisioning wait:

**Frontend:** `AmdProvisioningCard.tsx` shows cancel button with confirmation dialog

**Hook:** `useSubmissionStream.ts` exposes `cancelSubmission()` function

**API:** `POST /api/submissions/cancel` with `{ submissionId }`

**Backend:** `runner.ts` tracks active processes in `activeProcesses` Map:

```typescript
// Track process for cancellation
activeProcesses.set(submissionId, { process: pythonProcess, taskName });

// Cancel function
export function cancelAmdSubmission(submissionId: string): boolean {
  const entry = activeProcesses.get(submissionId);
  if (!entry) return false;

  entry.process.kill("SIGTERM"); // Kill Python process
  // Also calls dstack stop to terminate VM task
  return true;
}
```

**Auto-cleanup:** When client disconnects, `req.on("close")` triggers cleanup.

### Key Functions

**`amd_task_runner.py`:**

- `execute_task(payload)` - Main entry, submits to dstack, polls, streams events
- `parse_json_events(output)` - Extracts `JSON_EVENT:` lines from remote runner
- `process_remote_events(events)` - Maps remote events to SSE events

**`amd_remote_runner.py`:**

- `compile_hip_kernel(code)` - Compiles with hipcc, caches result
- `run_checker(problem, solution_func, dtype)` - Correctness testing
- `run_benchmark(problem, solution_func, dtype)` - Performance measurement
- `run_dynamic_benchmark(...)` - Single test case with CV convergence

**`dstack_cli_wrapper.py`:**

- `submit_task(...)` - Creates temp dir, writes `.dstack.yml`, runs `dstack apply`
- `get_task_status(task_name)` - Parses `dstack ps --all` output
- `get_task_logs(task_name)` - Runs `dstack logs` with retry

---

## Environment Variables

### Used Variables

| Variable                  | Default            | Description                 |
| ------------------------- | ------------------ | --------------------------- |
| `AMD_BACKEND`             | `amddevcloud`      | dstack backend name         |
| `AMD_FLEET_NAME`          | `amd-mi300x-fleet` | Fleet identifier            |
| `AMD_GPU_TYPE`            | `MI300X`           | GPU type to provision       |
| `AMD_GPU_MEMORY`          | `192GB`            | Required GPU memory         |
| `AMD_FLEET_MAX_NODES`     | `1`                | Max concurrent VMs          |
| `AMD_FLEET_IDLE_DURATION` | `10m`              | Auto-shutdown after idle    |
| `AMD_SPOT_POLICY`         | `auto`             | Spot instance policy        |
| `AMD_FLEET_DRY_RUN`       | `false`            | Test mode (no actual fleet) |

### Unused Variables (Do NOT Set)

These appear in old `.env` examples but are **not used**:

| Variable                     | Why Unused                               |
| ---------------------------- | ---------------------------------------- |
| `DSTACK_AMDDEVCLOUD_API_KEY` | dstack reads from `~/.dstack/config.yml` |
| `DSTACK_TOKEN`               | Same - use config file instead           |
| `DSTACK_BACKEND`             | Not referenced in code                   |
| `AMD_VM_IDLE_TIMEOUT`        | Only in dead SDK code                    |
| `AMD_DEFAULT_GPU`            | Not referenced                           |
| `AMD_MI300X_HOURLY_RATE`     | Only in dead SDK code                    |
| `AMD_GRANT_CREDITS_TOTAL`    | Only in dead SDK code                    |
| `AMD_DEVCLOUD_SSH_USER`      | Only in dead SSH fleet config            |
| `AMD_DEVCLOUD_SSH_KEY_PATH`  | Only in dead SSH fleet config            |
| `AMD_DEVCLOUD_HOSTS`         | Only in setup script                     |

---

## dstack Internals

### What is dstack?

**dstack** is an infrastructure orchestration tool for running AI/ML workloads on cloud GPUs. Tensara uses it to access AMD DevCloud's MI300X GPUs.

Key concepts:

- **Fleet**: A pool of GPU resources (must be created before submitting tasks)
- **Task**: A unit of work (our benchmark execution)
- **Backend**: Cloud provider (amddevcloud for AMD DevCloud)

### Why CLI Wrapper (Not SDK)

We use `dstack_cli_wrapper.py` which calls dstack CLI via subprocess, rather than the official Python SDK.

**The SDK had critical bugs:**

#### Bug 1: Enum-to-String Conversion

```python
# SDK (incorrect)
str(RunStatus.DONE)  # Returns "runstatus.done"

# Expected
"DONE"  # Uppercase
```

Status polling never detected task completion.

#### Bug 2: Terminated Status Handling

```python
# SDK (incorrect) - always treats as failure
if run.status == "terminated":
    return TaskStatus.FAILED

# Correct - check exit code
if run.status == "terminated":
    return TaskStatus.SUCCEEDED if run.exit_code == 0 else TaskStatus.FAILED
```

#### Bug 3: Stale Cached Run Objects

SDK caches `Run` objects, so repeated `client.runs.get()` returns stale data.

#### Bug 4: Missing Metadata

`exit_code` only available in internal `run._run` object, not public API.

**CLI wrapper advantages:**

- Calls `dstack` commands directly (no SDK bugs)
- Parses deterministic CLI output
- Easier to debug (run commands manually)
- Retry logic for log retrieval

### Fleet Requirements (dstack 0.20+)

In older dstack versions, you could submit tasks directly. In 0.20+, you MUST create a fleet first:

```yaml
# Step 1: Create fleet (one-time)
type: fleet
name: amd-mi300x-fleet
backends: [amddevcloud]

# Step 2: Tasks reference the fleet
type: task
fleets: [amd-mi300x-fleet]
```

`amd_fleet_manager.py` automatically ensures the fleet exists before task submission.

---

## Troubleshooting

### Common Issues

#### "No offers" / Fleet Not Found

```bash
cd engine && python create_amd_fleet.py
dstack fleet list  # Verify fleet exists
```

#### "No capacity" for MI300X

AMD DevCloud has limited availability. Try:

- Wait 5-10 minutes and retry
- Try during off-peak hours (late night PST)

#### Task Stuck in PROVISIONING

Check dstack status:

```bash
dstack ps --all
dstack logs <task-name>
```

If stuck for >10 minutes, cancel and retry:

```bash
dstack stop <task-name> -y
```

#### Compilation Errors

Check the COMPILE_ERROR event details. Common issues:

- Missing `#include` statements
- HIP API differences from CUDA
- Syntax errors

#### SSE Connection Drops

The implementation sends heartbeats every 15 seconds. If still dropping:

- Check network/proxy timeouts
- Verify `send_sse()` is being called in polling loop

### Debug Commands

```bash
# Check fleet status
dstack fleet list

# View all tasks (including completed)
dstack ps --all

# Get task logs
dstack logs <task-name>

# Stop a task
dstack stop <task-name> -y

# Force stop (abort)
dstack stop <task-name> -x -y

# Delete and recreate fleet
dstack fleet delete amd-mi300x-fleet -y
cd engine && python create_amd_fleet.py
```

---

## Dead Code Reference

These files are **not used in production** but kept for historical reference:

| File                                           | Purpose                    | Why Dead                       |
| ---------------------------------------------- | -------------------------- | ------------------------------ |
| `engine/dstack_runner.sdk_attempt.py`          | SDK-based dstack interface | SDK bugs (see above)           |
| `engine/vm_orchestrator.sdk_attempt.py`        | VM pool management         | Built on broken SDK            |
| `engine/vm_orchestrator_client.sdk_attempt.py` | Orchestrator wrapper       | Built on broken SDK            |
| `engine/fleet-amd-mi300x.yml`                  | SSH fleet configuration    | We use backend fleets, not SSH |

These files document what we tried with the SDK and why it didn't work.

---

## Future Plans

> **Note:** This section describes potential future improvements that have **NOT been implemented yet**.

### Unified Backend Architecture

Currently, Tensara has two separate execution paths:

1. **Modal** → NVIDIA GPUs (serverless functions)
2. **dstack** → AMD GPUs (fleet-based tasks)

**Goal:** Create a unified `Backend` abstraction to easily add new GPU providers.

### Proposed Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

@dataclass
class TaskConfig:
    source_code: str
    problem_id: str
    gpu_type: str
    language: str  # "cuda", "hip", "mojo"
    dtype: str

@dataclass
class TaskResult:
    task_id: str
    status: str
    runtime_ms: Optional[float] = None
    gflops: Optional[float] = None
    error: Optional[str] = None

class Backend(ABC):
    @abstractmethod
    async def submit(self, config: TaskConfig) -> str:
        """Submit task, return task_id"""
        pass

    @abstractmethod
    async def poll(self, task_id: str) -> str:
        """Get current status"""
        pass

    @abstractmethod
    async def stream_logs(self, task_id: str) -> AsyncIterator[str]:
        """Stream logs in real-time"""
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Cancel running task"""
        pass
```

### Implementation Sketch

```python
class ModalBackend(Backend):
    """NVIDIA GPUs via Modal.com"""
    supported_gpus = ["T4", "A10G", "L4", "L40S", "A100-80GB", "H100", "H200", "B200"]
    # Fast cold start (~5-10s), per-second billing

class DStackBackend(Backend):
    """AMD GPUs via dstack + AMD DevCloud"""
    supported_gpus = ["MI300X", "MI250X", "MI210"]
    # Slower cold start (~2-6min), hourly billing

class BackendFactory:
    @staticmethod
    def create(gpu_type: str) -> Backend:
        if gpu_type in ModalBackend.supported_gpus:
            return ModalBackend()
        elif gpu_type in DStackBackend.supported_gpus:
            return DStackBackend()
        raise ValueError(f"Unsupported GPU: {gpu_type}")
```

### Key Challenges

1. **Different paradigms**: Modal is serverless, dstack is fleet-based
2. **Compilation location**: Modal compiles locally, dstack compiles on VM
3. **Log streaming**: Modal yields directly, dstack requires polling
4. **Cold start**: Modal ~10s, dstack ~3min

### Potential Phases

1. Create `Backend` interface and types
2. Implement `ModalBackend` (wrap existing code)
3. Implement `DStackBackend` (wrap existing code)
4. Update API endpoints to use `BackendFactory`
5. Add new backends (RunPod, Lambda Labs, etc.)

---

**Last Updated:** January 2025
**dstack Version:** 0.20+
**AMD DevCloud:** ROCm 6.2, PyTorch 2.3.0
