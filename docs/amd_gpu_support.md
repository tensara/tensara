# AMD GPU Support via dstack.ai

## Overview

Tensara now supports AMD GPU execution (MI210, MI250X, MI300A, MI300X) through a serverless task-based architecture powered by [dstack.ai](https://dstack.ai). This enables on-demand GPU provisioning without managing infrastructure or incurring idle costs.

## Architecture

```
Frontend (SubmissionForm.tsx)
    ↓ (User selects MI* GPU)
API Endpoint (/api/amd/submit.ts)
    ↓ (Spawns Python runner)
Task Runner (amd_task_runner.py)
    ↓ (Creates dstack task)
dstack SDK
    ↓ (Provisions VM via backend)
AMD GPU (Hot Aisle / RunPod)
    ↓ (Compiles HIP, executes)
Results (SSE stream back to frontend)
```

**Key Characteristics:**
- **Serverless**: VMs are provisioned on-demand and terminated immediately after task completion
- **No infrastructure management**: dstack handles VM lifecycle, networking, and cleanup
- **Cost-efficient**: Only pay for actual execution time (2-5 minutes per task)
- **Supports multiple backends**: Hot Aisle, RunPod, AWS, GCP, Azure

## How It Works

1. **User selects AMD GPU** (MI210, MI250X, MI300A, MI300X) in submission form
2. **Frontend routes to `/api/amd/submit`** instead of Modal endpoint
3. **API spawns Python runner** ([`amd_task_runner.py`](../engine/amd_task_runner.py)) as subprocess
4. **Python runner creates dstack task**:
   - Injects HIP kernel code
   - Configures GPU requirements
   - Submits to dstack SDK
5. **dstack provisions VM**:
   - Allocates AMD GPU from configured backend
   - Loads ROCm-enabled Docker image
   - Compiles HIP code with `hipcc`
   - Executes kernel
6. **Results stream back** via Server-Sent Events (SSE):
   - `PROVISIONING` → `COMPILING` → `CHECKING`/`BENCHMARKING` → `CHECKED`/`BENCHMARKED`
   - Execution time: 2-5 minutes (cold start)
   - VM terminates immediately after completion

## Setup

### Prerequisites

- Python 3.8+ with pip
- dstack CLI installed
- Hot Aisle account (or other dstack-supported backend)
- Environment variables configured

### Installation

1. **Install dstack CLI:**
   ```bash
   pip install dstack
   ```

2. **Configure dstack:**
   ```bash
   dstack config
   ```
   
   Follow prompts to configure:
   - Server URL (default: `https://sky.dstack.ai`)
   - Access token (from dstack.ai dashboard)
   - Workspace name

3. **Add backend (Hot Aisle example):**
   ```bash
   dstack config add-backend hotaisle \
     --api-key YOUR_HOTAISLE_API_KEY \
     --project-id YOUR_PROJECT_ID \
     --region us-east-1
   ```

### Environment Variables

Add to `.env` (see [`.env.example`](../.env.example)):

```bash
# dstack Configuration
DSTACK_TOKEN="your-token-from-dstack-dashboard"
DSTACK_WORKSPACE="your-workspace-name"
DSTACK_SERVER_URL="https://sky.dstack.ai"

# Backend Configuration
DSTACK_BACKEND="hotaisle"
DSTACK_HOTAISLE_API_KEY="your-hotaisle-api-key"
DSTACK_HOTAISLE_PROJECT_ID="your-project-id"
DSTACK_HOTAISLE_REGION="us-east-1"

# Optional
PYTHON_PATH="python3"  # Path to Python interpreter
```

## API Endpoints

### POST `/api/amd/submit`

Main endpoint for AMD GPU task submission.

**Request Body:**
```json
{
  "solution_code": "// HIP kernel code",
  "problem": "matmul",
  "problem_def": "Matrix multiplication problem",
  "gpu_type": "MI300X",
  "dtype": "float32",
  "language": "hip",
  "endpoint": "checker"
}
```

**Response:** SSE stream with events:
- `IN_QUEUE`: Task queued
- `PROVISIONING`: VM provisioning (1-3 min)
- `COMPILING`: HIP compilation with hipcc
- `CHECKING`/`BENCHMARKING`: Kernel execution
- `CHECKED`/`BENCHMARKED`: Results with metrics
- `ERROR`: Failure with details

**Example SSE Events:**
```
event: PROVISIONING
data: {"status":"PROVISIONING","message":"Provisioning MI300X GPU..."}

event: COMPILING
data: {"status":"COMPILING","message":"Compiling HIP kernel with hipcc..."}

event: CHECKED
data: {"status":"CHECKED","passed_tests":5,"total_tests":5,"execution_time":120,"cost_usd":0.05}
```

## Supported GPUs

| GPU | Memory | Compute | Typical Cost | Availability |
|-----|--------|---------|--------------|--------------|
| MI210 | 64 GB | CDNA 2 | ~$1.50/hr | Hot Aisle |
| MI250X | 128 GB | CDNA 2 | ~$3.00/hr | Hot Aisle |
| MI300A | 128 GB | CDNA 3 | ~$4.00/hr | Hot Aisle |
| MI300X | 192 GB | CDNA 3 | ~$5.00/hr | Hot Aisle |

**Notes:**
- Costs are per-execution (2-5 minutes), not per-hour
- Spot instances used automatically for cost savings
- No idle costs - VMs terminate after task completion

## Testing

### Quick Test with Working Example

The [`testing/rocm-dstack-hotaisle/`](../testing/rocm-dstack-hotaisle/) directory contains a verified working example:

```bash
cd testing/rocm-dstack-hotaisle/

# Test dstack task locally
dstack apply .

# This will:
# 1. Provision MI300X GPU (or modify .dstack.yml for different GPU)
# 2. Compile matmul.hip with hipcc
# 3. Execute kernel
# 4. Return results
# 5. Terminate VM
```

**Files:**
- [`.dstack.yml`](../testing/rocm-dstack-hotaisle/.dstack.yml): Task configuration
- [`matmul.hip`](../testing/rocm-dstack-hotaisle/matmul.hip): Example HIP kernel

### End-to-End Test via API

```bash
# From project root
curl -X POST http://localhost:3000/api/amd/submit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "solution_code": "$(cat testing/rocm-dstack-hotaisle/matmul.hip)",
    "problem": "matmul",
    "problem_def": "Matrix multiplication",
    "gpu_type": "MI300X",
    "dtype": "float32",
    "language": "hip",
    "endpoint": "checker"
  }'
```

## Trade-offs

### Advantages
✅ **Zero infrastructure management** - No VMs to maintain  
✅ **No idle costs** - Pay only for execution time  
✅ **Auto-scaling** - Unlimited concurrent tasks  
✅ **Multi-backend support** - Switch providers easily  
✅ **Simple setup** - Just configure dstack CLI  

### Disadvantages
❌ **Cold start latency** - 2-5 minutes per task  
❌ **Higher per-execution cost** - vs. persistent VM  
❌ **Backend dependency** - Requires Hot Aisle/RunPod/cloud access  
❌ **Limited control** - No custom VM configuration  

### When to Use
- **Good for**: Occasional testing, burst workloads, prototyping, cost-conscious development
- **Not ideal for**: Continuous development, sub-second latency requirements, high-frequency testing

## Implementation Files

### Core Files (Essential)
- [`engine/amd_task_runner.py`](../engine/amd_task_runner.py): Python runner that submits tasks to dstack
- [`engine/dstack_runner.py`](../engine/dstack_runner.py): dstack SDK wrapper for task management
- [`src/pages/api/amd/submit.ts`](../src/pages/api/amd/submit.ts): API endpoint that spawns Python runner
- [`testing/rocm-dstack-hotaisle/.dstack.yml`](../testing/rocm-dstack-hotaisle/.dstack.yml): Task configuration template
- [`testing/rocm-dstack-hotaisle/matmul.hip`](../testing/rocm-dstack-hotaisle/matmul.hip): Example HIP kernel

### Modified Routes (Frontend Integration)
- [`src/pages/api/submissions/checker.ts`](../src/pages/api/submissions/checker.ts): Routes AMD GPUs to `/api/amd/submit`
- [`src/pages/api/submissions/benchmark.ts`](../src/pages/api/submissions/benchmark.ts): Routes AMD GPUs to `/api/amd/submit`
- [`src/pages/api/submissions/sample.ts`](../src/pages/api/submissions/sample.ts): Routes AMD GPUs to `/api/amd/submit`
- [`src/pages/api/submissions/sandbox.ts`](../src/pages/api/submissions/sandbox.ts): Routes AMD GPUs to `/api/amd/submit`

### Type Definitions
- [`src/types/submission.ts`](../src/types/submission.ts): Added AMD GPU types to submission interfaces
- [`src/components/problem/Console.tsx`](../src/components/problem/Console.tsx): SSE event handling for AMD tasks
- [`src/components/problem/SubmissionForm.tsx`](../src/components/problem/SubmissionForm.tsx): GPU selection UI with AMD support

### Configuration
- [`.env.example`](../.env.example): Environment variable template with dstack configuration
- [`.dstack/profiles.yml`](../.dstack/profiles.yml): dstack profile configuration
- [`.dstack/tasks/rocm-runner.yaml`](../.dstack/tasks/rocm-runner.yaml): Task template configuration

## Troubleshooting

### Common Issues

**1. `dstack` command not found**
```bash
pip install dstack
# Verify installation
dstack version
```

**2. Authentication failure**
```bash
# Reconfigure dstack
dstack config
# Or set environment variables directly in .env
```

**3. Backend not configured**
```bash
# List configured backends
dstack config list-backends
# Add backend
dstack config add-backend hotaisle --api-key KEY --project-id ID
```

**4. Task timeout**
- Default timeout: 10 minutes
- Check backend capacity: `dstack run --help`
- Try different region or GPU type

**5. Compilation errors**
- Verify HIP syntax (use `hipcc --version` locally)
- Check ROCm image compatibility in [`.dstack.yml`](../testing/rocm-dstack-hotaisle/.dstack.yml)

### Debug Mode

Enable verbose logging:
```bash
# In Python runner
export DSTACK_LOG_LEVEL=DEBUG

# Test task submission
python3 engine/amd_task_runner.py '{"solution_code":"...","problem":"test","gpu_type":"MI300X"}'
```

## Future Enhancements

- [ ] Add support for RunPod backend
- [ ] Implement task result caching
- [ ] Add pre-warming for faster cold starts
- [ ] Support multi-GPU tasks
- [ ] Add cost estimation before execution

## Resources

- [dstack Documentation](https://dstack.ai/docs)
- [Hot Aisle GPU Cloud](https://hotaisle.ai)
- [ROCm Documentation](https://rocm.docs.amd.com)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)