# Quick Start: AMD MI300X on Tensara

This guide will get you running HIP kernels on AMD MI300X GPUs in under 10 minutes.

## Prerequisites

- AMD DevCloud account (sign up at https://amd.digitalocean.com)
- $1,400 grant credits (you have this confirmed)
- dstack CLI installed: `pip install dstack`
- Python 3.8+ with pip

## Step 1: Get AMD DevCloud API Token (5 minutes)

1. **Login**: Go to https://amd.digitalocean.com/login
2. **Navigate to API**: Click "API" in the left sidebar
3. **Generate Token**:
   - Click "Generate New Token"
   - Name it "tensara-production"
   - Select "Full Access" (or all permissions)
4. **Copy Token**: Save the token (starts with `dop_v1_...`)

## Step 2: Configure Environment (2 minutes)

1. **Copy environment template**:

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your token**:

   ```bash
   # Find this line and add your token:
   DSTACK_AMDDEVCLOUD_API_KEY="dop_v1_your_token_here"
   ```

3. **Verify other AMD settings** (should already be configured):
   ```bash
   DSTACK_BACKEND="amddevcloud"
   AMD_VM_IDLE_TIMEOUT=600
   AMD_DEFAULT_GPU="MI300X"
   AMD_MI300X_HOURLY_RATE=1.99
   AMD_GRANT_CREDITS_TOTAL=1400.00
   ```

## Step 3: Test MI300X Provisioning (3-5 minutes)

1. **Navigate to test directory**:

   ```bash
   cd testing/rocm-dstack-hotaisle/
   ```

2. **Run test provisioning**:

   ```bash
   dstack apply -f .dstack.yml
   ```

3. **Watch for**:

   - ✅ VM provisioning (2-5 minutes)
   - ✅ "GPU MI300X (192GB VRAM, CDNA 3)"
   - ✅ "Compilation successful"
   - ✅ Kernel execution with performance output
   - ✅ "Test Complete"

4. **Test warm reuse** (within 10 minutes):
   ```bash
   # Run again immediately
   dstack apply -f .dstack.yml
   # Should start in < 10 seconds (warm VM reuse)
   ```

## Step 4: Start Development Server

```bash
# Install Python dependencies
cd engine/
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install dstack docker

# Go back to project root
cd ..

# Start Next.js server
npm install
npm run dev
```

## Step 5: Test via Web UI

1. **Open browser**: http://localhost:3000
2. **Navigate to a problem** (e.g., Matrix Multiplication)
3. **Select GPU**: Choose "AMD MI300X (192GB, $1.99/hr)"
4. **Write HIP kernel** or use starter code
5. **Submit**: Watch for VM state in console:
   - First submission: "Provisioning MI300X..." (2-5 min)
   - Next submissions: "Using warm VM" (< 10 sec)

## Step 6: Monitor Costs

```bash
# Check metrics via API
curl http://localhost:3000/api/amd/metrics

# Or via Python
cd engine/
source .venv/bin/activate
python3 -c "
from vm_orchestrator_client import get_orchestrator_metrics
import json
print(json.dumps(get_orchestrator_metrics(), indent=2))
"
```

**Expected Output**:

```json
{
  "total_cost_usd": 0.33,
  "remaining_credits_usd": 1399.67,
  "credits_used_pct": 0.0,
  "tasks_completed": 2,
  "cold_starts": 1,
  "warm_reuses": 1,
  "active_vms": 1
}
```

## Troubleshooting

### "Backend not found"

```bash
# Verify dstack configuration
dstack config

# Check if amddevcloud backend is listed
dstack ps
```

### "No GPU offers available"

```bash
# Check AMD DevCloud availability
dstack offer -b amddevcloud --gpu MI300X

# Try different regions
dstack offer -b amddevcloud
```

### "API authentication failed"

- Verify token in `.env` is correct
- Regenerate token in AMD DevCloud console if needed
- Make sure token has "Full Access" permissions

### "rocm-smi: error: argument --showmeminfo: expected at least one argument"

This error occurs when using old versions of the `.dstack.yml` file. The fix:

- Make sure you have the latest `.dstack.yml` from the repo
- The file should NOT contain `rocm-smi --showmeminfo` (without arguments)
- If you see this error, pull the latest changes or copy the fixed version

### "time: not found" or "/bin/sh: 1: time: not found"

This error occurs when the shell doesn't have the `time` command as a builtin:

- Make sure you're using the latest `.dstack.yml` file
- The current version doesn't use `time` commands
- If you need timing, use HIP events (as shown in `matmul.hip`)

### Kernel compilation fails

```bash
# Check if hipcc is available in the container
which hipcc

# Verify ROCm path
echo $ROCM_PATH

# Check HIP platform
echo $HIP_PLATFORM
```

### Python import errors

```bash
# Make sure virtual environment is activated
cd engine/
source .venv/bin/activate

# Install dependencies
pip install dstack docker
```

## What Happens Behind the Scenes

### First Request (Cold Start)

```
1. User submits HIP kernel
2. VM Orchestrator checks pool → No warm VM
3. Provisions MI300X on AMD DevCloud (2-5 min)
4. Compiles HIP kernel with hipcc
5. Executes in Docker container
6. Returns results
7. Marks VM as IDLE (10-minute timer starts)

Cost: ~$0.17 (5 minutes × $1.99/hour)
```

### Second Request (Warm Reuse)

```
1. User submits HIP kernel
2. VM Orchestrator checks pool → Found warm VM!
3. Reuses existing MI300X (< 10 sec)
4. Executes in Docker container
5. Returns results
6. Resets idle timer (10 minutes)

Cost: ~$0.03 (1 minute × $1.99/hour)
```

### After 10 Minutes Idle

```
1. Background cleanup thread checks idle VMs
2. VM idle for > 10 minutes → Shutdown
3. VM terminated, removed from pool
4. Next request will be cold start again

Cost: $0.33 (10 minutes × $1.99/hour)
```

## Performance Expectations

| Metric              | Target       | Notes                             |
| ------------------- | ------------ | --------------------------------- |
| **Cold Start**      | 2-5 minutes  | First request provisioning MI300X |
| **Warm Reuse**      | < 10 seconds | Subsequent requests within 10 min |
| **Idle Shutdown**   | 10 minutes   | Auto-cleanup of idle VMs          |
| **Cost/Task (avg)** | $0.05        | With 10:1 warm reuse ratio        |

## Cost Optimization Tips

1. **Batch submissions**: Submit multiple kernels within 10 minutes to maximize VM reuse
2. **Monitor metrics**: Check `/api/amd/metrics` to track costs
3. **Adjust idle timeout**: Increase to 15 minutes if you have bursty workloads
4. **Use grants wisely**: $1,400 = ~700 hours of MI300X = months of development

## Security Features

All user code runs in isolated Docker containers with:

- ✅ No network access
- ✅ Read-only root filesystem
- ✅ Memory limit: 16GB
- ✅ CPU limit: 4 cores
- ✅ Auto-cleanup after execution
- ✅ Cross-user isolation

## Next Steps

1. ✅ **Setup complete** - You can now run HIP kernels on MI300X
2. ⚠️ **Test thoroughly** - Run security and performance tests
3. 📝 **Monitor costs** - Track credit usage via metrics API
4. 🚀 **Deploy** - Ready for production after validation

## Support

- **Setup Issues**: See `docs/amd_devcloud_setup.md`
- **Architecture**: See `IMPLEMENTATION_SUMMARY.md`
- **AMD Support**: See `docs/amd_gpu_support.md`

---

**You're ready to go! Start with Step 1 above and you'll be running HIP kernels on MI300X in under 10 minutes.**
