# dstack AMD GPU Configuration with Instance Reuse

This directory contains the configuration for running AMD GPU tasks on dstack with efficient instance reuse on hotaisle.

## Problem Solved

Previously, running `dstack apply` multiple times would provision a new VM instance each time, leading to:
- High costs due to multiple concurrent instances
- Resource exhaustion (hotaisle running out of available VMs)
- No instance reuse even with idle_duration configured

## Solution: Task-Level Instance Reuse

The correct approach with dstack 0.19.37 and cloud providers like hotaisle is to use task-level instance management settings, **not** fleet configurations (fleets are for SSH-based infrastructure).

## File Structure

```
testing/rocm-dstack-hotaisle/
├── .dstack.yml                         # Task configuration with instance reuse
├── matmul.hip                          # HIP kernel code
└── README.md                           # This file
```

## Configuration

The key settings in [`.dstack.yml`](testing/rocm-dstack-hotaisle/.dstack.yml) are:

```yaml
type: task

# Keep instance warm for 5 minutes after task completion
idle_duration: 5m

# Reuse existing instances when possible, create new ones if needed
creation_policy: reuse-or-create

# Use spot instances for cost savings
spot_policy: auto

resources:
  gpu:
    name: MI300X
    count: 1
  cpu: 2..
  memory: 8GB..
  disk: 100GB..

backends:
  - hotaisle
```

### Key Settings Explained

- **`idle_duration: 5m`**: Instance stays allocated for 5 minutes after task completion, allowing reuse
- **`creation_policy: reuse-or-create`**: dstack will reuse an existing idle instance if available, otherwise create a new one
- **`backends: [hotaisle]`**: Specify hotaisle as the backend provider
- **`spot_policy: auto`**: Use spot instances for cost savings

## Workflow

### Run Your First Task

```bash
cd testing/rocm-dstack-hotaisle

# Run the task
dstack apply
```

**What happens:**
1. dstack provisions a new MI300X instance on hotaisle
2. Your task executes
3. Instance remains allocated for 5 minutes after completion

### Run Consecutive Tasks (Within 5 Minutes)

```bash
# Run again within 5 minutes
dstack apply

# And again...
dstack apply
```

**What happens:**
1. ✅ dstack finds the existing idle instance
2. ✅ Reuses it immediately (no provisioning delay)
3. ✅ Task executes on the warm instance
4. ✅ Instance idle timer resets to 5 minutes

### After 5 Minutes of Inactivity

If you wait more than 5 minutes:

```bash
# More than 5 minutes later
dstack apply
```

**What happens:**
1. Instance was automatically released after 5-minute idle period
2. dstack provisions a fresh instance
3. Task executes
4. Instance stays warm for another 5 minutes

## How It Prevents Multiple Instances

The `creation_policy: reuse-or-create` setting ensures:

1. **Before starting a new run**, dstack checks for existing idle instances matching your resource requirements
2. **If found**, it reuses that instance immediately
3. **If not found**, it creates a new instance
4. **Only one instance is created per resource specification**

This prevents the issue where multiple concurrent instances were being provisioned.

## Cost Optimization

### Before (Without Proper Configuration)

```
Run 1: Provision VM → Execute (30s) → Release immediately
Run 2: Provision VM → Execute (30s) → Release immediately  ← New instance!
Run 3: Provision VM → Execute (30s) → Release immediately  ← Another new instance!

Cost: (Provisioning time + 30s) × 3 runs × $1.99/hr
      ≈ (2 min + 0.5 min) × 3 × $1.99/hr ≈ $0.25
```

### After (With Instance Reuse)

```
Run 1: Provision VM → Execute (30s) → Keep warm (5 min)
Run 2: Reuse warm VM → Execute (30s) → Keep warm (5 min)  ← Same instance!
Run 3: Reuse warm VM → Execute (30s) → Keep warm (5 min)  ← Same instance!

Cost: Provisioning + (3 × 30s execution) + 5 min idle
      ≈ 2 min + 1.5 min + 5 min × $1.99/hr ≈ $0.28

But with consecutive runs within 5 minutes, you only pay the idle cost once!
```

**Key savings:**
- No multiple concurrent instances
- No repeated provisioning overhead
- Predictable maximum idle cost of ~$0.17 per 5-minute window

## Pricing Details

- **MI300X on hotaisle**: $1.99/hour
- **5 minute idle duration**: Maximum ~$0.17 idle cost per execution batch
- **Execution time**: Pay only for actual kernel runtime
- **Provisioning time**: ~2 minutes, charged at $1.99/hour ≈ $0.07

## Monitoring

### Check Running Instances

```bash
# List all active runs
dstack ps

# View details of a specific run
dstack ps <run-name>

# Watch logs in real-time
dstack logs <run-name> -f
```

### Stop a Run Early

```bash
# Stop and release the instance
dstack stop <run-name>
```

## Troubleshooting

### Problem: Multiple Instances Still Getting Created

**Diagnosis:**
- Check that `creation_policy: reuse-or-create` is set in [`.dstack.yml`](testing/rocm-dstack-hotaisle/.dstack.yml)
- Verify you're running tasks within the 5-minute idle window
- Ensure resource specifications are identical across runs

**Solution:**
```bash
# View your configuration
cat .dstack.yml

# Ensure these lines are present:
# creation_policy: reuse-or-create
# idle_duration: 5m
```

### Problem: Instance Not Releasing After 5 Minutes

**Diagnosis:**
- Check if there are any active runs: `dstack ps`
- Look for runs in "idle" state

**Solution:**
```bash
# Manually stop idle runs
dstack stop <run-name>
```

### Problem: "No Offers Available" Error

**Diagnosis:**
- hotaisle may be at capacity
- All MI300X instances may be in use

**Solution:**
1. Wait a few minutes and retry
2. Or stop other idle instances: `dstack ps` then `dstack stop <run-name>`
3. Consider using a different GPU if available: Change `name: MI300X` to `name: MI210` in [`.dstack.yml`](testing/rocm-dstack-hotaisle/.dstack.yml:17)

## Best Practices

### For Development (Frequent Testing)

Keep the 5-minute idle duration:

```yaml
idle_duration: 5m
```

This gives you a good balance between convenience and cost.

### For Production (Batch Processing)

Increase idle duration if you have predictable workloads:

```yaml
idle_duration: 15m  # For jobs that run every 10-15 minutes
```

Or decrease it to minimize idle costs:

```yaml
idle_duration: 2m  # For infrequent runs
```

### For One-Off Runs

Set idle duration to minimum:

```yaml
idle_duration: 1m
```

Or use `creation_policy: create` to force immediate release:

```yaml
creation_policy: create
idle_duration: 0m
```

## Advanced Configuration

### Different GPU Types

Modify the resources section in [`.dstack.yml`](testing/rocm-dstack-hotaisle/.dstack.yml:12):

```yaml
resources:
  gpu:
    name: MI210     # or MI250X, MI300A, MI300X
    count: 1
  cpu: 2..
  memory: 8GB..
```

### Multiple GPUs

```yaml
resources:
  gpu:
    name: MI300X
    count: 2        # Request 2 GPUs
```

### Force New Instance (No Reuse)

```yaml
creation_policy: create  # Always create new instance
idle_duration: 0m        # Release immediately after task
```

## Why Fleets Don't Work with hotaisle

**Important:** Fleet configurations in dstack 0.19.37 are designed for **SSH-based infrastructure** (on-premise servers, already provisioned VMs), not for cloud providers like hotaisle that use APIs for provisioning.

If you try to use fleet configs with hotaisle, you'll get:
```
Error: No ssh_config or nodes specified
```

For cloud providers, use the task-level settings shown above instead.

## References

- [dstack Task Configuration](https://dstack.ai/docs/reference/dstack.yml/task)
- [dstack Creation Policies](https://dstack.ai/docs/concepts/runs#creation-policy)
- [hotaisle Documentation](https://hotaisle.io/docs)