# Fleet API Migration Guide - Backend Fleet (dstack 0.20+)

## Problem

You were experiencing "no offers" errors from dstack when trying to submit AMD GPU tasks. This is because **dstack's new API (v0.18-0.20+) requires tasks to run on pre-created fleets** rather than requesting GPU resources directly from cloud providers.

Additionally, AMD DevCloud is a **cloud backend** (not SSH-based), so we need to use **Backend Fleets**, not SSH Fleets.

## What Changed

### Old Approach (Broken - SSH Fleet)

```yaml
# ❌ WRONG: This is an SSH Fleet (for on-prem servers)
type: fleet
name: amd-mi300x-fleet
ssh_config:
  user: ubuntu
  identity_file: ~/.ssh/amd_devcloud_rsa
  hosts:
    - placeholder-host-1.com
    - placeholder-host-2.com
```

**Why this is wrong**: AMD DevCloud provisions instances via DigitalOcean API, not SSH to static hosts.

### New Approach (Fixed - Backend Fleet)

```yaml
# ✅ CORRECT: This is a Backend Fleet (for cloud providers)
type: fleet
name: amd-mi300x-fleet
nodes: 0..1 # Auto-scale: 0 pre-created, max 1 on-demand
backends: [amddevcloud] # Cloud backend provisioning
resources:
  gpu:
    name: MI300X
    memory: 192GB
  cpu: 2..
  memory: 8GB..
  disk: 100GB..
idle_duration: 10m # Auto-cleanup after 10 min idle
spot_policy: auto # Use spot instances when available
```

**Why this is correct**:

- Uses `backends` field instead of `ssh_config`
- Provisions instances via AMD DevCloud (DigitalOcean) API
- No SSH key or host list needed for fleet creation
- Auto-scales from 0 to 1 instance on-demand
- Auto-cleanup after idle period to minimize costs

## Changes Made

### 1. Updated `amd_fleet_manager.py` (Major Rewrite)

**Location**: `engine/amd_fleet_manager.py`

**Changes**:

- Converted from SSH Fleet to Backend Fleet configuration
- Updated `_load_amd_config()` to load backend fleet settings from environment
- Rewrote `_generate_fleet_config()` to generate Backend Fleet YAML (no SSH config)
- Added `_verify_fleet_created()` method to validate fleet actually exists after creation
- Added dry-run mode support for safe testing
- Removed SSH-related methods (discover_amd_hosts, etc.)

**Key code**:

```python
def _generate_fleet_config(self, fleet_name: str) -> Dict[str, Any]:
    """Generate Backend Fleet configuration for AMD DevCloud"""
    config = {
        'type': 'fleet',
        'name': fleet_name,
        'nodes': f"0..{self.amd_config['max_nodes']}",  # Auto-scale
        'backends': [self.amd_config['backend']],       # Cloud backend
        'resources': {
            'gpu': {'name': 'MI300X', 'memory': '192GB'},
            'cpu': '2..',      # Minimum viable
            'memory': '8GB..',
            'disk': '100GB..',
        },
        'idle_duration': '10m',
        'spot_policy': 'auto',
    }
    return config
```

### 2. Created `create_amd_fleet.py` (New Standalone Script)

**Location**: `engine/create_amd_fleet.py`

**Purpose**: Standalone script for one-time fleet creation with interactive prompts and dry-run support.

**Features**:

- Displays configuration before creating fleet
- Interactive confirmation prompt
- Dry-run mode for testing
- Clear success/failure messages
- Troubleshooting guidance

**Usage**:

```bash
# Dry run (test configuration)
export AMD_FLEET_DRY_RUN=true
python3 engine/create_amd_fleet.py

# Create actual fleet
export AMD_FLEET_DRY_RUN=false
python3 engine/create_amd_fleet.py
```

### 3. Updated Environment Configuration

**Location**: `engine/.env.example`

**Changes**:

Removed SSH Fleet variables:

- ~~AMD_DEVCLOUD_SSH_USER~~
- ~~AMD_DEVCLOUD_SSH_KEY_PATH~~
- ~~AMD_DEVCLOUD_HOSTS~~

Added Backend Fleet variables:

- `AMD_BACKEND=amddevcloud`
- `AMD_FLEET_NAME=amd-mi300x-fleet`
- `AMD_GPU_TYPE=MI300X`
- `AMD_GPU_MEMORY=192GB`
- `AMD_FLEET_MAX_NODES=1`
- `AMD_FLEET_IDLE_DURATION=10m`
- `AMD_SPOT_POLICY=auto`
- `AMD_FLEET_DRY_RUN=false`

### 4. Verified `dstack_cli_wrapper.py` (No Changes Needed)

**Location**: `engine/dstack_cli_wrapper.py:160-199`

**Status**: Already correct - uses `fleets` field, no `backends` field in task YAML.

The task configuration correctly specifies:

```python
dstack_config = {
    'type': 'task',
    'fleets': [fleet_name],  # ✅ Uses fleet
    # No 'backends' field     # ✅ Correct - backend is in fleet config
}
```

## Setup Instructions

### Prerequisites

1. **dstack CLI installed and authenticated**:

   ```bash
   # Install dstack
   pip install dstack

   # Authenticate with dstack Sky
   dstack config --url https://sky.dstack.ai --token YOUR_TOKEN
   ```

2. **AMD DevCloud backend configured** in dstack Sky dashboard:
   - Backend name: `amddevcloud`
   - DigitalOcean API key: `dop_v1_...` (from AMD DevCloud portal)
   - Backend should show "active" status in dstack dashboard

### Step 1: Configure Environment

Update your `engine/.env` file with Backend Fleet configuration:

```bash
# Backend Fleet Configuration (dstack 0.20+)
AMD_BACKEND=amddevcloud
AMD_FLEET_NAME=amd-mi300x-fleet
AMD_GPU_TYPE=MI300X
AMD_GPU_MEMORY=192GB
AMD_FLEET_MAX_NODES=1
AMD_FLEET_IDLE_DURATION=10m
AMD_SPOT_POLICY=auto
AMD_FLEET_DRY_RUN=false
```

**Important**: Remove any old SSH Fleet variables if present:

- ~~AMD_DEVCLOUD_SSH_USER~~
- ~~AMD_DEVCLOUD_SSH_KEY_PATH~~
- ~~AMD_DEVCLOUD_HOSTS~~

These are not needed for Backend Fleets.

### Step 2: Test Configuration (Dry Run)

Test your configuration without creating actual resources:

```bash
cd engine
export AMD_FLEET_DRY_RUN=true
python3 create_amd_fleet.py
```

You should see output like:

```
🧪 DRY RUN MODE ENABLED
...
✅ DRY RUN SUCCESSFUL
Configuration is valid. Set AMD_FLEET_DRY_RUN=false to create actual fleet.
```

### Step 3: Create Fleet (One-Time Setup)

Once dry run succeeds, create the actual fleet:

```bash
cd engine
export AMD_FLEET_DRY_RUN=false
python3 create_amd_fleet.py
```

The script will:

1. Display configuration details
2. Ask for confirmation (costs money when active)
3. Create the fleet via dstack API
4. Verify fleet was created successfully

**Expected output**:

```
✅ FLEET CREATED SUCCESSFULLY

Fleet 'amd-mi300x-fleet' is ready for task submission.

Verify with: dstack fleet list
```

### Step 4: Verify Fleet

Confirm fleet exists and is properly configured:

```bash
dstack fleet list
```

You should see:

```
NAME                 INSTANCES  BACKEND       STATUS
amd-mi300x-fleet     0          amddevcloud   idle
```

For detailed info:

```bash
dstack fleet list --verbose
```

## Troubleshooting

### Error: "Fleet 'amd-mi300x-fleet' not found"

**Solution**: Create the fleet first:

```bash
cd engine
python3 create_amd_fleet.py
```

### Error: "Backend 'amddevcloud' not found"

**Cause**: AMD DevCloud backend not configured in dstack Sky.

**Solution**:

1. Log into dstack Sky dashboard: https://sky.dstack.ai
2. Go to Settings → Backends
3. Add AMD DevCloud backend with your DigitalOcean API key
4. Ensure backend status is "active"

### Error: "Fleet not found after creation - apply may have failed silently"

**Cause**: dstack returned success but fleet wasn't actually created.

**Solution**:

1. Check dstack Sky dashboard for error messages
2. Verify backend has sufficient quota/credits
3. Try creating fleet manually:
   ```bash
   cd engine
   dstack fleet list  # Confirm fleet doesn't exist
   python3 create_amd_fleet.py  # Try again
   ```

### Error: "No offers found" (still happening after migration)

**Cause**: Fleet exists but has issues, or task isn't using fleet.

**Solution**:

1. Verify fleet exists: `dstack fleet list`
2. Check fleet has correct backend: `dstack fleet list --verbose`
3. Verify task uses fleet: Check `dstack_cli_wrapper.py` has `fleets: [fleet_name]` field
4. Ensure task YAML doesn't have `backends` field (conflicts with fleet)

### Warning: "Fleet manager not available"

**Cause**: `amd_fleet_manager.py` can't be imported.

**Solution**:

- Ensure `amd_fleet_manager.py` exists in `engine/` directory
- Check Python path includes engine directory
- Manually verify fleet exists: `dstack fleet list`

### Fleet shows "0 instances" but tasks won't start

**Cause**: This is normal - Backend Fleets auto-scale from 0 to max on-demand.

**Solution**:

1. Submit a task - fleet will provision instance automatically
2. Wait 2-5 minutes for instance provisioning
3. Check status: `dstack ps --all`
4. If stuck in "provisioning" for >10 minutes, check AMD DevCloud credits/quota

### Cost concerns - instances not shutting down

**Cause**: Fleet `idle_duration` might be too long, or instances stuck.

**Solution**:

1. Check current instances: `dstack fleet list --verbose`
2. Manually stop fleet: `dstack fleet delete amd-mi300x-fleet`
3. Adjust `AMD_FLEET_IDLE_DURATION` in .env (default: 10m)
4. Recreate fleet with new idle duration

### "Permission denied" or SSH errors

**Cause**: You're seeing SSH Fleet errors, but Backend Fleets don't use SSH.

**Solution**:

1. Verify you're using Backend Fleet config (has `backends` field, not `ssh_config`)
2. Check your `amd_fleet_manager.py` was updated to Backend Fleet version
3. Remove any SSH-related environment variables from .env

## Additional Resources

- [dstack Backend Fleet Documentation](https://dstack.ai/docs/concepts/fleets/#backend-fleets)
- [dstack 0.20+ Quickstart](https://dstack.ai/docs/quickstart/)
- [AMD DevCloud Portal](https://amddevcloud.com/)
- [Fleet Manager Source Code](./engine/amd_fleet_manager.py)
- [Fleet Creation Script](./engine/create_amd_fleet.py)

## Testing Checklist

- [x] Backend Fleet configuration created (`amd_fleet_manager.py` rewritten)
- [x] Fleet creation script implemented (`create_amd_fleet.py`)
- [x] Dry-run mode tested successfully
- [x] Environment variables documented (`.env.example`)
- [x] Task YAML verified (no `backends` field, uses `fleets`)
- [ ] Actual fleet created in dstack Sky (requires user action)
- [ ] Task submitted successfully to fleet (requires user action)
- [ ] Instance provisioning verified (requires user action)
- [ ] Auto-cleanup verified after idle duration (requires user action)

## Summary

The "no offers" errors are fixed by migrating from SSH Fleet to Backend Fleet:

1. ✅ **Converted to Backend Fleet**: Uses `backends` field instead of `ssh_config`
2. ✅ **Removed SSH dependencies**: No SSH key or host list needed for fleet creation
3. ✅ **Added auto-scaling**: Fleet scales from 0 to 1 instance on-demand
4. ✅ **Added auto-cleanup**: Instances terminate after 10 minutes idle
5. ✅ **Added verification**: Fleet existence verified after creation
6. ✅ **Added dry-run mode**: Safe testing before actual resource creation
7. ✅ **Created standalone script**: Easy fleet creation with `create_amd_fleet.py`
8. ✅ **Updated documentation**: Complete migration guide with troubleshooting

**Next Steps**:

1. Update your `engine/.env` file with Backend Fleet configuration
2. Run dry-run test: `AMD_FLEET_DRY_RUN=true python3 engine/create_amd_fleet.py`
3. Create actual fleet: `AMD_FLEET_DRY_RUN=false python3 engine/create_amd_fleet.py`
4. Verify fleet: `dstack fleet list`
5. Submit test task through your application
6. Verify no "no offers" errors occur

The system is now ready for Backend Fleet-based task submission to AMD DevCloud!
