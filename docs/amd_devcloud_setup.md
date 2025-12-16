# AMD DevCloud Setup Guide

## Step 1: Obtain AMD DevCloud API Token

### Prerequisites

- AMD DevCloud account at https://amd.digitalocean.com
- $1,400 grant credits (confirmed)

### Generate API Token

1. **Login to AMD DevCloud Console**

   - Navigate to: https://amd.digitalocean.com/login
   - Login with your account credentials

2. **Navigate to API Section**

   - Look for **"API"** in the left sidebar navigation
   - Click on **"API"** or **"Tokens/Keys"**

3. **Generate New Token**

   - Click **"Generate New Token"** button
   - Name it: `tensara-production`

4. **Select Required Scopes**

   Enable these permissions for dstack to work:

   ```
   ✅ account:read
   ✅ droplet:create
   ✅ droplet:read
   ✅ droplet:update
   ✅ droplet:delete
   ✅ project:create
   ✅ project:read
   ✅ project:update
   ✅ project:delete
   ✅ ssh_key:create
   ✅ ssh_key:read
   ✅ ssh_key:update
   ✅ ssh_key:delete
   ✅ regions:read
   ✅ sizes:read
   ```

   **Note**: If you see a simpler option like **"Full Access"** or **"Read & Write"**, select that.

5. **Copy the Token**

   - **IMPORTANT**: Copy the generated token immediately
   - It will look something like: `dop_v1_abc123def456...`
   - Store it securely - you won't be able to see it again!
   - If you lose it, you'll need to generate a new one

6. **Save to Environment Variables**

   ```bash
   # Copy .env.example to .env
   cp .env.example .env

   # Edit .env and add your token
   DSTACK_AMDDEVCLOUD_API_KEY="dop_v1_your_token_here"
   ```

## Step 2: Configure dstack Backend

### Option A: Using Environment Variables (Recommended)

Update your `.env` file:

```bash
# AMD DevCloud Configuration
DSTACK_BACKEND="amddevcloud"
DSTACK_AMDDEVCLOUD_API_KEY="dop_v1_your_token_here"
DSTACK_AMDDEVCLOUD_PROJECT_NAME="tensara-gpu-compute"
```

### Option B: Using dstack Config File

Create or update `~/.dstack/server/config.yml`:

```yaml
projects:
  - name: tensara
    backends:
      # Primary: AMD DevCloud (MI300X @ $1.99/hour)
      - type: amddevcloud
        project_name: tensara-gpu-compute
        creds:
          type: api_key
          api_key: ${DSTACK_AMDDEVCLOUD_API_KEY}

      # Fallback: Hot Aisle (COMMENTED - enable if needed)
      # - type: hotaisle
      #   team_handle: tensara-team
      #   creds:
      #     type: api_key
      #     api_key: ${DSTACK_HOTAISLE_API_KEY}
```

## Step 3: Test AMD DevCloud Connection

Run these commands to verify the setup:

```bash
# 1. Verify dstack configuration
dstack config

# 2. Check if AMD DevCloud backend is configured
dstack ps

# 3. List available GPU offers on AMD DevCloud
dstack offer -b amddevcloud --gpu MI300X

# 4. Check regions and availability
dstack offer -b amddevcloud
```

**Expected Output**:

- Backend `amddevcloud` appears in list
- MI300X GPU offers shown with $1.99/hour pricing
- Multiple regions available (nyc1, sfo3, etc.)

## Step 4: Test MI300X Provisioning

Once credentials are configured, test provisioning:

```bash
cd testing/rocm-dstack-hotaisle/

# Test provisioning MI300X
dstack apply -f .dstack.yml

# Follow logs
dstack logs -f
```

**What to Observe**:

1. **Provisioning time**: Should be 2-5 minutes
2. **GPU detection**: `rocm-smi` should show MI300X
3. **Compilation**: `hipcc` should compile successfully
4. **Execution**: Kernel runs and shows GFLOPS
5. **Cost**: Track actual hourly cost ($1.99/hour expected)

## Troubleshooting

### Issue: "Backend not found"

```bash
# Check configured backends
dstack ps

# Restart dstack server if using self-hosted
dstack server restart
```

### Issue: "No GPU offers available"

```bash
# Check AMD DevCloud availability
dstack offer -b amddevcloud

# Try different GPU types or regions
dstack offer -b amddevcloud --gpu MI210
```

### Issue: "API authentication failed"

- Verify API key has correct scopes
- Regenerate key in AMD DevCloud console
- Update `.env` file and restart server

### Issue: "SSH connection timeout"

- AMD DevCloud provisions VMs with public IPs
- Ensure dstack server can reach provisioned VMs
- Check firewall rules in AMD DevCloud console

## Next Steps

Once AMD DevCloud is configured and tested:

1. ✅ Phase 0 complete - Credentials working
2. ✅ Phase 1 complete - MI300X provisioning verified
3. → Move to Phase 2 - Build VM Orchestrator

## Support

- AMD DevCloud Console: https://amd.digitalocean.com
- dstack Documentation: https://dstack.ai/docs
- Tensara Issues: https://github.com/your-org/tensara/issues
