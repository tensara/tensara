# AMD MI300X Test Configuration - Fixed

## Issues Found and Fixed

### Issue 1: rocm-smi --showmeminfo Missing Argument

**Error**: `rocm-smi: error: argument --showmeminfo: expected at least one argument`

**Fix**: Added the required `vram` argument:

```bash
# Before (broken)
rocm-smi --showproductname --showmeminfo

# After (fixed)
rocm-smi --showproductname
rocm-smi --showmeminfo vram
```

### Issue 2: time Command Not Found

**Error**: `/bin/sh: 1: time: not found`

**Fix**: Removed the `time` command wrapper, just execute directly:

```bash
# Before (broken)
time ./matmul 1024

# After (fixed)
./matmul 1024
```

### Issue 3: working_dir Warning

**Warning**: `The working_dir is not set — using legacy default "/workflow"`

**Fix**: Added explicit working_dir to configuration:

```yaml
type: task
working_dir: /workspace # Added this line
idle_duration: 10m
```

### Issue 4: YAML String Quoting

**Error**: Special characters in strings causing YAML parsing issues

**Fix**: Simplified echo commands to avoid special characters:

```yaml
# Before (problematic)
- echo "Cost: $1.99/hour via AMD DevCloud grant"

# After (fixed)
- echo "Cost 1.99/hour via AMD DevCloud grant"
```

## Updated Test Configuration

The fixed `.dstack.yml` now includes:

- ✅ Explicit `working_dir: /workspace`
- ✅ Fixed `rocm-smi` commands with proper arguments
- ✅ Removed `time` command wrapper
- ✅ Simplified echo commands without special characters
- ✅ All commands properly formatted for YAML

## Test Again

```bash
cd testing/rocm-dstack-hotaisle/
dstack apply -f .dstack.yml
```

**Expected Output**:

```
=== AMD MI300X Test ===
GPU MI300X (192GB VRAM, CDNA 3)
Cost 1.99/hour via AMD DevCloud grant

=== ROCm Environment ===
GPU[0]          : Name: MI300X
GPU[0]          : Device Name: AMD Radeon PRO MI300X
VRAM Total Memory (B): 206158430208

HIP version: 7.1.52802-26aae437f6
AMD clang version 20.0.0git...

=== Compiling HIP Kernel ===
Compilation successful

=== Executing on MI300X ===
=== HIP Matrix Multiplication ===
Matrix dimensions: A(1024x1024) * B(1024x1024) = C(1024x1024)
Tile size: 16x16
Memory: 4.00MB per matrix

Kernel execution: 1.23 ms
Performance: 1745.32 GFLOPS

Result: PASSED ✓

=== Test Complete ===
```

## Success Criteria

After running the test, you should see:

1. ✅ No errors about missing arguments
2. ✅ ROCm version and GPU info displayed
3. ✅ Compilation successful message
4. ✅ Kernel execution completes
5. ✅ Performance metrics (GFLOPS)
6. ✅ Verification passed

## Next Steps

Once the test passes:

1. Test warm VM reuse (run again within 10 minutes)
2. Verify idle shutdown (wait 10+ minutes, check if VM terminates)
3. Integrate with web API
4. Start development!

## Troubleshooting

### If compilation fails

- Check if hipcc is in PATH: `which hipcc`
- Verify ROCm installation: `ls /opt/rocm`

### If execution fails

- Check GPU is visible: `rocm-smi`
- Verify HIP runtime: `hipcc --version`

### If VM won't provision

- Check backend: `dstack offer -b amddevcloud`
- Verify API credentials in `.env`
