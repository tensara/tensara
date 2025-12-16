# SSE Heartbeat Connection Timeout Fix

## Problem Description

**Symptom**: AMD GPU submissions would complete successfully on the backend (Python process exits with code 0, `BENCHMARKED` SSE event sent), but the frontend would show `PROVISIONING` status indefinitely and never receive the final success event.

**Root Cause**: The SSE connection was timing out and closing during long VM provisioning waits (~6 minutes), causing the final `BENCHMARKED` event to be sent to a closed connection.

## Evidence from Logs

### Backend Logs (Success):

```
2025-12-14 11:48:35,088 - __main__ - INFO - Task succeeded! Retrieving output...
2025-12-14 11:48:35,088 - __main__ - INFO - Sending SSE event: BENCHMARKED - {
  "status": "BENCHMARKED",
  "message": "Benchmark completed successfully",
  ...
}
event: BENCHMARKED
data: {"status": "BENCHMARKED", ...}

[AMD Runner] Python process exited with code: 0 for submission cmj5bus2p0002ngt7kfqomnle
```

### Frontend Console (Connection Closed):

```
[SSE] Event received: heartbeat
[SSE] Event received: heartbeat
[SSE] Event received: COMPILING
[SSE] Stream completed successfully  <-- Connection closed here
```

**Timeline Analysis**:

- Last heartbeat from API: After `COMPILING` event (~6 min into execution)
- Python sent `BENCHMARKED`: At 6 min 17 sec
- Frontend never received it: Connection already closed

## Technical Details

### SSE Connection Lifecycle

1. **Frontend** (`useSubmissionStream.ts`):

   - Opens SSE connection with 15-minute timeout: `AbortSignal.timeout(900000)` (line 378)
   - Expects regular events to keep connection alive

2. **API Endpoint** (`/api/submissions/benchmark.ts`):

   - Sets up heartbeat interval: 30 seconds (line 110)
   - Heartbeats are sent from Node.js, but Python subprocess controls the actual event stream

3. **Python Subprocess** (`amd_task_runner.py`):
   - Polls dstack status every 5 seconds (line 220)
   - Only sent SSE events on **status changes**
   - During 6-minute provisioning: No events sent → Connection timeout

### Why 30-Second Heartbeats Failed

The API's 30-second heartbeat interval (line 110-116) **only works for Modal GPU submissions** because they respond quickly. For AMD submissions:

- Python subprocess writes directly to stdout
- Node.js API just pipes Python output to response stream
- API's heartbeat interval has no effect on Python's output

## The Fix (Updated)

### Root Cause #2: Log Pollution in SSE Stream

After implementing heartbeat fix, discovered a **second critical issue**: Python INFO logs were being written directly to the SSE stream, breaking the SSE format.

**Problem:**

```
2025-12-14 12:17:55,485 - __main__ - INFO - Step 1: Sending initial queue status
event: IN_QUEUE
data: {"status": "IN_QUEUE", "message": "..."}
```

The frontend SSE parser expects clean SSE format:

```
event: IN_QUEUE
data: {"status": "IN_QUEUE", "message": "..."}

```

But it was receiving interleaved logs and SSE events, causing parse failures and stuck "In queue" status.

### Final Solution: SSE_EVENT Marker

Implemented Option 3 - clear delimiter to separate SSE events from logs.

### Changes Made

#### 1. Python Polling Loop + Heartbeat (`engine/amd_task_runner.py`)

**Added heartbeat tracking:**

```python
last_heartbeat = time.time()
HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds to keep SSE connection alive

while poll_count < max_polls:
    time.sleep(5)
    poll_count += 1

    # Send heartbeat to keep SSE connection alive during long waits
    current_time = time.time()
    if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
        send_sse("heartbeat", {"timestamp": int(current_time * 1000)})
        last_heartbeat = current_time

    # ... rest of polling logic
```

**Why 15 seconds?**

- Polling interval: 5 seconds
- Sends heartbeat every 3rd poll (5s × 3 = 15s)
- Well within typical SSE timeout limits (30-60 seconds)
- Balances network overhead vs. connection reliability

#### 2. Python SSE Event Output (`engine/amd_task_runner.py`, lines 46-58)

**Added SSE_EVENT: prefix to all SSE events:**

```python
def send_sse(event: str, data: Dict[str, Any]) -> None:
    """Send SSE event to stdout with SSE_EVENT marker for filtering"""
    logger.info(f"Sending SSE event: {event} - {json.dumps(data, indent=2)}")
    # Use SSE_EVENT: prefix to distinguish from regular logs
    print(f"SSE_EVENT:event: {event}", flush=True)
    print(f"SSE_EVENT:data: {json.dumps(data)}", flush=True)
    print("SSE_EVENT:", flush=True)
```

**Output format:**

```
SSE_EVENT:event: PROVISIONING
SSE_EVENT:data: {"status": "PROVISIONING", "message": "..."}
SSE_EVENT:
```

#### 3. Node.js Runner Filtering (`src/server/amd/runner.ts`, lines 98-154)

**Filter only SSE_EVENT: prefixed lines and strip prefix:**

```typescript
for (const line of lines) {
  if (line.trim()) {
    // Filter and process only SSE events (lines starting with SSE_EVENT:)
    if (line.startsWith("SSE_EVENT:")) {
      // Strip the SSE_EVENT: prefix and write to response
      const sseContent = line.substring("SSE_EVENT:".length);
      res.write(sseContent + "\n");

      // Parse for database updates...
    }
    // Non-SSE lines (Python logs) are not written to response, only logged to console
  }
}
```

**Result:**

- Python INFO logs → Only to Node.js console (for debugging)
- SSE events → Stripped of prefix and sent to frontend as clean SSE format
- Frontend receives clean, parseable SSE stream

#### 4. Frontend SSE Handler (`src/hooks/useSubmissionStream.ts`, lines 135-138)

**Added silent heartbeat consumption:**

```typescript
// Silently consume heartbeat events to keep connection alive
if (eventName === "heartbeat" || status === "heartbeat") {
  return;
}

console.log(`[SSE] Event received: ${eventName || status}`, {
  status,
  data: JSON.stringify(data).substring(0, 200),
});
```

**Why silent?**

- Heartbeats are infrastructure, not user-facing events
- Reduces console noise (previously logged "Event received: heartbeat" every 15s)
- Still processes the event (keeps connection alive) but doesn't log it

## Expected Behavior After Fix

### During 6-Minute Provisioning:

**Before Fix:**

```
[SSE] Event received: PROVISIONING
[6 minutes of silence]
[SSE] Stream completed successfully  <-- Connection timeout
```

**After Fix:**

```
[SSE] Event received: PROVISIONING
[15 seconds - heartbeat sent silently]
[15 seconds - heartbeat sent silently]
[15 seconds - heartbeat sent silently]
... (24 heartbeats over 6 minutes)
[SSE] Event received: COMPILING
[SSE] Event received: BENCHMARKING
[SSE] Event received: BENCHMARKED  <-- Success!
```

### Verification Steps

1. **Submit Test Kernel**:

   ```bash
   # Navigate to http://localhost:3000/problems/leaky-relu
   # Submit the Leaky ReLU kernel with MI300X GPU
   ```

2. **Monitor Console** (should see):

   ```
   [SSE] Event received: IN_QUEUE
   [SSE] Event received: PROVISIONING
   [Silent heartbeats every 15s]
   [SSE] Event received: COMPILING
   [SSE] Event received: BENCHMARKING
   [SSE] Event received: BENCHMARKED
   ```

3. **Check Terminal** (should see):

   ```
   event: heartbeat
   data: {"timestamp": 1702558935000}

   event: heartbeat
   data: {"timestamp": 1702558950000}

   event: BENCHMARKED
   data: {"status": "BENCHMARKED", ...}
   ```

4. **Database Status** (should be):
   ```sql
   SELECT status, runtime, gflops FROM submissions
   WHERE id = 'cmj5bus2p0002ngt7kfqomnle';
   -- status: ACCEPTED
   -- runtime: 0 (parsed from output)
   -- gflops: NULL
   ```

## Related Issues

### Issue 1: Missing Output Data

The logs show:

```
Output received (0 characters)
Output parsing completed: False
avg_runtime_ms: 0
avg_gflops: null
```

**Cause**: The kernel execution completes but dstack doesn't capture stdout output from the Docker container.

**Status**: Separate issue - does not affect success detection
**Impact**: Results show "BENCHMARKED" with 0 ms runtime (misleading but not fatal)
**Fix Required**: Investigate dstack logs capture mechanism

### Issue 2: PostgreSQL Connection Warnings

```
prisma:error Error in PostgreSQL connection: Error { kind: Closed, cause: None }
```

**Status**: Unrelated to SSE heartbeat issue
**Impact**: Harmless warning, connection pool recovers automatically
**Fix Required**: Adjust Prisma connection pool settings

## Testing Completed

- [x] Unit tests for status detection (`test_status_fix.py`)
- [x] Code changes implemented and verified
- [ ] **End-to-end web UI test** (NEXT STEP)

## Next Steps

1. **Test via Web UI**:

   ```bash
   npm run dev
   # Navigate to http://localhost:3000/problems/leaky-relu
   # Submit Leaky ReLU kernel with MI300X GPU
   # Expected: Completes successfully with BENCHMARKED status
   ```

2. **Verify Heartbeat Timing**:

   - Monitor browser console for SSE events
   - Should see heartbeats every ~15 seconds during provisioning
   - Should NOT see "[SSE] Event received: heartbeat" (silently consumed)

3. **Check Second Submission** (VM Reuse):
   - Submit again within 10 minutes
   - Should complete in < 30 seconds (warm VM)
   - Heartbeats may not be needed if VM is ready quickly

## Files Modified

```
tensara-app/
├── engine/
│   └── amd_task_runner.py         # Added heartbeat + SSE_EVENT: prefix
├── src/
│   ├── server/
│   │   └── amd/
│   │       └── runner.ts          # Filter SSE_EVENT: lines only
│   └── hooks/
│       └── useSubmissionStream.ts # Silent heartbeat consumption
└── docs/
    └── SSE_HEARTBEAT_FIX.md      # This document
```

## Commit Message

```
fix: Add SSE heartbeats and clean SSE stream filtering for AMD GPU submissions

Root Cause #1: SSE connection timeout during long provisioning
- Add 15-second heartbeat interval in Python polling loop
- Silently consume heartbeat events in frontend

Root Cause #2: Python logs polluting SSE stream
- Add SSE_EVENT: prefix to distinguish SSE events from logs
- Filter only SSE_EVENT: lines in Node.js runner
- Strip prefix before sending to frontend

Result: Clean SSE stream, connection stays alive, frontend updates correctly

Tested: Ready for end-to-end validation
```
