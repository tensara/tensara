# Worklog — <Project / Problem Name>

> **Goal.** One sentence on the problem and what "good" looks like (throughput, latency, GFLOPs, memory BW, etc.).

## Context

- **Dataset / Input shape:** <describe>
- **Hardware:** <GPU model> | **Driver/CUDA:** <ver> | **Clock mode:** <base/boost/locked?>
- **Software:** <compiler flags>, <framework versions>
- **Metric of record:** <e.g., runtime (ms) / GFLOPS / GB/s>

---

## Baseline

**Implementation:** <naive/kernel #0 / CPU ref>  
**Key idea:** <one-liner>  
**Numbers:**  
\`\`\`
runtime: <...> ms
GFLOPs: <...>
achieved BW: <...> GB/s
\`\`\`

**Notes:**

- <what you learned / correctness checks / obvious bottlenecks>

---

## Methodology

- **Experiment control:** fixed seeds, warmups, N iters, median of K, pin freq?
- **Timing:** CUDA events / wallclock / CUPTI
- **Validation:** unit tests / diff vs reference (max abs/rel error)

---

## Profiling Snapshot

- **Roofline guess:** <compute vs memory bound?>
- **Occupancy:** <est. / profiler value>
- **Top stall reasons:** <SM busy %, warp stall reasons, mem pipe stats>
- **Memory:** L2 hit rate, DRAM throughput, global load/store efficiency

---

## Iteration Log

### v1 — Coalesced Access & Stride Fix

- **Change:** <what changed>
- **Why:** <expected effect>
- **Result:** \`<x>%\` speedup → <...> ms (from <...> ms)

### v2 — Shared Memory Tiling

- **Change:** block tiling (BxB), LD/ST pattern, avoid bank conflicts
- **Params:** B=<>, unroll=<>, vector width=<>
- **Result:** <...>

### v3 — Register Blocking & Unroll

- **Change:** <…>
- **Result:** <…>

### v4 — Asynchronous Copies / Double Buffering

- **Change:** cp.async / ping–pong smem buffers
- **Result:** <…>

### v5 — Launch Config & Occupancy Tuning

- **Change:** blockDim, gridDim, smem bytes, maxrregcount
- **Result:** <…>

> Repeat as needed. Keep each step minimal: _what changed_, _why_, _measured result_.

---

## Results

### Summary Table

| Version | Key Idea                   | Runtime (ms) | GFLOPs | Δ vs Prev |
| ------: | -------------------------- | -----------: | -----: | --------: |
|      v0 | Baseline                   |        <...> |  <...> |         — |
|      v1 | Coalesced loads            |        <...> |  <...> |   +<...>% |
|      v2 | Shared memory tiling       |        <...> |  <...> |   +<...>% |
|      v3 | Register blocking + unroll |        <...> |  <...> |   +<...>% |
|      v4 | Async copy (double buffer) |        <...> |  <...> |   +<...>% |

### Scalability Notes

- problem size sensitivity, batch size, transpose/layouts, precision

---

## Pitfalls & Debugging Notes

- <bank conflicts>, <divergence>, <alignment>, <L2 thrash>, <atomic hot spots>
- tricky correctness bug you fixed (and how)

---

## Takeaways

- <3–5 bullets of what mattered most for perf>
- what you’d try next (tensor cores? persistent kernel? fused ops?)

---

## Appendix

- **Build flags:** \`nvcc ...\`
- **Profiler commands:** \`nsys / ncu\`
- **Microbench harness:** <notes/snippet>
