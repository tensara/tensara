/**
 * amdVmStatus.ts
 *
 * LocalStorage utilities for tracking AMD VM warmth status.
 * Used to provide better UX feedback about expected wait times
 * for AMD GPU submissions.
 */

const AMD_LAST_RUN_KEY = "tensara_amd_last_run_timestamp";
const VM_WARM_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes - VM idle timeout

// Over-estimated times to avoid progress bar stalling at 95%
const WARM_VM_ESTIMATE_SECONDS = 60; // Actual: ~10-30s, padded for safety
const COLD_VM_ESTIMATE_SECONDS = 420; // Actual: ~5-7 min (7 min estimate)

/**
 * Save the current timestamp as the last AMD run time.
 * Call this after a successful AMD submission completes.
 */
export function saveAmdRunTimestamp(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(AMD_LAST_RUN_KEY, Date.now().toString());
  } catch {
    // localStorage may be unavailable (e.g., private browsing)
    console.warn("[amdVmStatus] Failed to save AMD run timestamp");
  }
}

/**
 * Get the timestamp of the last AMD run.
 * Returns null if no previous run or localStorage unavailable.
 */
export function getLastAmdRunTimestamp(): number | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(AMD_LAST_RUN_KEY);
    if (!stored) return null;
    const timestamp = parseInt(stored, 10);
    return isNaN(timestamp) ? null : timestamp;
  } catch {
    return null;
  }
}

/**
 * Check if the AMD VM is likely still warm (last run < 10 minutes ago).
 * Returns true if the VM should be warm, false otherwise.
 */
export function isVmLikelyWarm(): boolean {
  const lastRun = getLastAmdRunTimestamp();
  if (lastRun === null) return false;

  const elapsed = Date.now() - lastRun;
  return elapsed < VM_WARM_TIMEOUT_MS;
}

/**
 * Get the estimated wait time in seconds based on VM warmth.
 * Returns a conservative estimate to avoid progress bar stalling.
 */
export function getEstimatedWaitTime(): number {
  return isVmLikelyWarm() ? WARM_VM_ESTIMATE_SECONDS : COLD_VM_ESTIMATE_SECONDS;
}

/**
 * Get the time remaining until the VM goes cold (in seconds).
 * Returns 0 if already cold or no previous run.
 */
export function getTimeUntilVmCold(): number {
  const lastRun = getLastAmdRunTimestamp();
  if (lastRun === null) return 0;

  const elapsed = Date.now() - lastRun;
  const remaining = VM_WARM_TIMEOUT_MS - elapsed;
  return remaining > 0 ? Math.ceil(remaining / 1000) : 0;
}

/**
 * Clear the stored AMD run timestamp.
 * Useful for testing or resetting state.
 */
export function clearAmdRunTimestamp(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(AMD_LAST_RUN_KEY);
  } catch {
    // Ignore errors
  }
}
