import time
import threading
from collections import deque
from functools import reduce


class GPUMonitor:
    """Monitor GPU metrics using pynvml in a separate thread during kernel execution.

    Collects temperature, clock speeds, power usage, utilization, and throttle reasons
    at a configurable sampling interval (default 5ms).
    """

    def __init__(self, device_id=0, sample_interval_ms=5):
        self.device_id = device_id
        self.sample_interval = sample_interval_ms / 1000.0  # Convert to seconds
        self.monitoring = False
        self.samples = deque()
        self.thread = None
        self.handle = None
        self.lock = threading.Lock()
        self._init_pynvml()

    def _init_pynvml(self):
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self.pynvml = pynvml
        except ImportError:
            self.handle = None
            self.pynvml = None
        except Exception:
            self.handle = None
            self.pynvml = None

    def _monitor_loop(self):
        if not self.pynvml or not self.handle:
            return

        while self.monitoring:
            try:
                timestamp = time.time()

                try:
                    sm_clock = self.pynvml.nvmlDeviceGetClockInfo(
                        self.handle, self.pynvml.NVML_CLOCK_SM
                    )
                    temp = self.pynvml.nvmlDeviceGetTemperature(
                        self.handle, self.pynvml.NVML_TEMPERATURE_GPU
                    )
                    pstate = self.pynvml.nvmlDeviceGetPerformanceState(self.handle)
                    throttle_reasons = self.pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(
                        self.handle
                    )

                    sample = {
                        "timestamp": timestamp,
                        "sm_clock_mhz": sm_clock,
                        "temp_c": temp,
                        "pstate": pstate,
                        "throttle_reasons": throttle_reasons,
                    }

                    with self.lock:
                        self.samples.append(sample)

                except Exception:
                    pass

                time.sleep(self.sample_interval)

            except Exception:
                break

    def start(self):
        """Start monitoring in a separate thread."""
        if not self.pynvml or not self.handle:
            return

        if self.monitoring:
            return

        self.monitoring = True
        with self.lock:
            self.samples.clear()

        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return collected samples."""
        if not self.monitoring:
            return []

        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

        with self.lock:
            samples = list(self.samples)
            self.samples.clear()

        return samples

    def get_samples(self):
        with self.lock:
            return list(self.samples)

    def get_samples_for_period(self, start_time, end_time, samples=None):
        if samples is None:
            with self.lock:
                samples_to_use = list(self.samples)
        else:
            samples_to_use = samples

        return [s for s in samples_to_use if start_time <= s["timestamp"] <= end_time]

    def compute_stats(self, samples):
        if not samples:
            return {
                "sample_count": 0,
                "temp_c_min": 0,
                "temp_c_max": 0,
                "temp_c_mean": 0,
                "sm_clock_mhz_min": 0,
                "sm_clock_mhz_max": 0,
                "sm_clock_mhz_mean": 0,
                "pstate_min": 0,
                "pstate_max": 0,
                "throttle_reasons_any": 0,
            }

        stats = {"sample_count": len(samples)}

        full_metrics = ["temp_c", "sm_clock_mhz"]

        for metric in full_metrics:
            values = [s[metric] for s in samples if metric in s]
            if values:
                stats[f"{metric}_min"] = min(values)
                stats[f"{metric}_max"] = max(values)
                stats[f"{metric}_mean"] = sum(values) / len(values)
            else:
                stats[f"{metric}_min"] = 0
                stats[f"{metric}_max"] = 0
                stats[f"{metric}_mean"] = 0

        pstate_values = [s["pstate"] for s in samples if "pstate" in s]
        if pstate_values:
            stats["pstate_min"] = min(pstate_values)
            stats["pstate_max"] = max(pstate_values)
        else:
            stats["pstate_min"] = 0
            stats["pstate_max"] = 0

        throttle_values = [s["throttle_reasons"] for s in samples if "throttle_reasons" in s]
        if throttle_values:
            stats["throttle_reasons_any"] = reduce(lambda a, b: a | b, throttle_values, 0)
        else:
            stats["throttle_reasons_any"] = 0

        return stats

