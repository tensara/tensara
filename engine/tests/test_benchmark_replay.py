"""Reproducing test: the benchmark checksum does not catch compute-once/copy-replay.

Background
----------
PR #172 ("Prevent static variable cheats via library isolation") aimed to defeat
submissions that detect the second call and skip the work. It was closed in favour of
the lighter PR #174 ("add checksum during benchmarking"), which is what ships today:
before each timed iteration `run_dynamic_benchmark` fills every output tensor with 1.0
and, after calling the solution, flags WRONG_ANSWER only if the output is *unchanged*
(`sum().item()` identical to the 1.0 fill).

That guard only proves the output was written, not that it was computed. Because the
benchmark reuses one input/output allocation for the warm-up and every timed iteration,
a submission can compute the real answer once during warm-up, stash it in a persistent
(`__device__`/static) buffer, and merely copy the cached bytes back on every timed
iteration. The output differs from the 1.0 fill, so the checksum passes -- but the
measured time is a memcpy, not the kernel. The L2 flush between iterations does not help
(the cache lives in global memory and the copy is still far cheaper than recompute).

This test drives the real `run_dynamic_benchmark` with such a replay "solution" and
asserts the property we *want* -- that replay is rejected. It is marked `xfail(strict)`
because today it is not: the exploit is benchmarked as if it did the work. When a fix
lands, this test will XPASS and (being strict) fail CI, prompting removal of the marker.

Requires a CUDA device (it exercises the real benchmark path); skipped otherwise.
"""

import pytest
import torch

import utils
from problem import Problem

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="benchmark path requires a CUDA device"
)


class double_elementwise(Problem):
    """out = 2 * in -- a trivial, deterministic, compute-light elementwise problem.

    Inputs are generated from a fixed per-case seed, exactly as real problems are, so
    every benchmark iteration sees the same input allocation.
    """

    is_exact = False

    parameters = [
        {"name": "inp", "type": "float", "pointer": True, "const": True},
        {"name": "out", "type": "float", "pointer": True, "const": False},
        {"name": "n", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(name="double-elementwise")

    def reference_solution(self, inp: torch.Tensor) -> torch.Tensor:
        return inp * 2

    def get_extra_params(self, test_case):
        return [test_case["n"]]

    def generate_test_cases(self):
        n = 4096
        seed = Problem.get_seed(f"{self.name}_{n}")
        return [
            {
                "name": f"{n}",
                "n": n,
                "create_inputs": lambda n=n, seed=seed: (
                    torch.rand(
                        n,
                        device="cuda",
                        dtype=torch.float32,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                    ),
                ),
            }
        ]

    def generate_sample(self):
        return self.generate_test_cases()[0]

    def verify_result(self, expected, actual):
        ok = torch.allclose(expected, actual, rtol=1e-4, atol=1e-5)
        return ok, {"message": "ok" if ok else "mismatch"}


def _run(problem, solution_func, min_iterations=3, max_iterations=5):
    """Drive the real benchmark path for a Python `solution_func(*inputs, *outputs, *extra)`."""
    test_case = problem.generate_test_cases()[0]
    input_tensors = test_case["create_inputs"]()
    ref = problem.reference_solution(*input_tensors)
    ref_outputs = ref if isinstance(ref, (tuple, list)) else (ref,)
    actual_outputs = [torch.zeros_like(o) for o in ref_outputs]
    return utils.run_dynamic_benchmark(
        solution_func,
        problem,
        test_id=1,
        test_case=test_case,
        input_tensors=input_tensors,
        actual_outputs=actual_outputs,
        language="python",
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Benchmark checksum (#174) does not catch compute-once/copy-replay; the stronger "
        "isolation approach (#172) was closed. Remove this marker once replay is rejected."
    ),
)
def test_copy_replay_is_rejected():
    """A compute-once/copy-replay submission SHOULD be rejected as WRONG_ANSWER.

    Today it is not: the cached result differs from the 1.0 fill on every iteration, so
    the checksum passes and the memcpy is benchmarked as if it were the kernel.
    """
    cache = {}

    def replay_solution(inp, out, n):
        # First call (warm-up) does the real work once and caches the bytes; every later
        # (timed) call ignores its inputs and replays the cached result -- the exploit.
        if "out" not in cache:
            out.copy_(inp * 2)
            cache["out"] = out.clone()
        else:
            out.copy_(cache["out"])

    result = _run(double_elementwise(), replay_solution)
    assert result.get("status") == "WRONG_ANSWER", (
        "copy/replay submission was accepted and benchmarked as real work: " + repr(result)
    )


def test_honest_solution_is_benchmarked():
    """Sanity check: a genuine per-call kernel passes and produces a timed result.

    Confirms the harness itself works, so the xfail above is about the exploit slipping
    through -- not about the benchmark path being broken.
    """

    def honest_solution(inp, out, n):
        out.copy_(inp * 2)

    result = _run(double_elementwise(), honest_solution)
    assert result.get("status") != "WRONG_ANSWER", result
    assert result["runtime_ms"] > 0
