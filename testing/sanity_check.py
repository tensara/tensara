#!/usr/bin/env python3
"""
Tensara CI Sanity Check
-----------------------
Submits known-good CUDA solutions to /api/submissions/sample and verifies
they all pass. Catches regressions in the checker/engine.

Usage:
    python ci/sanity_check.py                              # run all
    CI_SAMPLE_SIZE=5 python ci/sanity_check.py             # random sample
    CI_PROBLEMS=conv-1d,softmax python ci/sanity_check.py  # specific slugs
    TENSARA_URL=http://localhost:3000 python ci/sanity_check.py  # local dev
"""

import os
import sys
import json
import random
import time
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY  = os.environ.get("TENSARA_CI_API_KEY", "")
BASE_URL = os.environ.get("TENSARA_URL", "https://tensara.org")
GPU_TYPE = os.environ.get("CI_GPU_TYPE", "T4")
LANGUAGE = os.environ.get("CI_LANGUAGE", "cuda")
SAMPLE   = int(os.environ.get("CI_SAMPLE_SIZE", "0"))  # 0 = all
SPECIFIC = os.environ.get("CI_PROBLEMS", "")           # comma-separated slugs
TIMEOUT  = int(os.environ.get("CI_TIMEOUT_SECONDS", "300"))

SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solutions")
ENDPOINT      = f"{BASE_URL}/api/submissions/sample"

TERMINAL_PASS = {"PASSED"}
TERMINAL_FAIL = {"FAILED", "WRONG_ANSWER", "ERROR", "COMPILATION_ERROR",
                 "COMPILE_ERROR", "RATE_LIMIT_EXCEEDED"}
TERMINAL      = TERMINAL_PASS | TERMINAL_FAIL
SKIP_PRINT    = {"PTX", "SASS"}  # huge payloads, skip

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

def ok(msg):   print(f"{GREEN}✓{RESET} {msg}")
def fail(msg): print(f"{RED}✗{RESET} {msg}", file=sys.stderr)
def info(msg): print(f"  → {msg}")


def discover_problems() -> list[str]:
    if not os.path.isdir(SOLUTIONS_DIR):
        fail(f"Solutions directory not found: {SOLUTIONS_DIR}")
        sys.exit(1)
    slugs = [
        f.removesuffix(".cu")
        for f in sorted(os.listdir(SOLUTIONS_DIR))
        if f.endswith(".cu")
    ]
    if not slugs:
        fail("No .cu files found in ci/solutions/ — add known-good solutions first")
        sys.exit(1)
    return slugs


def select_problems(all_problems: list[str]) -> list[str]:
    if SPECIFIC:
        slugs = [s.strip() for s in SPECIFIC.split(",") if s.strip()]
        missing = [s for s in slugs if s not in all_problems]
        if missing:
            fail(f"Requested problems not in ci/solutions/: {missing}")
            sys.exit(1)
        return slugs
    if SAMPLE > 0:
        return random.sample(all_problems, min(SAMPLE, len(all_problems)))
    return all_problems


def submit(slug: str, code: str) -> dict:
    """POST to /api/submissions/sample, consume SSE until terminal event."""
    resp = requests.post(
        ENDPOINT,
        json={
            "code":        code,
            "gpuType":     GPU_TYPE,
            "language":    LANGUAGE,
            "problemSlug": slug,
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type":  "application/json",
            "Accept":        "text/event-stream",
        },
        stream=True,
        timeout=TIMEOUT,
    )

    if resp.status_code == 401:
        fail("Authentication failed — check TENSARA_CI_API_KEY secret")
        sys.exit(1)
    if resp.status_code == 404:
        raise RuntimeError(f"Problem '{slug}' not found in DB (404)")
    if not resp.ok:
        raise RuntimeError(
            f"Endpoint returned {resp.status_code}: {resp.text[:300]}"
        )

    current_event = None

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            current_event = None
            continue

        if raw.startswith("event: "):
            current_event = raw[7:].strip()
            continue

        if raw.startswith("data: "):
            try:
                data = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue

            status = data.get("status") or current_event or ""

            if status in SKIP_PRINT:
                pass  # PTX/SASS are huge — skip
            elif status == "COMPILING":
                info("compiling...")
            elif status == "CHECKING":
                info("checking...")
            elif status not in TERMINAL:
                info(status)

            if status in TERMINAL:
                return data

    raise RuntimeError("SSE stream ended without a terminal status event")


def run_checks(problems: list[str]) -> tuple[list[str], list[dict]]:
    passed_list, failed_list = [], []

    for i, slug in enumerate(problems, 1):
        code  = open(os.path.join(SOLUTIONS_DIR, f"{slug}.cu")).read()
        start = time.time()
        print(f"\n[{i}/{len(problems)}] {slug}")

        try:
            result  = submit(slug, code)
            elapsed = time.time() - start
            status  = result.get("status", "UNKNOWN")

            if status in TERMINAL_PASS:
                ok(f"PASSED  ({elapsed:.1f}s)")
                passed_list.append(slug)
            else:
                fail(f"{status}  ({elapsed:.1f}s)")
                for key in ("error", "details", "stderr", "stdout"):
                    val = result.get(key)
                    if val:
                        fail(f"  {key}: {str(val)[:400]}")
                failed_list.append({"problem": slug, "status": status})

        except Exception as e:
            elapsed = time.time() - start
            fail(f"Exception after {elapsed:.1f}s: {e}")
            failed_list.append({"problem": slug, "status": "EXCEPTION", "error": str(e)})

    return passed_list, failed_list


def main():
    if not API_KEY:
        fail("TENSARA_CI_API_KEY is not set")
        sys.exit(1)

    all_problems = discover_problems()
    problems     = select_problems(all_problems)

    print("Tensara Sanity Check")
    print(f"  Endpoint : {ENDPOINT}")
    print(f"  GPU      : {GPU_TYPE}")
    print(f"  Language : {LANGUAGE}")
    print(f"  Problems : {len(problems)} / {len(all_problems)} total", end="")
    if SAMPLE > 0 and not SPECIFIC:
        print(f"  (random sample of {SAMPLE})", end="")
    print()

    passed_list, failed_list = run_checks(problems)

    print("\n" + "─" * 50)
    print(f"Results: {len(passed_list)} passed, {len(failed_list)} failed")

    if failed_list:
        print("\nFailed:")
        for f in failed_list:
            fail(f"  {f['problem']}: {f.get('status')} {f.get('error', '')}")
        sys.exit(1)

    ok(f"All {len(passed_list)} problems passed!")


if __name__ == "__main__":
    main()