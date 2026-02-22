#!/usr/bin/env python3
"""
Tensara CI Sanity Check
-----------------------
Submits known-good solutions to /api/submissions/sample and verifies they pass.
Catches regressions in the checker/engine across languages.

Usage:
    python testing/sanity_check.py                              # run all (cuda)
    CI_SAMPLE_SIZE=5 python testing/sanity_check.py             # random sample (cuda)
    CI_LANGUAGES=cuda,mojo python testing/sanity_check.py       # run across languages
    CI_PROBLEMS=vector-addition python testing/sanity_check.py  # specific slugs
    CI_SEED=123 CI_SAMPLE_SIZE=5 python testing/sanity_check.py # reproducible sampling
    TENSARA_URL=http://localhost:3000 python testing/sanity_check.py  # local dev
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
API_KEY = os.environ.get("TENSARA_CI_API_KEY", "")
BASE_URL = os.environ.get("TENSARA_URL", "https://tensara.org")
GPU_TYPE = os.environ.get("CI_GPU_TYPE", "T4")
LANGUAGE = os.environ.get(
    "CI_LANGUAGE", ""
).strip()  # backwards compat: single language
LANGUAGES = os.environ.get("CI_LANGUAGES", "").strip()  # comma-separated languages
SAMPLE = int(os.environ.get("CI_SAMPLE_SIZE", "0"))  # 0 = all
SPECIFIC = os.environ.get("CI_PROBLEMS", "")  # comma-separated slugs
TIMEOUT = int(os.environ.get("CI_TIMEOUT_SECONDS", "300"))
SEED_RAW = os.environ.get("CI_SEED", "").strip()

SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solutions")
ENDPOINT = f"{BASE_URL}/api/submissions/sample"

# Terminal SSE statuses
TERMINAL_PASS = {"PASSED"}
TERMINAL_FAIL = {
    "FAILED",
    "WRONG_ANSWER",
    "ERROR",
    "COMPILATION_ERROR",
    "COMPILE_ERROR",
    "RATE_LIMIT_EXCEEDED",
}
TERMINAL = TERMINAL_PASS | TERMINAL_FAIL
SKIP_PRINT = {"PTX", "SASS"}  # huge payloads, skip

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg):
    print(f"{GREEN}✓{RESET} {msg}")


def fail(msg):
    print(f"{RED}✗{RESET} {msg}", file=sys.stderr)


def info(msg):
    print(f"  → {msg}")


EXT_TO_LANGUAGE = {
    ".cu": "cuda",
    ".py": "python",
    ".mojo": "mojo",
    ".cute": "cute",
    ".cutile": "cutile",
}


def parse_languages() -> list[str]:
    if LANGUAGES:
        langs = [p.strip() for p in LANGUAGES.split(",") if p.strip()]
        return langs or ["cuda"]
    if LANGUAGE:
        return [LANGUAGE]
    return ["cuda", "mojo"]


def parse_specific_slugs() -> list[str]:
    if not SPECIFIC:
        return []
    return [s.strip() for s in SPECIFIC.split(",") if s.strip()]


def parse_seed() -> int | None:
    if not SEED_RAW:
        return None
    try:
        return int(SEED_RAW)
    except ValueError:
        return int.from_bytes(SEED_RAW.encode("utf-8"), "little") % (2**31 - 1)


def discover_solutions() -> dict[str, dict[str, str]]:
    if not os.path.isdir(SOLUTIONS_DIR):
        fail(f"Solutions directory not found: {SOLUTIONS_DIR}")
        sys.exit(1)

    by_language: dict[str, dict[str, str]] = {}

    for root, _, files in os.walk(SOLUTIONS_DIR):
        for filename in files:
            path = os.path.join(root, filename)
            _, ext = os.path.splitext(filename)
            if not ext:
                continue

            language = EXT_TO_LANGUAGE.get(ext.lower())
            if language is None:
                continue

            slug = filename.removesuffix(ext)
            by_language.setdefault(language, {})[slug] = path

    if not by_language:
        fail(f"No recognized solution files found under {SOLUTIONS_DIR}/")
        fail(f"Supported extensions: {', '.join(sorted(EXT_TO_LANGUAGE.keys()))}")
        sys.exit(1)

    return by_language


def select_problems(
    all_problems: list[str],
    *,
    rng: random.Random,
    specific_slugs: list[str],
    language: str,
) -> list[str]:
    if specific_slugs:
        present = [s for s in specific_slugs if s in all_problems]
        missing = [s for s in specific_slugs if s not in all_problems]
        if missing:
            info(f"Skipping {len(missing)} missing slug(s) for {language}: {missing}")
        return present
    if SAMPLE > 0:
        return rng.sample(all_problems, min(SAMPLE, len(all_problems)))
    return all_problems


def fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def print_test_result(
    tc_name: str, passed: bool, debug_info: dict, runtime_ms: float | None
):
    indent = "    "
    rt = f"  {runtime_ms:.2f}ms" if runtime_ms is not None else ""
    if passed:
        print(f"{indent}{GREEN}✓{RESET} {tc_name}{rt}")
        return

    print(f"{indent}{RED}✗{RESET} {tc_name}{rt}")
    if not debug_info:
        return

    max_diff = debug_info.get("max_difference")
    mean_diff = debug_info.get("mean_difference")
    msg = debug_info.get("message") or debug_info.get("error")
    if max_diff is not None:
        print(f"{indent}  max_diff={fmt_val(max_diff)}", end="")
        if mean_diff is not None:
            print(f"  mean_diff={fmt_val(mean_diff)}", end="")
        print()
    if msg:
        print(f"{indent}  {msg}")
    sample = (
        debug_info.get("sample_differences")
        or debug_info.get("sample_mismatches")
        or {}
    )
    if sample:
        print(f"{indent}  sample diffs:")
        for k, v in list(sample.items())[:3]:
            exp = fmt_val(v.get("expected", "?"))
            act = fmt_val(v.get("actual", "?"))
            diff = fmt_val(v.get("diff", "")) if "diff" in v else ""
            diff_str = f"  Δ={diff}" if diff else ""
            print(f"{indent}    [{k}]  expected={exp}  actual={act}{diff_str}")


def submit(*, slug: str, code: str, language: str) -> dict:
    """POST to /api/submissions/sample, consume SSE until terminal event."""
    resp = requests.post(
        ENDPOINT,
        json={
            "code": code,
            "gpuType": GPU_TYPE,
            "language": language,
            "problemSlug": slug,
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        stream=True,
        timeout=TIMEOUT,
    )

    try:
        if resp.status_code == 401:
            fail("Authentication failed — check TENSARA_CI_API_KEY secret")
            sys.exit(1)
        if resp.status_code == 404:
            raise RuntimeError(f"Problem '{slug}' not found in DB (404)")
        if not resp.ok:
            raise RuntimeError(f"Endpoint returned {resp.status_code}: {resp.text[:300]}")

        current_event = None
        printed_compiling = False
        any_test_failed = False

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
                    pass
                elif status == "COMPILING":
                    if not printed_compiling:
                        info("compiling...")
                        printed_compiling = True
                elif status == "CHECKING":
                    pass
                elif status == "TEST_RESULT":
                    tc_name = (
                        data.get("testCaseName")
                        or data.get("test_case_name")
                        or data.get("name", "?")
                    )
                    tc_passed = data.get("passed", False)
                    tc_debug = data.get("debugInfo") or data.get("debug_info") or {}
                    tc_rt = data.get("runtimeMs") or data.get("runtime_ms")
                    print_test_result(tc_name, tc_passed, tc_debug, tc_rt)
                    if not tc_passed:
                        any_test_failed = True
                elif status not in TERMINAL:
                    info(status)

                if status in TERMINAL:
                    data["_any_test_failed"] = any_test_failed
                    return data
    finally:
        resp.close()

    raise RuntimeError("SSE stream ended without a terminal status event")


def run_checks(
    *, language: str, solutions: dict[str, str], problems: list[str]
) -> tuple[list[str], list[dict]]:
    passed_list: list[str] = []
    failed_list: list[dict] = []

    for i, slug in enumerate(problems, 1):
        code_path = solutions.get(slug)
        if not code_path:
            fail(f"Missing solution for language={language} slug={slug}")
            failed_list.append(
                {"language": language, "problem": slug, "status": "MISSING_SOLUTION"}
            )
            continue

        code = open(code_path).read()
        start = time.time()
        print(f"\n[{i}/{len(problems)}] {slug}  ({language})")

        try:
            result = submit(slug=slug, code=code, language=language)
            elapsed = time.time() - start
            status = result.get("status", "UNKNOWN")

            if status in TERMINAL_PASS:
                ok(f"PASSED  ({elapsed:.1f}s)")
                passed_list.append(slug)
            else:
                fail(f"{status}  ({elapsed:.1f}s)")
                for key in ("error", "details", "message"):
                    val = result.get(key)
                    if val:
                        fail(f"  {key}: {str(val)[:400]}")
                raw_keys = {
                    k: v
                    for k, v in result.items()
                    if k not in ("code",) and not isinstance(v, (bytes,))
                }
                info(f"raw result: {json.dumps(raw_keys, default=str)[:600]}")
                failed_list.append(
                    {"language": language, "problem": slug, "status": status}
                )

        except Exception as e:
            elapsed = time.time() - start
            fail(f"Exception after {elapsed:.1f}s: {e}")
            failed_list.append(
                {
                    "language": language,
                    "problem": slug,
                    "status": "EXCEPTION",
                    "error": str(e),
                }
            )

    return passed_list, failed_list


def main():
    if not API_KEY:
        fail("TENSARA_CI_API_KEY is not set")
        sys.exit(1)

    languages = parse_languages()
    seed = parse_seed()
    rng = random.Random(seed)
    specific_slugs = parse_specific_slugs()

    solutions_by_language = discover_solutions()

    missing_langs = [lang for lang in languages if lang not in solutions_by_language]
    if missing_langs:
        available = ", ".join(sorted(solutions_by_language.keys()))
        fail(f"No solutions found for requested language(s): {missing_langs}")
        fail(f"Available languages in solutions/: {available}")
        sys.exit(1)

    print("Tensara Sanity Check")
    print(f"  Endpoint : {ENDPOINT}")
    print(f"  GPU      : {GPU_TYPE}")
    print(f"  Seed     : {seed if seed is not None else '(unset)'}")
    print(f"  Languages: {', '.join(languages)}")

    all_failures: list[dict] = []
    total_passed = 0
    total_failed = 0
    skipped_languages: list[str] = []

    for lang in languages:
        all_slugs = sorted(solutions_by_language[lang].keys())
        problems = select_problems(
            all_slugs, rng=rng, specific_slugs=specific_slugs, language=lang
        )
        if not problems:
            info(f"Skipping language {lang}: no matching solution(s) selected")
            skipped_languages.append(lang)
            continue

        print(f"\nLanguage: {lang}")
        print(f"  Problems : {len(problems)} / {len(all_slugs)} total", end="")
        if SAMPLE > 0 and not specific_slugs:
            print(f"  (random sample of {SAMPLE})", end="")
        print()

        passed_list, failed_list = run_checks(
            language=lang, solutions=solutions_by_language[lang], problems=problems
        )
        total_passed += len(passed_list)
        total_failed += len(failed_list)
        all_failures.extend(failed_list)

    print("\n" + "─" * 50)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    if skipped_languages:
        info(f"Skipped languages: {', '.join(skipped_languages)}")

    if all_failures:
        print("\nFailed:")
        for f in all_failures:
            lang = f.get("language", "?")
            fail(f"  {lang} / {f['problem']}: {f.get('status')} {f.get('error', '')}")
        sys.exit(1)

    ok(f"All {total_passed} checks passed!")


if __name__ == "__main__":
    main()
