"""
Verify every Response_*.md in this folder against the Prompt.md specification.

Usage:  python verify.py            (summary table)
        python verify.py -v         (also print why each failing test failed)

The script extracts the FIRST ```python block from each response, executes it,
and runs the same ten checks against `top_p_filter`. Standard library only.
"""

import copy
import glob
import os
import random
import re
import sys

TOL = 1e-9


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_function(path):
    text = open(path, encoding="utf-8").read()
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if not match:
        raise RuntimeError("no python code block found")
    namespace = {}
    exec(match.group(1), namespace)
    return namespace["top_p_filter"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def close(a, b):
    return len(a) == len(b) and all(abs(x - y) <= TOL for x, y in zip(a, b))


def expect_close(actual, expected):
    if not close(actual, expected):
        raise AssertionError(f"expected {expected}, got {actual}")


def random_distribution(rng, size):
    weights = [rng.random() ** 3 for _ in range(size)]  # skewed, realistic
    total = sum(weights)
    return [w / total for w in weights]


# --------------------------------------------------------------------------
# Checks (each takes the function under test and raises on failure)
# --------------------------------------------------------------------------
def t_includes_crossing_token(f):
    # 0.5 + 0.3 = 0.8 < 0.9, so the 0.15 token crosses the threshold and is kept.
    out = f([0.5, 0.3, 0.15, 0.05], 0.9)
    expect_close(out, [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0.0])


def t_exact_boundary(f):
    # Binary fractions: 0.5 + 0.25 == 0.75 exactly, so exactly two tokens are kept.
    out = f([0.5, 0.25, 0.125, 0.125], 0.75)
    expect_close(out, [2 / 3, 1 / 3, 0.0, 0.0])


def t_dominant_token(f):
    # One token already exceeds p: the nucleus is that single token.
    expect_close(f([0.9, 0.05, 0.05], 0.5), [1.0, 0.0, 0.0])


def t_p_one_keeps_all_mass(f):
    # Fixed case: ten 0.1s sum to 0.9999999999999999 in floating point.
    expect_close(f([0.1] * 10, 1.0), [0.1] * 10)
    # Randomized: at p = 1.0 the nucleus must contain (essentially) the whole
    # distribution. A token may only be skipped if its mass is below float
    # precision (1e-12), because the smallest set reaching 1.0 in floating point
    # can legitimately omit such a token.
    rng = random.Random(11)
    for _ in range(2000):
        probs = random_distribution(rng, rng.randint(2, 60))
        kept = {i for i, v in enumerate(f(probs, 1.0)) if v > 0.0}
        dropped_mass = sum(v for i, v in enumerate(probs) if i not in kept)
        if dropped_mass > 1e-12:
            raise AssertionError(
                f"p=1.0 dropped {len(probs) - len(kept)} token(s) carrying mass {dropped_mass:.3e}"
            )


def t_zero_prob_never_kept(f):
    expect_close(f([0.6, 0.4, 0.0, 0.0], 1.0), [0.6, 0.4, 0.0, 0.0])


def t_output_is_a_distribution(f):
    rng = random.Random(2026)
    for _ in range(200):
        probs = random_distribution(rng, rng.randint(2, 40))
        for p in (0.1, 0.5, 0.9, 0.95, 1.0):
            out = f(probs, p)
            if len(out) != len(probs) or min(out) < 0.0:
                raise AssertionError("wrong length or negative probability")
            if abs(sum(out) - 1.0) > 1e-9:
                raise AssertionError(f"sums to {sum(out):.6f}, expected 1 (p={p})")


def t_nucleus_is_minimal(f):
    rng = random.Random(7)
    for _ in range(200):
        probs = random_distribution(rng, rng.randint(2, 40))
        for p in (0.1, 0.5, 0.9, 0.95):
            kept = [i for i, v in enumerate(f(probs, p)) if v > 0.0]
            mass = sum(probs[i] for i in kept)
            if mass < p - 1e-12:
                raise AssertionError(f"kept mass {mass:.6f} < p={p}")
            smallest = min(probs[i] for i in kept)
            if len(kept) > 1 and mass - smallest >= p + 1e-12:
                raise AssertionError("nucleus is not minimal (a token could be removed)")
            others = [probs[i] for i in range(len(probs)) if i not in kept]
            if others and max(others) > smallest + 1e-12:
                raise AssertionError("kept set is not the top-probability prefix")


def t_does_not_mutate_input(f):
    probs = [0.4, 0.3, 0.2, 0.1]
    snapshot = copy.deepcopy(probs)
    f(probs, 0.8)
    if probs != snapshot:
        raise AssertionError(f"input changed to {probs}")


def t_rejects_invalid_p(f):
    for bad in (0, 0.0, -0.1, 1.0000001, 1.5):
        try:
            f([0.5, 0.5], bad)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for p={bad}")


def t_ties_keep_minimal_count(f):
    out = f([0.25, 0.25, 0.25, 0.25], 0.5)
    if sum(1 for v in out if v > 0.0) != 2 or abs(sum(out) - 1.0) > TOL:
        raise AssertionError(f"expected two tokens summing to 1, got {out}")


CHECKS = [
    ("includes_crossing_token", t_includes_crossing_token),
    ("exact_boundary", t_exact_boundary),
    ("dominant_token", t_dominant_token),
    ("p_one_keeps_all_mass", t_p_one_keeps_all_mass),
    ("zero_prob_never_kept", t_zero_prob_never_kept),
    ("output_is_a_distribution", t_output_is_a_distribution),
    ("nucleus_is_minimal", t_nucleus_is_minimal),
    ("does_not_mutate_input", t_does_not_mutate_input),
    ("rejects_invalid_p", t_rejects_invalid_p),
    ("ties_keep_minimal_count", t_ties_keep_minimal_count),
]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def main():
    verbose = "-v" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(here, "Response_*.md")))

    results = {}  # response name -> {check name -> error message or None}
    for path in paths:
        name = os.path.basename(path)[:-3]
        results[name] = {}
        try:
            func = load_function(path)
        except Exception as exc:  # code does not even load
            for check_name, _ in CHECKS:
                results[name][check_name] = f"could not load code: {exc}"
            continue
        for check_name, check in CHECKS:
            try:
                check(func)
                results[name][check_name] = None
            except Exception as exc:  # noqa: BLE001 - report any failure mode
                results[name][check_name] = f"{type(exc).__name__}: {exc}"

    names = list(results)
    width = max(len(c) for c, _ in CHECKS)
    print(f"{'check':<{width}}  " + "  ".join(f"{n[-1]:^3}" for n in names))
    print("-" * (width + 2 + 5 * len(names)))
    for check_name, _ in CHECKS:
        cells = "  ".join(f"{'PASS' if results[n][check_name] is None else 'FAIL':^4}" for n in names)
        print(f"{check_name:<{width}}  {cells}")
    print("-" * (width + 2 + 5 * len(names)))
    totals = "  ".join(
        f"{sum(v is None for v in results[n].values()):>2}/{len(CHECKS)}" for n in names
    )
    print(f"{'passed':<{width}}  {totals}")

    if verbose:
        print()
        for n in names:
            failures = {c: m for c, m in results[n].items() if m is not None}
            if failures:
                print(f"{n} failures:")
                for c, m in failures.items():
                    print(f"  - {c}: {m}")


if __name__ == "__main__":
    main()
