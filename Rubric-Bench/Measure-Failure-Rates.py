"""
Measure HOW OFTEN each response is wrong, not just whether it fails a test.

Pass/fail counts hide severity: two responses can both fail 5 of 10 checks,
one with a single trivial root cause and one with a pervasive algorithmic bug.
This script quantifies the difference on random distributions.

Usage:  python measure_failure_rates.py
Reference: an independent, straightforward nucleus implementation (below).
Distribution family: weights = U(0,1)**3, normalized (skewed, LLM-like).
"""

import random

from verify import load_function, random_distribution

RESPONSES = {"A": "Response_A.md", "B": "Response_B.md", "C": "Response_C.md"}
FUNCS = {label: load_function(path) for label, path in RESPONSES.items()}


def reference_nucleus_size(probs, p):
    """Size of the smallest top-probability prefix with cumulative mass >= p."""
    cumulative = 0.0
    for count, value in enumerate(sorted(probs, reverse=True), start=1):
        cumulative += value
        if cumulative >= p:
            return count
    return len(probs)


def kept_count(output):
    return sum(1 for v in output if v > 0.0)


def main():
    rng = random.Random(2026)
    n = 5000
    print(f"{n} random distributions per cell (vocab size 2-60)\n")

    # 1. Wrong nucleus size at typical p values (reference = independent implementation)
    print("Share of cases with the WRONG number of kept tokens")
    print(f"{'p':>5}  " + "  ".join(f"{label:>7}" for label in FUNCS))
    for p in (0.5, 0.9, 0.95):
        wrong = dict.fromkeys(FUNCS, 0)
        for _ in range(n):
            probs = random_distribution(rng, rng.randint(2, 60))
            expected = reference_nucleus_size(probs, p)
            for label, f in FUNCS.items():
                if kept_count(f(probs, p)) != expected:
                    wrong[label] += 1
        print(f"{p:>5}  " + "  ".join(f"{wrong[l] / n:>7.1%}" for l in FUNCS))

    # 2. Output is not a probability distribution
    print("\nShare of outputs whose sum differs from 1 by more than 1e-9")
    print(f"{'p':>5}  " + "  ".join(f"{label:>7}" for label in FUNCS))
    for p in (0.1, 0.5, 0.9):
        bad = dict.fromkeys(FUNCS, 0)
        for _ in range(n):
            probs = random_distribution(rng, rng.randint(2, 60))
            for label, f in FUNCS.items():
                if abs(sum(f(probs, p)) - 1.0) > 1e-9:
                    bad[label] += 1
        print(f"{p:>5}  " + "  ".join(f"{bad[l] / n:>7.1%}" for l in FUNCS))

    # 3. Silent all-zero output (no exception, no usable distribution)
    print("\nShare of outputs that are entirely zero (p = 0.1)")
    zeros = dict.fromkeys(FUNCS, 0)
    for _ in range(n):
        probs = random_distribution(rng, rng.randint(2, 60))
        for label, f in FUNCS.items():
            if sum(f(probs, 0.1)) == 0.0:
                zeros[label] += 1
    print("       " + "  ".join(f"{zeros[l] / n:>7.1%}" for l in FUNCS))

    # 4. p = 1.0 should keep (essentially) all the mass
    print("\nShare of cases where p = 1.0 drops a token carrying mass > 1e-12")
    m = 20000
    dropped = dict.fromkeys(FUNCS, 0)
    for _ in range(m):
        probs = random_distribution(rng, rng.randint(2, 60))
        for label, f in FUNCS.items():
            out = f(probs, 1.0)
            lost = sum(v for v, o in zip(probs, out) if o == 0.0)
            if lost > 1e-12:
                dropped[label] += 1
    print(f"({m} distributions)")
    print("       " + "  ".join(f"{dropped[l] / m:>7.2%}" for l in FUNCS))


if __name__ == "__main__":
    main()
