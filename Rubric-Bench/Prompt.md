# Prompt 01: Nucleus (top-p) sampling filter

| Field | Value |
|---|---|
| Domain | LLM inference / probability / Python |
| Task type | Code generation with a precise specification |
| Difficulty | Medium (short code, easy to get subtly wrong) |
| Evaluation date | 2026-10-13 |
| Responses compared | 3 (labels A, B, C; order randomized, see `Evaluation.md`) |

## Prompt (verbatim, as given to every response)

> Write a Python function `top_p_filter(probs, p)` for nucleus (top-p) sampling.
>
> - `probs` is a list of floats: a probability distribution over a vocabulary (non-negative, sums to 1).
> - Find the **smallest** set of highest-probability tokens whose cumulative probability is **at least** `p`.
> - Return a **new** list of the same length in which every token outside that set has probability `0.0`, and the tokens inside the set are **renormalized** so the list sums to 1.
> - Raise `ValueError` if `p` is not in the interval `(0, 1]`.
> - Do not modify the input list.
>
> Use only the standard library. Briefly explain your approach and list the edge cases you handled.

## Why this prompt

The specification is exact enough that correctness can be checked mechanically, yet the task has well-known pitfalls:

- Off-by-one on the token that **crosses** the threshold.
- Forgetting to **renormalize**.
- Floating-point error when `p = 1.0`.
- Silent failure when one token already exceeds `p`.
- Explanations that confidently misstate the definition.

## What a correct answer must satisfy

1. The kept set is the smallest top-probability prefix with cumulative mass `>= p` (the crossing token is included).
2. Output has the same length, is non-negative, and sums to 1.
3. Tokens outside the nucleus are exactly `0.0`; zero-probability tokens are never kept.
4. `ValueError` for `p <= 0` or `p > 1`.
5. Input is not mutated.
6. Any explanation must be factually consistent with the code and with the original definition (Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration").
