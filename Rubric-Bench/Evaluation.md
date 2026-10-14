# Evaluation 01: Nucleus (top-p) sampling filter

**Task:** `top_p_filter(probs, p)`, a spec-exact implementation prompt (see `Prompt.md`).
**Responses compared:** A, B, C (synthetic; see the limitations section).
**Date:** 2026-10-13

## 1. Verdict

| Rank | Response | Weighted score (1-5) | Checks passed | One-line reason |
|---|---|---|---|---|
| **1** | **B** | **4.85** | 10 / 10 | Correct, robust, and every claim in its explanation checks out. |
| 2 | C | 3.45 | 5 / 10 | Picks the correct nucleus but never renormalizes, violating an explicit requirement. |
| 3 | A | 1.55 | 5 / 10 | Polished and confident, but excludes the token that crosses the threshold, and its explanation misstates the definition. |

**Confidence: high for the ranking.** B is at least as good as C, and C at least as good as A, on every rubric criterion (section 6), so the order does not depend on the weights chosen. **Medium for the size of the B-versus-C gap**, which depends on how strictly the "renormalize" requirement is weighted.

**Key lesson:** A and C both fail 5 of 10 checks, yet they are very different. C has one trivial root cause. A has an algorithmic error that makes its output wrong on every ordinary input. Pass counts do not measure severity (section 4).

## 2. Method

1. **Ground truth first.** The reference definition comes from Holtzman et al. (2020), "The Curious Case of Neural Text Degeneration": the nucleus is the *smallest* set of top-probability tokens whose cumulative probability is *at least* `p`.
2. **Run code before reading explanations**, to avoid being anchored by confident prose. `python verify.py -v` extracts the first Python block from each response and runs ten checks.
3. **Quantify severity** with `python measure_failure_rates.py`, which compares each response with an independent reference implementation on thousands of random distributions.
4. **Check every factual claim** each response makes in its own explanation (section 5).
5. **Score** against a weighted rubric, then rank pairwise (sections 6-7).
6. **Write feedback** addressed to the model (section 8).

Labels were deliberately shuffled so that the best response is not first, to avoid position bias.

## 3. Verification results (`verify.py`)

| Check | A | B | C |
|---|---|---|---|
| `includes_crossing_token` (the token that reaches `p` is kept) | FAIL | PASS | FAIL |
| `exact_boundary` (cumulative lands exactly on `p`) | PASS | PASS | FAIL |
| `dominant_token` (first token already exceeds `p`) | FAIL | PASS | FAIL |
| `p_one_keeps_all_mass` (randomized, `p = 1.0`) | FAIL | PASS | PASS |
| `zero_prob_never_kept` | PASS | PASS | PASS |
| `output_is_a_distribution` (sums to 1, non-negative) | FAIL | PASS | FAIL |
| `nucleus_is_minimal` (correct, minimal, top-probability set) | FAIL | PASS | PASS |
| `does_not_mutate_input` | PASS | PASS | PASS |
| `rejects_invalid_p` | PASS | PASS | PASS |
| `ties_keep_minimal_count` | PASS | PASS | FAIL |
| **Total** | **5 / 10** | **10 / 10** | **5 / 10** |

### Root-cause analysis

| Response | Failing checks | Distinct root causes |
|---|---|---|
| A | 5 | **(1)** Stops *before* the token that would cross `p` (`cumulative + prob > p`), so the nucleus is too small. **(2)** When the first token alone exceeds `p`, the kept set is empty and the function silently returns all zeros. The loop over the empty set never runs, so the division by `total` is never executed and nothing raises. **(3)** At `p = 1.0`, floating-point rounding makes `cumulative + prob` slightly exceed 1.0 on the last token, which gets dropped. |
| C | 5 | **One:** the kept probabilities are not divided by their sum. Every failure follows from that, because the checks that compare exact values, the sum-to-1 check, and the ties check all assume renormalization. C's `nucleus_is_minimal` check passes, which confirms that it selects the right tokens. |

Why A passes `exact_boundary`: `>` and `>=` agree when the cumulative sum lands exactly on `p`. The bug only appears when the threshold is crossed, which is nearly always.

## 4. How often is each response wrong? (`measure_failure_rates.py`)

5,000 random distributions per cell (20,000 for the last table), vocabulary size 2-60, weights `U(0,1)^3` normalized. The reference is an independent implementation inside the script.

**Wrong number of kept tokens**

| p | A | B | C |
|---|---|---|---|
| 0.5 | **100%** | 0% | 0% |
| 0.9 | **100%** | 0% | 0% |
| 0.95 | **100%** | 0% | 0% |

**Output is not a valid distribution (sum differs from 1 by more than 1e-9)**

| p | A | B | C |
|---|---|---|---|
| 0.1 | 61.2% | 0% | **100%** |
| 0.5 | 7.4% | 0% | **100%** |
| 0.9 | 1.4% | 0% | **98.9%** |

**Entirely-zero output at `p = 0.1`:** A 61.4%, B 0%, C 0%.

**`p = 1.0` drops a token carrying mass above 1e-12:** A 23.71%, B 0%, C 0%.

**Reading the numbers:**
- A's crossing-token error is not an edge case. It is wrong on essentially every input where the cumulative sum does not land exactly on `p`.
- A's failure is also silent. For low `p`, which is common in practice, it returns a zero vector without any error. Sampling from that vector would crash or, depending on the sampler, behave unpredictably.
- C is wrong in a uniform, predictable, easily fixable way: the output has the right support and the wrong scale.
- B and C do not "drop" tokens at `p = 1.0` in any meaningful sense. The only tokens they skip carry mass below float precision.

## 5. Claim-by-claim check of each explanation

| Response | Claim in the explanation | Verified? |
|---|---|---|
| A | Nucleus sampling "as defined in the original paper" keeps tokens while cumulative probability does *not exceed* `p`. | **False.** The paper defines the smallest set whose cumulative probability is *at least* `p`. |
| A | `p = 1.0` "returns the original distribution unchanged." | **False** in 23.7% of random cases. Example found: vocabulary size 37, `sum(probs) == 1.0`, yet a token carrying mass 5.1e-05 was dropped. |
| A | "Renormalization: the output sums to 1." | **False** whenever the kept set is empty (61% of cases at `p = 0.1`). |
| A | Ties keep original order; input not mutated; O(V log V); `ValueError` on invalid `p`. | True. |
| B | The crossing token is included because the nucleus is the *smallest* set with mass *at least* `p`. | True, and matches the paper. |
| B | A single token with probability `>= p` gives a one-hot result. | True (`[0.9, 0.05, 0.05]`, `p = 0.5` gives `[1.0, 0.0, 0.0]`). |
| B | Float shortfall at `p = 1.0` does not drop tokens; zero-probability tokens are never kept. | True. |
| B | Ties break by lower index; all-zero input raises `ValueError`. | True (`[0.25]*4`, `p = 0.5` gives `[0.5, 0.5, 0, 0]`). |
| C | A new list is returned and the input is unchanged; `ValueError` on invalid `p`. | True. |
| C | `random.choices` accepts unnormalized weights. | True, but beside the point: the prompt required a renormalized output. |
| C | Omitting renormalization is acceptable because it is "redundant work." | **Rationalizes ignoring an explicit requirement.** Other consumers (entropy, log-probabilities, reporting) require a true distribution. |

**Response A made three checkable statements that were false (the definition, the `p = 1.0` behavior, and the normalization guarantee) and presented them with confidence.** Response C was transparent that it skipped renormalization, which is more honest even though it did not follow the instruction.

## 6. Rubric scoring

| Criterion | Weight | What it measures |
|---|---|---|
| Correctness | 40% | Output matches the specification on valid inputs |
| Robustness | 20% | Edge cases, failure behavior, no silent errors |
| Explanation accuracy | 15% | Every factual claim in the prose is true |
| Code quality | 15% | Readability, structure, idiom, maintainability |
| Communication | 10% | Clarity, concision, calibrated confidence |

| Criterion | Weight | A | B | C |
|---|---|---|---|---|
| Correctness | 0.40 | 1 | 5 | 3 |
| Robustness | 0.20 | 1 | 5 | 4 |
| Explanation accuracy | 0.15 | 1 | 5 | 3 |
| Code quality | 0.15 | 4 | 4 | 4 |
| Communication | 0.10 | 2 | 5 | 4 |
| **Weighted score** | | **1.55** | **4.85** | **3.45** |

**Justifications**
- **A, Correctness 1:** wrong nucleus on every ordinary input (section 4). **Robustness 1:** silent all-zero output and token loss at `p = 1.0`. **Explanation 1:** misstates the definition and makes false claims about `p = 1.0` and normalization. **Code quality 4:** clean structure and a good docstring. **Communication 2:** well organized, but the confident tone is unjustified because the content is wrong.
- **B, Correctness 5:** passes all ten checks and every random test. **Robustness 5:** handles dominant tokens, float shortfall, zero probabilities, and all-zero input. **Code quality 4:** minor point only; a partial selection approach could avoid a full sort for very large vocabularies. **Communication 5:** short, accurate, and every claim checks out.
- **C, Correctness 3:** correct nucleus, but violates the explicit "renormalized so the list sums to 1" requirement. **Robustness 4:** handles the other edge cases but has no guard for all-zero input. **Explanation 3:** accurate facts, but argues against an explicit requirement. **Communication 4:** concise and honest about the deviation.

**Weight sensitivity:** B scores at least as high as C, and C at least as high as A, on every criterion. The ranking B > C > A therefore holds for any choice of weights.

## 7. Pairwise preferences

Scale: 1 = slightly better, 2 = better, 3 = much better.

| Comparison | Preferred | Strength | Reason |
|---|---|---|---|
| B vs A | **B** | 3 (much better) | A is wrong on essentially every input and fails silently. B is fully correct. |
| B vs C | **B** | 2 (better) | Both pick the right nucleus. Only B returns a valid distribution as the prompt demanded. |
| C vs A | **C** | 3 (much better) | C is one division away from correct. A needs a rewritten loop, handling of the empty set, and explanation corrections. |

**Final ranking: B > C > A**

## 8. Written feedback to the model

**To Response A**
1. Your stopping condition excludes the token that crosses the threshold. Nucleus sampling keeps the smallest set whose cumulative probability is *at least* `p`, so add the token first, then test `cumulative >= p` and stop.
2. When the first token already exceeds `p`, your kept list is empty and the function returns all zeros without raising. Either include the first token (which the corrected rule does automatically) or raise a clear error.
3. Do not claim that `p = 1.0` returns the original distribution without testing it. Floating-point sums can exceed 1.0 on the final token, which your check then rejects. Prefer `cumulative >= p` on the *kept* set so rounding cannot remove tokens.
4. Verify claims about published definitions against the source before stating them as fact. Here the statement about the original paper is wrong.

**To Response C**
1. The prompt required the output to be renormalized so it sums to 1. Dividing each kept probability by the kept total is one line, so follow the instruction rather than substituting your own preference.
2. Your reasoning that samplers normalize internally is true for `random.choices` but not for every consumer. Entropy, log-probability reporting, and other distribution-based code need a real distribution.
3. Consider rejecting an all-zero input explicitly instead of returning zeros.

**To Response B**
1. Strong answer. For very large vocabularies, `heapq.nlargest` or a partial selection could avoid sorting the whole list. Mention the trade-off if you suggest it.

## 9. Evaluator pitfalls this example illustrates

- **Polish and length are not correctness.** A is the longest and most structured response and is the worst. A reviewer who skims the explanation or counts bullet points could easily rank it first. Running the code before reading the prose prevents this.
- **Equal pass counts hide unequal severity.** A and C both pass 5 of 10 checks. A is wrong on 100% of ordinary inputs; C is wrong in one fixable way.
- **Hand-picked tests can pass by luck.** See the revision log below.
- **Confident explanations deserve fact-checking.** A's false statement about the original paper is exactly the kind of error that sounds authoritative.

### Test revision log

My first `p = 1.0` check used the single case `[0.1] * 10`. Response A passed it, because that particular sum is slightly *below* 1.0 in floating point. A randomized probe then showed A drops a token in about 23.7% of random distributions. I replaced the check with a randomized version and a tolerance that does not penalize B and C for skipping a token whose mass is below float precision (1e-12). The final check separates the responses correctly. Single hand-picked examples are not enough for numerical edge cases.

## 10. Limitations

- **The three responses are synthetic.** They were written for this evaluation to represent realistic failure patterns, not collected from named models. Replace them with real model outputs (recording model name, version, date, and settings) before presenting this as a comparison of real systems.
- **One prompt, one task.** Conclusions apply to this prompt only.
- **The tests were written by the evaluator.** They encode my reading of the specification. Ties and all-zero input are not specified in the prompt, so the tie-break order and the all-zero behavior are not scored beyond the count check and the guard noted in the rubric.
- **Failure rates depend on the distribution family** (`U(0,1)^3` weights). Real LLM distributions are typically more peaked, which would make A's all-zero failure more frequent, but this was not measured.
- **Scores are one evaluator's judgment.** A second independent rater would be needed to measure agreement.

## 11. Reproduce

```bash
python verify.py -v                  # ten checks per response, with failure reasons
python measure_failure_rates.py      # failure rates on random distributions
```
