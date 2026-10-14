# AI Response Evaluation

Structured, evidence-based comparisons of AI-generated answers to technical prompts. Each evaluation ranks the responses, gives written reasoning, and backs every verdict with executable checks wherever the task allows.

## Evaluations

| # | Topic | Responses | Ranking | Folder |
|---|---|---|---|---|
| 01 | Nucleus (top-p) sampling filter | 3 | B > C > A | [`01-nucleus-sampling-top-p`](01-nucleus-sampling-top-p) |

## Folder structure

```
NN-topic-name/
  Prompt.md                    the exact prompt, plus why it was chosen
  Response_A.md, B, C ...      each response, with a source label
  verify.py                    runs each response's code against the spec
  measure_failure_rates.py     how often each response is wrong (optional)
  Evaluation.md                scores, ranking, evidence, written feedback
```

## Method

1. **Define ground truth** from the specification or a primary source before reading any response.
2. **Verify first, read second.** Run executable checks before reading explanations, so polished prose cannot anchor the judgment.
3. **Measure severity,** not just pass/fail. Two responses with the same pass count can differ enormously in how often they are wrong.
4. **Fact-check the explanation.** Every checkable claim in a response's prose is verified.
5. **Score with a fixed rubric,** then rank pairwise.
6. **Write actionable feedback** addressed to the model.
7. **Record limitations,** including any revision made to the tests themselves.

Response labels are shuffled so the best answer is not always first.

## Rubric

| Criterion | Weight (code tasks) | Question it answers |
|---|---|---|
| Correctness | 40% | Does the output match the specification? |
| Robustness | 20% | Edge cases, failure behavior, silent errors? |
| Explanation accuracy | 15% | Is every factual claim in the prose true? |
| Code quality | 15% | Readable, idiomatic, maintainable? |
| Communication | 10% | Clear, concise, and appropriately confident? |

**Scale (1-5):** 1 = fundamentally wrong or misleading, 2 = major defects, 3 = partly correct with notable gaps, 4 = good with minor issues, 5 = correct and complete.

**Pairwise preference scale:** 1 = slightly better, 2 = better, 3 = much better.

Each `Evaluation.md` also states whether the ranking is robust to different weights.

## Source-labeling policy

Every response file has a `Source` field. It states either the model name, version, date, and settings it came from, or that it is a synthetic response written to represent a failure pattern. **The responses in evaluation 01 are synthetic.** They are not the output of any named model.

## Running the checks

Each evaluation needs only Python 3 and the standard library:

```bash
cd 01-nucleus-sampling-top-p
python verify.py -v
python measure_failure_rates.py
```
