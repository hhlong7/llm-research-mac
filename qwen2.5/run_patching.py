import json
import random
import time

import numpy as np
import torch

from utilities import load_model_fp32, answer_metrics, NUM_PATCH_LAYERS

SEED = 0
MAX_TEST = 100        # test prompts per op; lower this (e.g. 50) for a faster run
OTHER_OP = {"mul": "add", "add": "mul"}
OTHER_ANSWER = {"mul": lambda a, b: a + b, "add": lambda a, b: a * b}

model, tokenizer, device = load_model_fp32()

with open("patch_dataset.json") as f:
    rows = json.load(f)
reps = np.load("canonical_reps.npz")

means = {"mul": reps["mean_mul"], "add": reps["mean_add"]}
singles = {"mul": reps["mul_states"], "add": reps["add_states"]}
to_vec = lambda x: torch.tensor(x, dtype=torch.float32, device=device)

rng = random.Random(SEED)
results = {}
start = time.time()

for op in ["mul", "add"]:
    other = OTHER_OP[op]
    tests = [r for r in rows if r["role"] == "test" and r["op"] == op][:MAX_TEST]
    n = len(tests)
    # (c): each test prompt gets one fixed donor from the canonical set (different operands)
    donors = [rng.randrange(len(singles[op])) for _ in tests]

    clean_lp = np.zeros(n)
    acc = {c: np.zeros((n, NUM_PATCH_LAYERS), dtype=bool) for c in ["avg_same", "avg_other", "single_same"]}
    lp = {c: np.zeros((n, NUM_PATCH_LAYERS)) for c in acc}
    switched = np.zeros((n, NUM_PATCH_LAYERS), dtype=bool)   # avg_other made it answer the OTHER op

    for i, r in enumerate(tests):
        ok, clean_lp[i] = answer_metrics(model, tokenizer, device, r["prompt"], r["answer"])
        assert ok, f"clean prompt wrong, should have been filtered: {r['prompt']}"
        other_answer = OTHER_ANSWER[op](r["a"], r["b"])

        for L in range(NUM_PATCH_LAYERS):
            vecs = {
                "avg_same": to_vec(means[op][L]),
                "avg_other": to_vec(means[other][L]),
                "single_same": to_vec(singles[op][donors[i], L]),
            }
            for c, vec in vecs.items():
                acc[c][i, L], lp[c][i, L] = answer_metrics(
                    model, tokenizer, device, r["prompt"], r["answer"], patch=(L, r["op_pos"], vec))
            switched[i, L] = answer_metrics(
                model, tokenizer, device, r["prompt"], other_answer,
                patch=(L, r["op_pos"], vecs["avg_other"]))[0]

        if (i + 1) % 10 == 0:
            print(f"{op}: {i + 1}/{n} test prompts done  ({time.time() - start:.0f}s)", flush=True)

    results[op] = {"clean_lp": clean_lp, "acc": acc, "lp": lp, "switched": switched}

# ---- summary tables ----
for op in ["mul", "add"]:
    R = results[op]
    other = OTHER_OP[op]
    print(f"\n=== test op: {op}  (clean accuracy 1.00, clean mean logprob {R['clean_lp'].mean():.3f}) ===")
    print(f"layer   acc: avg {op}  avg {other}  1 other {op}   |   logprob: avg {op}  avg {other}  1 other {op}   |  "
          f"answers {other} instead")
    for L in range(NUM_PATCH_LAYERS):
        a = [R["acc"][c][:, L].mean() for c in ["avg_same", "avg_other", "single_same"]]
        l = [R["lp"][c][:, L].mean() for c in ["avg_same", "avg_other", "single_same"]]
        print(f"{L:5d}   {a[0]:12.2f} {a[1]:9.2f} {a[2]:10.2f}     |   {l[0]:16.3f} {l[1]:9.3f} {l[2]:10.3f}     |  "
              f"{R['switched'][:, L].mean():8.2f}")

np.savez("patching_results.npz",
         **{f"{op}_{k}": v for op in results for k, v in
            [("clean_lp", results[op]["clean_lp"]), ("switched", results[op]["switched"])]
            + [(f"acc_{c}", results[op]["acc"][c]) for c in results[op]["acc"]]
            + [(f"lp_{c}", results[op]["lp"][c]) for c in results[op]["lp"]]})
print(f"\nsaved patching_results.npz  (total {time.time() - start:.0f}s)")


"""
The three conditions (named so they work for both ops):
avg_same: (a), the average of the same op.
avg_other: (b), the average of the other op. This is the control.
single_same: (c), the same op’s state from one other prompt. 

Each test prompt gets one fixed random donor from the canonical set, 
so the operands are always different.
The clean check: every test prompt is first run with no swap, 
and the script asserts it’s correct. That confirms step 1’s filtering held up,
and it records the clean log prob as the baseline.

The swap: patch=(L, op_pos, vec) replaces the vector going into 
layer L at the operator token, using the hook from patch_utils.py.

The extra “answers other op instead” column: for the avg-other swap, 
it also checks whether the model now gives the other operation’s 
answer, e.g. 6 × 7 → 13. That separates two different outcomes: 
the swap confused the model, or it actually switched it to adding.

Output: a table per test op, plus patching_results.npz with every 
individual result for step 4’s plots.

"""