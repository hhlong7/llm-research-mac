import json
import time

import numpy as np
import torch

from utilities import load_model_fp32, NUM_PATCH_LAYERS, TEMPLATE, SYMBOLS

MAX_TEST = 100   # test prompts per op; lower this for a faster run
CHUNK = 64       # rows per forward pass; lower this if you run out of memory
OTHER = {"mul": "add", "add": "mul"}
ANSWER = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}
NAMES = ["operator", "b_last", "equals", "final"]

# which token positions get replaced together
PATCH_SETS = {
    "operator": ["operator"],
    "final": ["final"],
    "both": ["operator", "final"],
    "b_last": ["b_last"],
    "equals": ["equals"],
    "op+b": ["operator", "b_last"],
    "b+final": ["b_last", "final"],
    "op+b+final": ["operator", "b_last", "final"],
}
# where the replacement vector comes from
SOURCES = ["paired_other", "avg_other", "avg_same"]

model, tokenizer, device = load_model_fp32()

with open("patch_dataset.json") as f:
    rows = json.load(f)


def positions(row):
    n = len(tokenizer(row["prompt"])["input_ids"])
    return {"operator": row["op_pos"], "b_last": n - 3, "equals": n - 2, "final": n - 1}


def named_states(prompt, pos):
    """hidden states at each named token -> {name: (24, 896)}"""
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    return {name: torch.stack([hs[L][0, pos[name]] for L in range(NUM_PATCH_LAYERS)])
            for name in NAMES}


# ---- canonical averages at all four positions ----
canonical = {}
for op in ["mul", "add"]:
    collected = {name: [] for name in NAMES}
    for r in rows:
        if r["role"] == "canonical" and r["op"] == op:
            states = named_states(r["prompt"], positions(r))
            for name in NAMES:
                collected[name].append(states[name])
    canonical[op] = {name: torch.stack(v).mean(dim=0) for name, v in collected.items()}
print("built canonical averages", flush=True)

# one batch row per (source, patch set, layer)
plan = [(s, p, L) for s in SOURCES for p in PATCH_SETS for L in range(NUM_PATCH_LAYERS)]


def run_all_patches(prompt, answer, pos, vectors):
    """for every row in plan: exact answer correct?, P(answer), top first token"""
    prompt_ids = tokenizer(prompt)["input_ids"]
    answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    all_correct, all_prob, all_first = [], [], []

    for start in range(0, len(plan), CHUNK):
        chunk = plan[start:start + CHUNK]
        ids = torch.tensor([prompt_ids + answer_ids], device=device).repeat(len(chunk), 1)

        # one hook per layer; it patches only the rows assigned to that layer
        handles = []
        for layer in sorted({row[2] for row in chunk}):
            row_ids = [i for i, row in enumerate(chunk) if row[2] == layer]

            def hook(module, inputs, output, layer=layer, row_ids=row_ids):
                hs = output[0] if isinstance(output, tuple) else output
                for i in row_ids:
                    source, patch_set, _ = chunk[i]
                    for name in PATCH_SETS[patch_set]:
                        hs[i, pos[name], :] = vectors[source][name][layer]
                return output

            target = model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
            handles.append(target.register_forward_hook(hook))

        with torch.no_grad():
            # only the answer positions go through lm_head, which saves a lot of memory
            hidden = model.model(ids).last_hidden_state[:, n - 1:]
            logits = model.lm_head(hidden).float()
        for handle in handles:
            handle.remove()

        logprobs = torch.log_softmax(logits, dim=-1)
        total = torch.zeros(len(chunk), device=device)
        correct = torch.ones(len(chunk), dtype=torch.bool, device=device)
        for k, token in enumerate(answer_ids):
            total += logprobs[:, k, token]
            correct &= logits[:, k].argmax(dim=-1) == token

        # the token after the answer must not be another digit
        next_tokens = logits[:, -1].argmax(dim=-1).tolist()
        no_extra_digit = torch.tensor(
            [not tokenizer.decode([t]).strip().isdigit() for t in next_tokens], device=device)

        all_correct.append((correct & no_extra_digit).cpu().numpy())
        all_prob.append(total.exp().cpu().numpy())
        all_first.append(logits[:, 0].argmax(dim=-1).cpu().numpy())

    return np.concatenate(all_correct), np.concatenate(all_prob), np.concatenate(all_first)


# ---- the experiment ----
results = {}
start_time = time.time()
for op in ["mul", "add"]:
    other = OTHER[op]
    tests = [r for r in rows if r["role"] == "test" and r["op"] == op][:MAX_TEST]
    shape = (len(tests), len(plan))
    R = {k: np.zeros(shape) for k in ["acc_orig", "acc_alt", "prob_orig", "prob_alt",
                                      "first_orig", "first_alt"]}
    first_differs = np.zeros(len(tests), dtype=bool)

    for i, r in enumerate(tests):
        pos = positions(r)
        alt_answer = ANSWER[other](r["a"], r["b"])
        paired_prompt = TEMPLATE.format(a=r["a"], op=SYMBOLS[other], b=r["b"])
        vectors = {
            "paired_other": named_states(paired_prompt, pos),   # same numbers, other op
            "avg_other": canonical[other],
            "avg_same": canonical[op],
        }
        R["acc_orig"][i], R["prob_orig"][i], top_first = run_all_patches(
            r["prompt"], r["answer"], pos, vectors)
        R["acc_alt"][i], R["prob_alt"][i], _ = run_all_patches(
            r["prompt"], alt_answer, pos, vectors)

        orig_first = tokenizer(str(r["answer"]), add_special_tokens=False)["input_ids"][0]
        alt_first = tokenizer(str(alt_answer), add_special_tokens=False)["input_ids"][0]
        first_differs[i] = orig_first != alt_first
        R["first_orig"][i] = top_first == orig_first
        R["first_alt"][i] = top_first == alt_first

        if (i + 1) % 10 == 0:
            print(f"{op}: {i + 1}/{len(tests)} ({time.time() - start_time:.0f}s)", flush=True)

    R["first_differs"] = first_differs
    results[op] = R

# ---- print the blocks that matter most ----
PRINT = [("paired_other", p) for p in PATCH_SETS] + \
        [(s, p) for s in ["avg_same", "avg_other"] for p in ["operator", "final"]]
layer_header = " ".join(f"{L:4d}" for L in range(NUM_PATCH_LAYERS))
for op in results:
    R = results[op]
    keep = R["first_differs"]
    print(f"\n########## {op} prompts ({keep.sum()}/{len(keep)} have different first digits) ##########")
    for source, patch_set in PRINT:
        idx = [i for i, row in enumerate(plan) if row[0] == source and row[1] == patch_set]
        print(f"\n=== source: {source} | patched: {patch_set} ===")
        print(f"layer       {layer_header}")
        for key, label in [("acc_orig", "acc orig  "), ("acc_alt", "acc alt   "),
                           ("prob_orig", "P(orig)   "), ("prob_alt", "P(alt)    ")]:
            print(f"{label}  " + " ".join(f"{v:4.2f}" for v in R[key][:, idx].mean(axis=0)))
        for key, label in [("first_orig", "1st=orig  "), ("first_alt", "1st=alt   ")]:
            print(f"{label}  " + " ".join(f"{v:4.2f}" for v in R[key][keep][:, idx].mean(axis=0)))

np.savez("patch_positions_results.npz",
         plan=np.array([f"{s}|{p}|{L}" for s, p, L in plan]),
         **{f"{op}_{k}": v for op in results for k, v in results[op].items()})
print(f"\nsaved patch_positions_results.npz ({time.time() - start_time:.0f}s total)")