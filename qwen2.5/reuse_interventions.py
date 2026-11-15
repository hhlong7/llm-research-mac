import json
import random
import re
import sys
import time

import numpy as np
import torch

from utilities import load_model_fp32

# usage: python reuse_interventions.py <dataset.json> <name>
#   reads frozen_<name>.npz (step 2), writes interventions_<name>.npz
DATASET, NAME = sys.argv[1], sys.argv[2]

L = 14            # hidden_states[14] = input to attention block 15 (output of model.layers[13])
MAX_NEW = 6       # greedy tokens (answers are at most 3 digits)
SEED = 0
OTHER = {"mul": "add", "add": "mul"}
CATEGORIES = ["original", "other_op", "unrelated", "invalid"]

# what goes into the b-digit slot for each condition (x = this prompt's own clean state)
CONDITIONS = [
    "baseline",          # no swap
    "self",              # swap in x itself: must equal baseline, checks the hook
    "donor_same",        # b state of a random TRAIN prompt with the same op
    "donor_other",       # b state of a random TRAIN prompt with the other op
    "mean_same",         # frozen same-op average
    "mean_other",        # frozen other-op average
    "mean_global",       # frozen average over both ops (new, option C)
    "delta_plus",        # x + (mul avg - add avg)
    "delta_minus",       # x - (mul avg - add avg)
    "random_state",      # fixed random vector, same size as the same-op average
    "random_direction",  # x + random direction, same size as (mul avg - add avg)
    "zero",              # all zeros
]

model, tokenizer, device = load_model_fp32()
layers = model.model.layers

with open(DATASET) as f:
    rows = json.load(f)
frozen = np.load(f"frozen_{NAME}.npz")
avg = {k: torch.tensor(frozen[f"avg_{k}_b"][L], device=device) for k in ["mul", "add", "global"]}
diff = torch.tensor(frozen["diff_b"][L], device=device)

train = [r for r in rows if r["split"] == "train"]
evals = [r for r in rows if r["split"] in ("validation", "test")]


def clean_states(prompt):
    with torch.no_grad():
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        return model(**ids, output_hidden_states=True).hidden_states


# donor bank: the b-digit state of every train prompt (train only, so test is never a donor)
donors = {"mul": [], "add": []}
for r in train:
    donors[r["op"]].append(clean_states(r["prompt"])[L][0, r["b_pos"]].clone())

# fixed random vectors (same for every prompt, so they behave like a stored state)
gen = torch.Generator().manual_seed(SEED)
unit = lambda v: v / v.norm()
random_state = {op: unit(torch.randn(896, generator=gen)).to(device) * avg[op].norm() for op in ["mul", "add"]}
random_dir = unit(torch.randn(896, generator=gen)).to(device) * diff.norm()


def replacements(r, x, rng):
    """one vector (or None = leave alone) per condition"""
    op = r["op"]
    return {
        "baseline": None,
        "self": x,
        "donor_same": rng.choice(donors[op]),
        "donor_other": rng.choice(donors[OTHER[op]]),
        "mean_same": avg[op],
        "mean_other": avg[OTHER[op]],
        "mean_global": avg["global"],
        "delta_plus": x + diff,
        "delta_minus": x - diff,
        "random_state": random_state[op],
        "random_direction": x + random_dir,
        "zero": torch.zeros_like(x),
    }


def add_hook(pos, vecs, prefill_len):
    """put vecs[i] into row i at position pos of hidden_states[L]; prompt pass (prefill) only"""
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] < prefill_len:          # a decode step: leave it alone
            return output
        for i, v in enumerate(vecs):
            if v is not None:
                hs[i, pos, :] = v
        return output
    return layers[L - 1].register_forward_hook(hook)


def greedy_decode(prompt_ids, pos, vecs):
    n = len(prompt_ids)
    ids = torch.tensor([prompt_ids], device=device).repeat(len(vecs), 1)
    handle = add_hook(pos, vecs, n)
    try:
        with torch.no_grad():
            out = model(ids, use_cache=True)
    finally:
        handle.remove()
    cache = out.past_key_values
    next_tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_tok]
    with torch.no_grad():
        for _ in range(MAX_NEW - 1):
            out = model(next_tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
    return [tokenizer.decode(g) for g in torch.cat(generated, dim=1).tolist()]


def answer_prob(prompt_ids, answer, pos, vecs):
    """teacher-forced P(answer digits | prompt) for every row, digits only (same as block15_hidden.py)"""
    answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    ids = torch.tensor([prompt_ids + answer_ids], device=device).repeat(len(vecs), 1)
    handle = add_hook(pos, vecs, n)
    try:
        with torch.no_grad():
            hidden = model.model(ids).last_hidden_state[:, n - 1:n - 1 + len(answer_ids)]
            logprobs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)
    finally:
        handle.remove()
    total = sum(logprobs[:, k, t] for k, t in enumerate(answer_ids))
    return total.exp().cpu().numpy()


def categorize(text, original, other):
    match = re.match(r"\s*(\d+)", text)
    if match is None:
        return "invalid"
    value = int(match.group(1))
    return "original" if value == original else "other_op" if value == other else "unrelated"


# ---- run every condition on every validation and test prompt ----
n_eval, n_cond = len(evals), len(CONDITIONS)
cat = np.empty((n_eval, n_cond), dtype=object)
prob_orig = np.zeros((n_eval, n_cond))
prob_other = np.zeros((n_eval, n_cond))
texts = []
start = time.time()
for i, r in enumerate(evals):
    rng = random.Random(f"{SEED}-{r['prompt']}")        # same donors every run
    prompt_ids = tokenizer(r["prompt"])["input_ids"]
    x = clean_states(r["prompt"])[L][0, r["b_pos"]].clone()
    reps = replacements(r, x, rng)
    vecs = [reps[c] for c in CONDITIONS]

    decoded = greedy_decode(prompt_ids, r["b_pos"], vecs)
    cat[i] = [categorize(t, r["answer"], r["other_answer"]) for t in decoded]
    texts.append(decoded)
    prob_orig[i] = answer_prob(prompt_ids, r["answer"], r["b_pos"], vecs)
    prob_other[i] = answer_prob(prompt_ids, r["other_answer"], r["b_pos"], vecs)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{n_eval} prompts ({time.time() - start:.0f}s)", flush=True)

assert (cat[:, CONDITIONS.index("self")] == cat[:, 0]).all(), "self swap changed the answer: hook is wrong"

np.savez(f"interventions_{NAME}.npz", conditions=np.array(CONDITIONS),
         split=np.array([r["split"] for r in evals]), op=np.array([r["op"] for r in evals]),
         template_id=np.array([r["template_id"] for r in evals]), prompt=np.array([r["prompt"] for r in evals]),
         cat=cat.astype(str), prob_orig=prob_orig, prob_other=prob_other, text=np.array(texts))
print(f"saved interventions_{NAME}.npz ({time.time() - start:.0f}s)")

# ---- summary: fraction of prompts in each category, per split and op ----
split_arr = np.array([r["split"] for r in evals])
op_arr = np.array([r["op"] for r in evals])
for split in ["validation", "test"]:
    for op in ["mul", "add"]:
        keep = (split_arr == split) & (op_arr == op)
        print(f"\n=== {NAME} | {split} | {op} prompts ({keep.sum()}) | b digit, hidden_states[{L}] ===")
        print(f"{'condition':18s} {'original':>9s} {'other_op':>9s} {'unrelated':>9s} {'invalid':>8s} {'P(orig)':>8s} {'P(other)':>8s}")
        for j, c in enumerate(CONDITIONS):
            fr = [(cat[keep, j] == k).mean() for k in CATEGORIES]
            print(f"{c:18s} {fr[0]:9.2f} {fr[1]:9.2f} {fr[2]:9.2f} {fr[3]:8.2f} "
                  f"{prob_orig[keep, j].mean():8.2f} {prob_other[keep, j].mean():8.2f}")
