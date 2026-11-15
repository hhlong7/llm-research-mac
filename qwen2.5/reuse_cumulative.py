import json
import math
import re
import sys
import time

import numpy as np
import torch

from utilities import load_model_fp32

# usage: python reuse_cumulative.py <dataset.json> <name>
#   reads frozen_<name>.npz (step 2) and compression_<name>.npz (step 4 validation picks)
#   writes cumulative_<name>.npz
DATASET, NAME = sys.argv[1], sys.argv[2]

LAYERS = list(range(7, 16))       # hidden_states[7..15] = inputs to attention blocks 8..16
POSITIONS = ["op_pos", "b_pos"]   # operator token and last digit of b, replaced together
MAX_NEW = 6
DIM = 896
FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
VALUE_BYTES, INDEX_BYTES = 4, 2
OTHER = {"mul": "add", "add": "mul"}

model, tokenizer, device = load_model_fp32()
layers = model.model.layers

with open(DATASET) as f:
    rows = json.load(f)
frozen = np.load(f"frozen_{NAME}.npz")
picked = np.load(f"compression_{NAME}.npz")
chosen = {"same": float(picked["chosen_same"]), "global": float(picked["chosen_global"])}

# avg[(kind, pos)][L], kind = mul / add / global
avg = {(k, pos): torch.tensor(frozen[f"avg_{k}_{pos.replace('_pos', '')}"], device=device)
       for k in ["mul", "add", "global"] for pos in POSITIONS}
evals = [r for r in rows if r["split"] in ("validation", "test")]

plan = ([("baseline", None), ("mean_other", None), ("zero", None)]
        + [("same", f) for f in FRACTIONS] + [("global", f) for f in FRACTIONS])


def reconstruct(x, reference, f):
    """reference + the largest f share of (x - reference)"""
    if f == 1.0:
        return x.clone()
    residual = x - reference
    k = math.ceil(f * DIM)
    out = reference.clone()
    if k > 0:
        keep = residual.abs().topk(k).indices
        out[keep] += residual[keep]
    return out


def memory_ratio(ref, f):
    full = DIM * VALUE_BYTES
    if f == 1.0:
        return 1.0
    sites = len(LAYERS) * len(POSITIONS)
    shared = (2 if ref == "same" else 1) * full * sites / len(evals)   # averages stored once per site
    return (math.ceil(f * DIM) * (VALUE_BYTES + INDEX_BYTES) * sites + shared) / (full * sites)


def add_hooks(positions, vecs, prefill_len):
    """vecs[i] is None (leave row i alone) or {(L, pos): vector}; prompt pass only"""
    handles = []
    for L in LAYERS:
        def hook(module, inputs, output, L=L):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] < prefill_len:
                return output
            for i, v in enumerate(vecs):
                if v is not None:
                    for pos in POSITIONS:
                        hs[i, positions[pos], :] = v[(L, pos)]
            return output
        handles.append(layers[L - 1].register_forward_hook(hook))
    return handles


def greedy_decode(prompt_ids, positions, vecs):
    n = len(prompt_ids)
    ids = torch.tensor([prompt_ids], device=device).repeat(len(vecs), 1)
    handles = add_hooks(positions, vecs, n)
    try:
        with torch.no_grad():
            out = model(ids, use_cache=True)
    finally:
        for h in handles:
            h.remove()
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


def answer_prob(prompt_ids, answer, positions, vecs):
    answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    ids = torch.tensor([prompt_ids + answer_ids], device=device).repeat(len(vecs), 1)
    handles = add_hooks(positions, vecs, n)
    try:
        with torch.no_grad():
            hidden = model.model(ids).last_hidden_state[:, n - 1:n - 1 + len(answer_ids)]
            logprobs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)
    finally:
        for h in handles:
            h.remove()
    total = sum(logprobs[:, k, t] for k, t in enumerate(answer_ids))
    return total.exp().cpu().numpy()


def categorize(text, original, other):
    match = re.match(r"\s*(\d+)", text)
    if match is None:
        return "invalid"
    value = int(match.group(1))
    return "original" if value == original else "other_op" if value == other else "unrelated"


# ---- run ----
n_eval = len(evals)
cat = np.empty((n_eval, len(plan)), dtype=object)
prob_orig = np.zeros((n_eval, len(plan)))
prob_other = np.zeros((n_eval, len(plan)))
start = time.time()
for i, r in enumerate(evals):
    prompt_ids = tokenizer(r["prompt"])["input_ids"]
    positions = {pos: r[pos] for pos in POSITIONS}
    with torch.no_grad():
        hs = model(torch.tensor([prompt_ids], device=device), output_hidden_states=True).hidden_states
    # the CLEAN state at every site; each site is rebuilt from its own clean state
    clean = {(L, pos): hs[L][0, r[pos]].clone() for L in LAYERS for pos in POSITIONS}

    vecs = []
    for kind, f in plan:
        if kind == "baseline":
            vecs.append(None)
        elif kind == "mean_other":
            vecs.append({(L, pos): avg[(OTHER[r["op"]], pos)][L] for L in LAYERS for pos in POSITIONS})
        elif kind == "zero":
            vecs.append({site: torch.zeros_like(v) for site, v in clean.items()})
        else:
            ref = r["op"] if kind == "same" else "global"
            vecs.append({(L, pos): reconstruct(clean[(L, pos)], avg[(ref, pos)][L], f)
                         for L in LAYERS for pos in POSITIONS})

    decoded = greedy_decode(prompt_ids, positions, vecs)
    cat[i] = [categorize(t, r["answer"], r["other_answer"]) for t in decoded]
    prob_orig[i] = answer_prob(prompt_ids, r["answer"], positions, vecs)
    prob_other[i] = answer_prob(prompt_ids, r["other_answer"], positions, vecs)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{n_eval} prompts ({time.time() - start:.0f}s)", flush=True)

split_arr = np.array([r["split"] for r in evals])
op_arr = np.array([r["op"] for r in evals])
memory = np.array([memory_ratio(k, f) if f is not None else np.nan for k, f in plan])
np.savez(f"cumulative_{NAME}.npz", plan=np.array([f"{k}|{f}" for k, f in plan]), memory=memory,
         split=split_arr, op=op_arr, template_id=np.array([r["template_id"] for r in evals]),
         cat=cat.astype(str), prob_orig=prob_orig, prob_other=prob_other,
         chosen_same=np.array(chosen["same"]), chosen_global=np.array(chosen["global"]))
print(f"saved cumulative_{NAME}.npz ({time.time() - start:.0f}s)")

# ---- summary on the TEST set ----
test = split_arr == "test"
print(f"\n=== {NAME} | test | operator + b digit, blocks 8-16 at once | share keeping the original answer ===")
for kind in ["baseline", "mean_other", "zero"]:
    j = plan.index((kind, None))
    acc = [(cat[test & (op_arr == op), j] == "original").mean() for op in ["mul", "add"]]
    oth = [(cat[test & (op_arr == op), j] == "other_op").mean() for op in ["mul", "add"]]
    print(f"{kind:12s}  x {acc[0]:.2f}  + {acc[1]:.2f}   (other-op answer: x {oth[0]:.2f}  + {oth[1]:.2f})")
print(f"\n{'kept':>6s} {'memory':>8s} | {'same x':>7s} {'same +':>7s} | {'glob x':>7s} {'glob +':>7s}")
for f in FRACTIONS:
    line = f"{f:6.0%} {memory_ratio('same', f):8.1%} |"
    for kind in ["same", "global"]:
        j = plan.index((kind, f))
        acc = [(cat[test & (op_arr == op), j] == "original").mean() for op in ["mul", "add"]]
        mark = "*" if f == chosen[kind] else " "
        line += f" {acc[0]:6.2f}{mark} {acc[1]:6.2f}{mark} |"
    print(line)
print("* = fraction picked on validation in step 4 (single location)")
