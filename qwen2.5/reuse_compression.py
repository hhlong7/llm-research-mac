import json
import math
import re
import sys
import time

import numpy as np
import torch

from utilities import load_model_fp32

# usage: python reuse_compression.py <dataset.json> <name>
#   reads frozen_<name>.npz (step 2), writes compression_<name>.npz
DATASET, NAME = sys.argv[1], sys.argv[2]

L = 14            # hidden_states[14] = input to attention block 15
MAX_NEW = 6
DIM = 896
FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]   # share of residual entries kept
REFERENCES = ["same", "global"]       # residual measured from the same-op average or the global average
VALUE_BYTES, INDEX_BYTES = 4, 2       # float32 value + uint16 position for every kept entry
LOG_FRACTION = 0.05                   # save which entries were kept at 5%, for the outlier check later

model, tokenizer, device = load_model_fp32()
layers = model.model.layers

with open(DATASET) as f:
    rows = json.load(f)
frozen = np.load(f"frozen_{NAME}.npz")
avg = {k: torch.tensor(frozen[f"avg_{k}_b"][L], device=device) for k in ["mul", "add", "global"]}
evals = [r for r in rows if r["split"] in ("validation", "test")]

# one row per (reference, fraction), plus the clean run first
plan = [("baseline", None)] + [(ref, f) for ref in REFERENCES for f in FRACTIONS]


def memory_ratio(ref, f):
    """stored bytes for one prompt / bytes of the full 896-number state"""
    full = DIM * VALUE_BYTES
    if f == 1.0:
        return 1.0                                    # just store the original
    k = math.ceil(f * DIM)
    # the averages are stored once and shared by every eval prompt
    shared = (2 if ref == "same" else 1) * full / len(evals)
    return (k * (VALUE_BYTES + INDEX_BYTES) + shared) / full


def reconstruct(x, reference, f):
    """reference + the largest f share of (x - reference); returns the vector and the kept positions"""
    if f == 1.0:
        return x.clone(), torch.arange(DIM, device=x.device)
    residual = x - reference
    k = math.ceil(f * DIM)
    keep = residual.abs().topk(k).indices if k > 0 else torch.tensor([], dtype=torch.long, device=x.device)
    out = reference.clone()
    out[keep] += residual[keep]
    return out, keep


def add_hook(pos, vecs, prefill_len):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] < prefill_len:
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


# ---- run ----
n_eval = len(evals)
cat = np.empty((n_eval, len(plan)), dtype=object)
prob_orig = np.zeros((n_eval, len(plan)))
prob_other = np.zeros((n_eval, len(plan)))
states = np.zeros((n_eval, DIM), dtype=np.float32)                     # each prompt's clean b state
kept_at_log = {ref: np.zeros((n_eval, math.ceil(LOG_FRACTION * DIM)), dtype=np.int64) for ref in REFERENCES}
start = time.time()
for i, r in enumerate(evals):
    prompt_ids = tokenizer(r["prompt"])["input_ids"]
    with torch.no_grad():
        hs = model(torch.tensor([prompt_ids], device=device), output_hidden_states=True).hidden_states
    x = hs[L][0, r["b_pos"]].clone()
    states[i] = x.float().cpu().numpy()

    vecs = []
    for ref, f in plan:
        if ref == "baseline":
            vecs.append(None)
            continue
        reference = avg[r["op"]] if ref == "same" else avg["global"]
        v, keep = reconstruct(x, reference, f)
        vecs.append(v)
        if f == LOG_FRACTION:
            kept_at_log[ref][i] = keep.cpu().numpy()

    decoded = greedy_decode(prompt_ids, r["b_pos"], vecs)
    cat[i] = [categorize(t, r["answer"], r["other_answer"]) for t in decoded]
    prob_orig[i] = answer_prob(prompt_ids, r["answer"], r["b_pos"], vecs)
    prob_other[i] = answer_prob(prompt_ids, r["other_answer"], r["b_pos"], vecs)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{n_eval} prompts ({time.time() - start:.0f}s)", flush=True)

split_arr = np.array([r["split"] for r in evals])
op_arr = np.array([r["op"] for r in evals])
memory = np.array([np.nan if ref == "baseline" else memory_ratio(ref, f) for ref, f in plan])

# ---- validation picks the fraction: smallest one that loses at most 1 validation prompt
#      the model originally got right (Tyler and Julian's rule); test is never used here ----
val = split_arr == "validation"
base_ok = cat[:, 0] == "original"
chosen = {}
for ref in REFERENCES:
    for f in FRACTIONS:
        j = plan.index((ref, f))
        lost = (val & base_ok & (cat[:, j] != "original")).sum()
        if lost <= 1:
            chosen[ref] = f
            break
    print(f"validation picked {chosen[ref]:.0%} for the {ref}-op reference"
          if ref == "same" else f"validation picked {chosen[ref]:.0%} for the global reference")

np.savez(f"compression_{NAME}.npz", plan=np.array([f"{ref}|{f}" for ref, f in plan]), memory=memory,
         split=split_arr, op=op_arr, template_id=np.array([r["template_id"] for r in evals]),
         cat=cat.astype(str), prob_orig=prob_orig, prob_other=prob_other, states=states,
         kept_same=kept_at_log["same"], kept_global=kept_at_log["global"],
         chosen_same=np.array(chosen["same"]), chosen_global=np.array(chosen["global"]))
print(f"saved compression_{NAME}.npz ({time.time() - start:.0f}s)")

# ---- summary on the TEST set ----
test = split_arr == "test"
print(f"\n=== {NAME} | test | b digit, hidden_states[{L}] | share of prompts keeping the original answer ===")
print(f"{'kept':>6s} {'memory':>8s} | {'same x':>7s} {'same +':>7s} {'same all':>8s} | "
      f"{'glob x':>7s} {'glob +':>7s} {'glob all':>8s}")
for f in FRACTIONS:
    line = f"{f:6.0%} {memory_ratio('same', f):8.1%} |"
    for ref in REFERENCES:
        j = plan.index((ref, f))
        acc = [(cat[test & (op_arr == op), j] == "original").mean() for op in ["mul", "add"]]
        line += f" {acc[0]:7.2f} {acc[1]:7.2f} {(cat[test, j] == 'original').mean():8.2f} |"
    print(line)
