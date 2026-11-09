import json
import re
import time

import numpy as np
import torch

from utilities import load_model_fp32, TEMPLATE, SYMBOLS

LAYERS = [12, 13, 14, 15, 16, 17]   # hidden_states index; [L] is the input to model.layers[L]
MAX_NEW = 6                         # greedy tokens to generate (answers are at most 3 digits)
OTHER = {"mul": "add", "add": "mul"}
ANSWER = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}
PATCH_SETS = {
    "operator": ["operator"],
    "b_last": ["b_last"],
    "final": ["final"],
    "all_three": ["operator", "b_last", "final"],
}
CATEGORIES = ["original", "other_op", "unrelated", "invalid"]

model, tokenizer, device = load_model_fp32()
layers = model.model.layers

with open("patch_dataset.json") as f:
    rows = json.load(f)


# ---- 1. layer numbering check ----
# hidden_states[L] should equal the output of model.layers[L-1], i.e. the input to model.layers[L]
check_prompt = TEMPLATE.format(a=6, op=SYMBOLS["mul"], b=7)
captured = {}
handles = [layers[i].register_forward_hook(
    lambda m, inp, out, i=i: captured.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach()))
    for i in (13, 14)]
with torch.no_grad():
    hs = model(**tokenizer(check_prompt, return_tensors="pt").to(device),
               output_hidden_states=True).hidden_states
for h in handles:
    h.remove()
for i in (13, 14):
    print(f"hidden_states[{i + 1}] == output of model.layers[{i}]: "
          f"{torch.allclose(hs[i + 1], captured[i], atol=1e-5)}")
print("so attention block 15 (counting from 1) is model.layers[14]:"
      " input = hidden_states[14], output = hidden_states[15]\n")


def positions(row):
    n = len(tokenizer(row["prompt"])["input_ids"])
    return {"operator": row["op_pos"], "b_last": n - 3, "final": n - 1}


def donor_states(prompt, pos):
    """hidden states of the donor prompt at each named token -> {name: {layer: vec}}"""
    with torch.no_grad():
        hs = model(**tokenizer(prompt, return_tensors="pt").to(device),
                   output_hidden_states=True).hidden_states
    return {name: {L: hs[L][0, p].clone() for L in LAYERS} for name, p in pos.items()}


# row 0 is the clean run, then one row per (patch set, layer)
plan = [("clean", None)] + [(p, L) for p in PATCH_SETS for L in LAYERS]


def add_hooks(pos, donor, prefill_len):
    """patch each row at its layer; only fires on the prompt pass (prefill)"""
    handles = []
    for L in LAYERS:
        row_ids = [i for i, (p, layer) in enumerate(plan) if layer == L]

        def hook(module, inputs, output, L=L, row_ids=row_ids):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] < prefill_len:          # a decode step: leave it alone
                return output
            for i in row_ids:
                for name in PATCH_SETS[plan[i][0]]:
                    hs[i, pos[name], :] = donor[name][L]
            return output

        target = model.model.embed_tokens if L == 0 else layers[L - 1]
        handles.append(target.register_forward_hook(hook))
    return handles


def greedy_decode(prompt_ids, pos, donor):
    """generate MAX_NEW tokens for every row; the patch applies during prefill only"""
    n = len(prompt_ids)
    ids = torch.tensor([prompt_ids], device=device).repeat(len(plan), 1)
    handles = add_hooks(pos, donor, n)
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
    generated = torch.cat(generated, dim=1).tolist()
    return [tokenizer.decode(g) for g in generated], [g[0] for g in generated]


def answer_prob(prompt_ids, answer, pos, donor):
    """teacher-forced P(answer digits | prompt) for every row, digits only"""
    answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    ids = torch.tensor([prompt_ids + answer_ids], device=device).repeat(len(plan), 1)
    handles = add_hooks(pos, donor, n)
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
    if value == original:
        return "original"
    if value == other:
        return "other_op"
    return "unrelated"


# ---- 2. the experiment ----
results = {}
examples = {}
start_time = time.time()
for op in ["mul", "add"]:
    other = OTHER[op]
    tests = [r for r in rows if r["role"] == "test" and r["op"] == op]
    cat = np.empty((len(tests), len(plan)), dtype=object)
    prob_orig = np.zeros((len(tests), len(plan)))
    prob_alt = np.zeros((len(tests), len(plan)))
    first_orig = np.zeros((len(tests), len(plan)), dtype=bool)
    first_alt = np.zeros((len(tests), len(plan)), dtype=bool)
    first_differs = np.zeros(len(tests), dtype=bool)
    baseline_ok = np.zeros(len(tests), dtype=bool)
    texts = []

    for i, r in enumerate(tests):
        pos = positions(r)
        alt = ANSWER[other](r["a"], r["b"])
        donor_prompt = TEMPLATE.format(a=r["a"], op=SYMBOLS[other], b=r["b"])
        donor = donor_states(donor_prompt, pos)
        prompt_ids = tokenizer(r["prompt"])["input_ids"]

        decoded, first_tok = greedy_decode(prompt_ids, pos, donor)
        cat[i] = [categorize(t, r["answer"], alt) for t in decoded]
        texts.append(decoded)

        # the donor prompt must also be answered correctly with no patch
        donor_ids = tokenizer(donor_prompt)["input_ids"]
        with torch.no_grad():
            g = model.generate(torch.tensor([donor_ids], device=device), max_new_tokens=MAX_NEW,
                               do_sample=False, repetition_penalty=1.0,   # Qwen's default 1.1 penalizes reusing prompt digits
                               pad_token_id=tokenizer.eos_token_id)
        donor_text = tokenizer.decode(g[0, len(donor_ids):])
        baseline_ok[i] = cat[i, 0] == "original" and categorize(donor_text, alt, r["answer"]) == "original"

        prob_orig[i] = answer_prob(prompt_ids, r["answer"], pos, donor)
        prob_alt[i] = answer_prob(prompt_ids, alt, pos, donor)

        orig_first = tokenizer(str(r["answer"]), add_special_tokens=False)["input_ids"][0]
        alt_first = tokenizer(str(alt), add_special_tokens=False)["input_ids"][0]
        first_differs[i] = orig_first != alt_first
        first_orig[i] = np.array(first_tok) == orig_first
        first_alt[i] = np.array(first_tok) == alt_first

        if i < 2:
            print(f"example {r['prompt']!r} -> clean {decoded[0]!r}, donor {donor_prompt!r} -> {donor_text!r}")
        if (i + 1) % 20 == 0:
            print(f"{op}: {i + 1}/{len(tests)} ({time.time() - start_time:.0f}s)", flush=True)

    results[op] = dict(cat=cat, prob_orig=prob_orig, prob_alt=prob_alt, first_orig=first_orig,
                       first_alt=first_alt, first_differs=first_differs, baseline_ok=baseline_ok)
    examples[op] = texts

# ---- 3. print the results ----
DIRECTION = {"mul": "× prompts, + donor (mul -> add)", "add": "+ prompts, × donor (add -> mul)"}
header = " ".join(f"{L:>6d}" for L in LAYERS)
for op, R in results.items():
    keep = R["baseline_ok"]
    fd = keep & R["first_differs"]
    print(f"\n########## {DIRECTION[op]}: {keep.sum()}/{len(keep)} baseline-correct pairs, "
          f"{fd.sum()} with different first digits ##########")
    for patch_set in PATCH_SETS:
        idx = [j for j, (p, L) in enumerate(plan) if p == patch_set]
        print(f"\n=== patched: {patch_set} ===")
        print(f"hidden_states[L]   {header}")
        print(f"(block L+1, 1-based) " + " ".join(f"{L + 1:>6d}" for L in LAYERS)[1:])
        for c in CATEGORIES:
            vals = (R["cat"][keep][:, idx] == c).mean(axis=0)
            print(f"{c:<18s} " + " ".join(f"{v:6.2f}" for v in vals))
        for key, label in [("prob_orig", "P(orig)"), ("prob_alt", "P(other)")]:
            print(f"{label:<18s} " + " ".join(f"{v:6.2f}" for v in R[key][keep][:, idx].mean(axis=0)))
        for key, label in [("first_orig", "1st digit = orig"), ("first_alt", "1st digit = other")]:
            print(f"{label:<18s} " + " ".join(f"{v:6.2f}" for v in R[key][fd][:, idx].mean(axis=0)))

np.savez("block15_hidden_results.npz",
         plan=np.array([f"{p}|{L}" for p, L in plan]), layers=np.array(LAYERS),
         **{f"{op}_{k}": (v.astype(str) if k == "cat" else v) for op in results for k, v in results[op].items()},
         **{f"{op}_text": np.array(examples[op]) for op in examples})
print(f"\nsaved block15_hidden_results.npz ({time.time() - start_time:.0f}s total)")
