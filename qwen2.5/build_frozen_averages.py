import json
import sys

import numpy as np
import torch

from utilities import load_model_fp32

# usage: python build_frozen_averages.py <dataset.json> <name>
#   e.g. python build_frozen_averages.py reuse_dataset.json ours   -> frozen_ours.npz
DATASET, NAME = sys.argv[1], sys.argv[2]

model, tokenizer, device = load_model_fp32()

with open(DATASET) as f:
    rows = json.load(f)
train = [r for r in rows if r["split"] == "train"]      # validation and test prompts are never touched here

# collect the hidden state at every layer for the operator and b positions
states = {(op, pos): [] for op in ["mul", "add"] for pos in ["op_pos", "b_pos"]}
for r in train:
    ids = tokenizer(r["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states   # 25 x (1, seq, 896)
    hs = torch.stack(hs)[:, 0].float().cpu().numpy()                  # (25, seq, 896)
    for pos in ["op_pos", "b_pos"]:
        states[(r["op"], pos)].append(hs[:, r[pos]])                  # (25, 896)

out = {}
for (op, pos), vecs in states.items():
    vecs = np.stack(vecs)                                             # (num prompts, 25, 896)
    name = pos.replace("_pos", "")
    out[f"avg_{op}_{name}"] = vecs.mean(axis=0)                       # (25, 896)
    out[f"norm_{op}_{name}"] = np.linalg.norm(vecs, axis=-1).mean(0)  # (25,) typical size, for the random control
for name in ["op", "b"]:
    out[f"diff_{name}"] = out[f"avg_mul_{name}"] - out[f"avg_add_{name}"]   # the mul - add direction
    # one average over BOTH operations (same number of + and x prompts, so this is the midpoint)
    out[f"avg_global_{name}"] = (out[f"avg_mul_{name}"] + out[f"avg_add_{name}"]) / 2
out["num_train"] = np.array(len(train))
out["dataset"] = np.array(DATASET)

np.savez(f"frozen_{NAME}.npz", **out)
print(f"built from {len(train)} train prompts of {DATASET}, saved frozen_{NAME}.npz")

# sanity check at the input to block 15 (hidden_states[14]), train prompts only
L = 14
for name in ["op", "b"]:
    mul, add, diff = out[f"avg_mul_{name}"][L], out[f"avg_add_{name}"][L], out[f"diff_{name}"][L]
    cos = mul @ add / (np.linalg.norm(mul) * np.linalg.norm(add))
    vecs = {op: np.stack(states[(op, f"{name}_pos")])[:, L] for op in ["mul", "add"]}
    other = {"mul": "add", "add": "mul"}
    dist = lambda v, avg: np.linalg.norm(v - avg, axis=-1)
    right = sum((dist(v, out[f"avg_{op}_{name}"][L]) < dist(v, out[f"avg_{other[op]}_{name}"][L])).sum()
                for op, v in vecs.items())
    to_own = np.mean([dist(v, out[f"avg_{op}_{name}"][L]).mean() for op, v in vecs.items()])
    to_global = np.mean([dist(v, out[f"avg_global_{name}"][L]).mean() for op, v in vecs.items()])
    print(f"\n{name} position, hidden_states[{L}] (input to block 15):")
    print(f"  size of one state                    {out[f'norm_mul_{name}'][L]:.1f}")
    print(f"  size of mul - add                    {np.linalg.norm(diff):.1f}")
    print(f"  avg distance to own op average       {to_own:.1f}   (the prompt-specific part)")
    print(f"  avg distance to global average       {to_global:.1f}")
    print(f"  cosine(mul avg, add avg)             {cos:.3f}")
    print(f"  train states closer to own op average: {right}/{2 * len(vecs['mul'])}")
