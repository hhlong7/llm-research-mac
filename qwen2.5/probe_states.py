import json
import re
from collections import defaultdict

import numpy as np

from functions import load_model, get_hidden_states

model, tokenizer, device = load_model()

with open("probe_dataset.json") as f:
    rows = json.load(f)

N = len(rows)
NUM_LAYERS = 25          # embedding output + 24 transformer layers
HIDDEN = 896

last_vecs = np.zeros((N, NUM_LAYERS, HIDDEN), dtype=np.float32)
operand_vecs = np.zeros((N, NUM_LAYERS, HIDDEN), dtype=np.float32)
last_idx = np.zeros(N, dtype=np.int32)
operand_idx = np.zeros(N, dtype=np.int32)
seq_lens = np.zeros(N, dtype=np.int32)

for i, r in enumerate(rows):
    prompt = r["prompt"]

    # character position of b's last digit (b is always the last number)
    b_match = list(re.finditer(r"\d+", prompt))[-1]
    assert int(b_match.group()) == r["b"], f"last number isn't b: {prompt}"
    b_last_char = b_match.end() - 1

    # map that character to a token index using offsets
    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    op_tok = next(t for t, (s, e) in enumerate(offsets) if s <= b_last_char < e)
    assert tokenizer.decode([enc["input_ids"][op_tok]]).strip() == str(r["b"])[-1], \
        f"operand token mismatch: {prompt}"

    hs = get_hidden_states(prompt, model, tokenizer, device)
    seq_len = hs[0].shape[1]
    assert seq_len == len(enc["input_ids"]), "tokenization mismatch"

    for layer in range(NUM_LAYERS):
        last_vecs[i, layer] = hs[layer][0, -1].float().cpu().numpy()
        operand_vecs[i, layer] = hs[layer][0, op_tok].float().cpu().numpy()

    last_idx[i] = seq_len - 1
    operand_idx[i] = op_tok
    seq_lens[i] = seq_len

    if (i + 1) % 100 == 0:
        print(f"{i + 1}/{N} done")

# length-leak check: the 4 ops for the same (pair, format, wording) must be equal length
groups = defaultdict(set)
for r, L in zip(rows, seq_lens):
    groups[(r["a"], r["b"], r["format"], r["wording"])].add(int(L))
bad = {k: v for k, v in groups.items() if len(v) > 1}
print(f"\ngroups with unequal lengths across ops: {len(bad)}")

np.savez(
    "probe_states.npz",
    last=last_vecs, operand=operand_vecs,
    last_idx=last_idx, operand_idx=operand_idx, seq_len=seq_lens,
    op=np.array([r["op"] for r in rows]),
    a=np.array([r["a"] for r in rows]),
    b=np.array([r["b"] for r in rows]),
    format=np.array([r["format"] for r in rows]),
    wording=np.array([r["wording"] for r in rows]),
    split=np.array([r["split"] for r in rows]),
)
print(f"saved probe_states.npz  last={last_vecs.shape}  operand={operand_vecs.shape}")

# print exactly which token each position points at (for the write-up)
for i in [0, 4]:
    ids = tokenizer(rows[i]["prompt"])["input_ids"]
    print(f"\n[{rows[i]['format']}] {rows[i]['prompt']}")
    print(f"  operand token #{operand_idx[i]} = {tokenizer.decode([ids[operand_idx[i]]])!r}")
    print(f"  last token    #{last_idx[i]} = {tokenizer.decode([ids[last_idx[i]]])!r}")