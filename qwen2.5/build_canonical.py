import json

import numpy as np
import torch

from utilities import load_model_fp32, NUM_PATCH_LAYERS

model, tokenizer, device = load_model_fp32()

with open("patch_dataset.json") as f:
    rows = json.load(f)

def op_token_states(row):
    """operator-token hidden state at layers 0..23 -> (24, 896)"""
    ids = tokenizer(row["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    return np.stack([hs[L][0, row["op_pos"]].cpu().numpy() for L in range(NUM_PATCH_LAYERS)])

states = {}
for op in ["mul", "add"]:
    canon = [r for r in rows if r["role"] == "canonical" and r["op"] == op]
    states[op] = np.stack([op_token_states(r) for r in canon])     # (100, 24, 896)
    print(f"{op}: collected {states[op].shape} from {len(canon)} canonical prompts")

mean_mul = states["mul"].mean(axis=0)     # (24, 896)
mean_add = states["add"].mean(axis=0)

np.savez("canonical_reps.npz",
         mean_mul=mean_mul, mean_add=mean_add,
         mul_states=states["mul"], add_states=states["add"])
print("saved canonical_reps.npz")

# ---- quick look: how different are the two averages, and how typical is each prompt? ----
def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

print("\nlayer  cos(avg×, avg+)  mean cos(each × prompt, avg×)  ||avg× - avg+|| / ||avg×||")
for L in range(NUM_PATCH_LAYERS):
    typical = np.mean([cos(s, mean_mul[L]) for s in states["mul"][:, L]])
    gap = np.linalg.norm(mean_mul[L] - mean_add[L]) / np.linalg.norm(mean_mul[L])
    print(f"{L:5d}  {cos(mean_mul[L], mean_add[L]):15.4f}  {typical:29.4f}  {gap:26.3f}")


"""
op_token_states: runs one prompt through qwen and pulls out the vector at the operator
token at layers 0 to 23. layer 24 is skipped because patching it cant affect anything,
no later layer reads it
=> gives a (24, 896) array
=> stacking: 100 prompts give a (100, 24, 896) array per op

.mean(axis = 0): avg over the 100 prompts, leaving one (24, 896) "generic x" annd one 
"generic +".

np.savez: saves both averages plus all individual states for next upcoming steps

"""