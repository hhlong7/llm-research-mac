import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

data = np.load("probe_states.npz")
op, split, fmt = data["op"], data["split"], data["format"]

OPS = ["add", "sub", "mul", "div"]
NUM_LAYERS = 25
POSITIONS = ["last", "operand"]
DIRECTIONS = [("symbolic", "natural"), ("natural", "symbolic")]

# load the big arrays into memory once (npz re-reads from disk on every access)
states = {pos: data[pos] for pos in POSITIONS}

def make_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))

def transfer(src, dst, pos, layer, center=False):
    """train on src-format train pairs, test on dst-format held-out pairs"""
    X = states[pos][:, layer]
    train = (split == "train") & (fmt == src)
    test = (split == "test") & (fmt == dst)
    X_train, X_test = X[train], X[test]
    if center:
        # subtract each format's own mean vector (uses no labels)
        X_train = X_train - X_train.mean(axis=0)
        X_test = X_test - X_test.mean(axis=0)
    probe = make_probe().fit(X_train, op[train])
    pred = probe.predict(X_test)
    return (pred == op[test]).mean(), pred, op[test]

# ---- run every direction / position / layer, raw and centered ----
results = {}
for src, dst in DIRECTIONS:
    for pos in POSITIONS:
        for center in [False, True]:
            key = f"{src[:3]}->{dst[:3]}/{pos}" + ("/centered" if center else "")
            results[key] = [transfer(src, dst, pos, layer, center)[0]
                            for layer in range(NUM_LAYERS)]

# ---- tables, one per position ----
for pos in POSITIONS:
    cols = [k for k in results if f"/{pos}" in k]
    print(f"\n=== position: {pos} ===")
    print("layer  " + "  ".join(f"{c.replace('/' + pos, ''):>20s}" for c in cols))
    for layer in range(NUM_LAYERS):
        print(f"{layer:5d}  " + "  ".join(f"{results[c][layer]:20.3f}" for c in cols))

# ---- confusion matrices: symbolic -> natural, last token, raw ----
for layer in [2, 8, 14, 20]:
    acc, pred, true = transfer("symbolic", "natural", "last", layer)
    cm = confusion_matrix(true, pred, labels=OPS)
    print(f"\nsym->nat/last, layer {layer}: acc {acc:.3f}")
    print("  rows = true, cols = predicted  " + " ".join(f"{o:>4s}" for o in OPS))
    for i, o in enumerate(OPS):
        print(f"  {o:>4s}" + " " * 27 + " ".join(f"{n:4d}" for n in cm[i]))

with open("cross_wording_results.json", "w") as fp:
    json.dump(results, fp, indent=2)

# ---- plot ----
with open("probe_results.json") as fp:
    within = json.load(fp)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, pos in zip(axes, POSITIONS):
    ax.plot(within[f"symbolic/{pos}"], color="gray", alpha=0.5, label="within symbolic (step 3)")
    ax.plot(results[f"sym->nat/{pos}"], marker="o", ms=3, label="symbolic -> natural")
    ax.plot(results[f"sym->nat/{pos}/centered"], marker="o", ms=3, label="symbolic -> natural (centered)")
    ax.plot(results[f"nat->sym/{pos}"], marker="o", ms=3, label="natural -> symbolic")
    ax.axhline(0.25, color="gray", ls="--", label="chance (25%)")
    ax.set_title(f"cross-wording transfer, {pos} token (held-out operand pairs)")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test accuracy (operation)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("cross_wording_by_layer.png", dpi=150)
print("\nsaved cross_wording_by_layer.png and cross_wording_results.json")


"""
Difference train probe n cross probing:
transfer(): the train and test masks now come from different formats. 
The train rows are symbolic + train pairs and the test rows 
are natural + test pairs, so both the operands and the wording are new at test time.
states = {...}: a speed fix. 
Each data["last"] access re-reads 100MB from the npz, 
and this script calls it 200 times, so we load the arrays once.
center=True: the offset check described above.
Confusion matrices: if transfer is partial, 
these show which ops carry over. For example, 
add and mul might transfer while sub and div get confused with each other.
The plot puts the step 3 within-symbolic 
curve in gray as the ceiling, so you can see 
how much transfer costs at each layer.

"""