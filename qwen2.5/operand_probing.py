import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

data = np.load("probe_states.npz")
split, fmt, op, wording = data["split"], data["format"], data["op"], data["wording"]
targets = {"a": data["a"].astype(float), "b": data["b"].astype(float)}

OPS = ["add", "sub", "mul", "div"]
WORDINGS = sorted(set(wording))
NUM_LAYERS = 25
POSITIONS = ["last", "operand"]
FORMATS = ["symbolic", "natural"]

states = {pos: data[pos] for pos in POSITIONS}

def make_probe():
    # RidgeCV picks the regularization strength itself from this list
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11)))

# ---- operand probes: train on 35 pairs, test on 15 held-out pairs ----
results = {}
for f in FORMATS:
    train = (split == "train") & (fmt == f)
    test = (split == "test") & (fmt == f)
    for pos in POSITIONS:
        for name, y in targets.items():
            r2s, maes = [], []
            for layer in range(NUM_LAYERS):
                X = states[pos][:, layer]
                probe = make_probe().fit(X[train], y[train])
                pred = probe.predict(X[test])
                r2s.append(r2_score(y[test], pred))
                maes.append(mean_absolute_error(y[test], pred))
            results[f"{f}/{pos}/{name}"] = {"r2": r2s, "mae": maes}

# ---- operand-only cosine similarity at the last token ----
def mean_pairwise_cos(V):
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    n = len(V)
    return (S.sum() - n) / (n * (n - 1))     # average of off-diagonal entries

cos = {}
for f in FORMATS:
    per_layer = []
    for layer in range(NUM_LAYERS):
        # each group = same format, op, wording; only the operands differ
        vals = [mean_pairwise_cos(states["last"][(fmt == f) & (op == o) & (wording == w), layer])
                for o in OPS for w in WORDINGS]
        per_layer.append(float(np.mean(vals)))
    cos[f] = per_layer

# ---- tables ----
for f in FORMATS:
    cols = [f"{pos}/{name}" for pos in POSITIONS for name in targets]
    print(f"\n=== {f}: R^2 on held-out pairs (and last-token operand-only cosine) ===")
    print("layer  " + "  ".join(f"{c:>12s}" for c in cols) + "  cos(last)")
    for layer in range(NUM_LAYERS):
        r2_row = "  ".join(f"{results[f'{f}/{c}']['r2'][layer]:12.3f}" for c in cols)
        print(f"{layer:5d}  {r2_row}  {cos[f][layer]:9.4f}")

print("\nmean absolute error (in units) at a few layers, symbolic:")
for layer in [0, 4, 12, 20, 24]:
    print(f"  layer {layer:2d}: " + "  ".join(
        f"{pos}/{name}={results[f'symbolic/{pos}/{name}']['mae'][layer]:.2f}"
        for pos in POSITIONS for name in targets))

with open("operand_probe_results.json", "w") as fp:
    json.dump({"probes": results, "cosine_last": cos}, fp, indent=2)

# ---- plot: probe R^2 (left axis) vs cosine similarity (right axis) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, f in zip(axes, FORMATS):
    for pos in POSITIONS:
        for name in targets:
            ax.plot(results[f"{f}/{pos}/{name}"]["r2"], marker="o", ms=3,
                    label=f"R² {name} @ {pos} token")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_title(f"{f}: operand decoding on held-out pairs")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(cos[f], color="black", ls="--", label="operand-only cosine (last token)")
    ax2.set_ylim(0, 1.05)
    if f == FORMATS[-1]:
        ax2.set_ylabel("cosine similarity")
    ax2.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("probe R² (held-out pairs)")
axes[0].legend(loc="center right", fontsize=8)
plt.tight_layout()
plt.savefig("operand_probes_by_layer.png", dpi=150)
print("\nsaved operand_probes_by_layer.png and operand_probe_results.json")


"""
Why regression this time

For the operation we used a classifier, because there were 4 categories. 
Operands are numbers from 2 to 30, and because we hold out whole pairs, 
a test prompt’s operand pair is new to the probe. 
So we use ridge regression: a linear probe that outputs a number and is scored with:

R²: 1.0 is perfect, 0 is no better than always guessing the average, and below 0 is worse than that.
MAE (mean absolute error): on average, how many units the prediction is off.

"""