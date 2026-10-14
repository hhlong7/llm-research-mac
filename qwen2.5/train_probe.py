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
FORMATS = ["symbolic", "natural"]

def make_probe():
    # scale each dimension, then a regularized linear classifier
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
"""
StandardScaler puts every dimension on the same scale. 
That matters because we already know a few dimensions have huge magnitudes, 
which is why mean pooling saturated. 
LogisticRegression is the linear probe: it learns one weight vector 
per operation and picks the highest score.

"""
"""
The main loop: a fresh probe is trained per layer, per format, per position, for 100 probes total. 
Training uses only split == "train" rows and scoring uses only held-out-pair rows, 
so every test number pair is new to the probe.
Confusion matrix: shows which ops get mixed up, e.g. whether add and sub 
get confused more than add and mul. 
Shuffled-label control: we train on randomly scrambled labels. 
If that still scored well above 25%, the setup would be leaking something. 
It should land near chance.

"""
# ---- train/test a probe at every layer ----
results = {}
for f in FORMATS:
    train = (split == "train") & (fmt == f)
    test = (split == "test") & (fmt == f)
    for pos in POSITIONS:
        X = data[pos]
        accs = []
        for layer in range(NUM_LAYERS):
            probe = make_probe()
            probe.fit(X[train, layer], op[train])
            accs.append(probe.score(X[test, layer], op[test]))
        results[f"{f}/{pos}"] = accs

# ---- accuracy table ----
keys = list(results)
print("layer  " + "  ".join(f"{k:>17s}" for k in keys))
for layer in range(NUM_LAYERS):
    print(f"{layer:5d}  " + "  ".join(f"{results[k][layer]:17.3f}" for k in keys))

# ---- per-op accuracy + confusion matrix at a few layers (symbolic, last token) ----
train = (split == "train") & (fmt == "symbolic")
test = (split == "test") & (fmt == "symbolic")
for layer in [0, 4, 12, 20]:
    probe = make_probe().fit(data["last"][train, layer], op[train])
    pred = probe.predict(data["last"][test, layer])
    cm = confusion_matrix(op[test], pred, labels=OPS)
    per_op = {o: cm[i, i] / cm[i].sum() for i, o in enumerate(OPS)}
    print(f"\nsymbolic/last, layer {layer}: per-op acc " +
          "  ".join(f"{o}={a:.2f}" for o, a in per_op.items()))
    print("  rows = true, cols = predicted  " + " ".join(f"{o:>4s}" for o in OPS))
    for i, o in enumerate(OPS):
        print(f"  {o:>4s}" + " " * 27 + " ".join(f"{n:4d}" for n in cm[i]))

# ---- control: shuffled labels should give ~25% ----
rng = np.random.default_rng(0)
shuffled = rng.permutation(op[train])
probe = make_probe().fit(data["last"][train, 12], shuffled)
print(f"\ncontrol (shuffled train labels, symbolic/last, layer 12): "
      f"{probe.score(data['last'][test, 12], op[test]):.3f}")

with open("probe_results.json", "w") as fp:
    json.dump(results, fp, indent=2)

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, f in zip(axes, FORMATS):
    for pos in POSITIONS:
        ax.plot(range(NUM_LAYERS), results[f"{f}/{pos}"], marker="o", ms=3,
                label=f"{pos} token")
    ax.axhline(0.25, color="gray", ls="--", label="chance (25%)")
    ax.set_title(f"{f} prompts: train on 35 pairs, test on 15 held-out pairs")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test accuracy (operation)")
axes[0].legend()
plt.tight_layout()
plt.savefig("probe_accuracy_by_layer.png", dpi=150)
print("\nsaved probe_accuracy_by_layer.png and probe_results.json")