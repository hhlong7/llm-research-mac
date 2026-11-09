import numpy as np
import matplotlib.pyplot as plt

d = np.load("block15_hidden_results.npz", allow_pickle=True)
plan = list(d["plan"])
LAYERS = list(d["layers"])
PATCH_SETS = ["operator", "b_last", "final", "all_three"]
TITLES = {"operator": "operator", "b_last": "last digit of b", "final": "final token",
          "all_three": "all three"}
DIRECTIONS = [("mul", "× prompts, + donor"), ("add", "+ prompts, × donor")]
CATS = [("original", "original answer", "#2a78d6"),
        ("other_op", "other-op answer", "#eb6834"),
        ("unrelated", "unrelated", "#b0aca4"),
        ("invalid", "invalid", "#4a4842")]

fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharey=True)
for r, (op, label) in enumerate(DIRECTIONS):
    cat = d[f"{op}_cat"][d[f"{op}_baseline_ok"]]
    for c, patch_set in enumerate(PATCH_SETS):
        ax = axes[r, c]
        idx = [plan.index(f"{patch_set}|{L}") for L in LAYERS]
        x = np.arange(len(LAYERS))
        bottom = np.zeros(len(LAYERS))
        for key, name, color in CATS:
            frac = (cat[:, idx] == key).mean(axis=0)
            ax.bar(x, frac, bottom=bottom, color=color, width=0.8, label=name,
                   edgecolor="white", linewidth=1.5)
            bottom += frac
        ax.axvline(2.5, color="#1f1e1c", linestyle=":", linewidth=1.2)   # between input and output of block 15
        ax.set_xticks(x)
        ax.set_xticklabels([f"{L}\n(in {L + 1})" for L in LAYERS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{label} · patched: {TITLES[patch_set]}", fontsize=10, loc="left")
        if c == 0:
            ax.set_ylabel("fraction of prompts (greedy decode)")
        if r == 1:
            ax.set_xlabel("hidden_states[L] (input to block L+1, 1-based)", fontsize=9)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.005, 0.965))
fig.suptitle("Hidden-state swaps around block 15 (dotted line = block 15: left is its input, right is its output)",
             x=0.005, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("block15_hidden_categories.png", dpi=150)
print("saved block15_hidden_categories.png")
