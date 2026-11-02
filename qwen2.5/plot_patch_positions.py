import numpy as np
import matplotlib.pyplot as plt

res = np.load("patch_positions_results.npz")
plan = [str(p).split("|") for p in res["plan"]]
layers = np.arange(24)
OPS = {"mul": "×", "add": "+"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "lines.linewidth": 2,
})


def curve(op, metric, source, patch_set):
    """average over test prompts, one value per layer"""
    idx = [i for i, (s, p, _) in enumerate(plan) if s == source and p == patch_set]
    values = res[f"{op}_{metric}"][:, idx]
    if metric.startswith("first"):      # only prompts whose two answers start differently
        values = values[res[f"{op}_first_differs"]]
    return values.mean(axis=0)


def finish(ax, title, ylabel=None, xlabel=True):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks(range(0, 24, 4))
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("patched layer")


# ---- plot 1: Andrew's three patches, with accuracy and both probabilities ----
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
for r, (op, sym) in enumerate(OPS.items()):
    other = OPS["add" if op == "mul" else "mul"]
    for c, patch_set in enumerate(["operator", "final", "both"]):
        ax = axes[r, c]
        ax.plot(layers, curve(op, "acc_orig", "paired_other", patch_set), color=BLUE,
                label="accuracy, original answer")
        ax.plot(layers, curve(op, "prob_orig", "paired_other", patch_set), color=BLUE, ls="--",
                label="P(original answer)")
        ax.plot(layers, curve(op, "acc_alt", "paired_other", patch_set), color=ORANGE,
                label="accuracy, other-op answer")
        ax.plot(layers, curve(op, "prob_alt", "paired_other", patch_set), color=ORANGE, ls="--",
                label="P(other-op answer)")
        finish(ax, f"{sym} prompts, patched: {patch_set}",
               "accuracy or probability" if c == 0 else None, r == 1)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.965), ncol=4, frameon=False)
fig.suptitle("Swapping in the same numbers with the other operation", y=0.995, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig("patch_operator_final_both.png", dpi=150)

# ---- plot 2: where the operation is, using the first digit of the answer ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, (op, sym) in zip(axes, OPS.items()):
    for patch_set, color, label in [("operator", BLUE, "operator"),
                                    ("b_last", ORANGE, "last digit of b"),
                                    ("final", AQUA, "final")]:
        ax.plot(layers, curve(op, "first_alt", "paired_other", patch_set), color=color, label=label)
    ax.plot(layers, curve(op, "first_alt", "paired_other", "op+b+final"), color=MUTED, ls="--",
            label="all three together")
    finish(ax, f"{sym} prompts", "first digit flips to the other op" if op == "mul" else None)
axes[0].legend(frameon=False, loc="center right", fontsize=9)
fig.suptitle("Which token controls the operation at each layer", y=0.995, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig("patch_operation_handoff.png", dpi=150)

# ---- plot 3: can a generic state replace the final token? ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, (op, sym) in zip(axes, OPS.items()):
    ax.axvspan(14.5, 16.5, color="#f0efec", zorder=0, label="layers 15 to 16")
    ax.plot(layers, curve(op, "acc_orig", "avg_same", "final"), color=BLUE,
            label="same-op average: original answer correct")
    ax.plot(layers, curve(op, "acc_orig", "avg_other", "final"), color=ORANGE,
            label="other-op average: original answer correct")
    ax.plot(layers, curve(op, "first_alt", "avg_other", "final"), color=ORANGE, ls="--",
            label="other-op average: first digit flips")
    finish(ax, f"{sym} prompts", "fraction of test prompts" if op == "mul" else None)
axes[0].legend(frameon=False, loc="center left", fontsize=9)
fig.suptitle("Replacing the final token with a generic average", y=0.995, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig("patch_final_token_reuse.png", dpi=150)

print("saved patch_operator_final_both.png, patch_operation_handoff.png, patch_final_token_reuse.png")