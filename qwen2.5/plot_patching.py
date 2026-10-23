import numpy as np
import matplotlib.pyplot as plt

res = np.load("patching_results.npz")
NUM_LAYERS = 24
layers = np.arange(NUM_LAYERS)
OTHER = {"mul": "add", "add": "mul"}
NAME = {"mul": "×", "add": "+"}

def mean_and_se(x):
    """mean over test prompts, and standard error (for the shaded band)"""
    x = x.astype(float)
    return x.mean(axis=0), x.std(axis=0) / np.sqrt(len(x))

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for row, op in enumerate(["mul", "add"]):
    o = OTHER[op]
    conditions = [
        ("avg_same",    f"avg {NAME[op]} (canonical)",       "tab:blue"),
        ("single_same", f"{NAME[op]} from 1 other prompt",   "tab:green"),
        ("avg_other",   f"avg {NAME[o]} (other op)",         "tab:red"),
    ]
    ax_acc, ax_lp = axes[row]

    for key, label, color in conditions:
        m, se = mean_and_se(res[f"{op}_acc_{key}"])
        ax_acc.plot(layers, m, marker="o", ms=3, color=color, label=label)
        ax_acc.fill_between(layers, m - se, m + se, color=color, alpha=0.2)

        # log prob change vs the clean run of the same prompt (0 = no effect)
        delta = res[f"{op}_lp_{key}"] - res[f"{op}_clean_lp"][:, None]
        m, se = mean_and_se(delta)
        ax_lp.plot(layers, m, marker="o", ms=3, color=color, label=label)
        ax_lp.fill_between(layers, m - se, m + se, color=color, alpha=0.2)

    m, _ = mean_and_se(res[f"{op}_switched"])
    ax_acc.plot(layers, m, ls=":", color="tab:red", label=f"answers {NAME[o]} instead")

    for ax in (ax_acc, ax_lp):
        ax.axvspan(8, 16, color="gold", alpha=0.12)
        ax.grid(alpha=0.3)
    ax_acc.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax_lp.axhline(0.0, color="gray", ls="--", lw=0.8, label="clean (no swap)")

    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_ylabel("answer accuracy")
    ax_lp.set_ylabel("Δ log P(correct answer) vs clean")
    ax_acc.set_title(f"{NAME[op]} test prompts: accuracy after swapping the {NAME[op]} token")
    ax_lp.set_title(f"{NAME[op]} test prompts: change in correct-answer log prob")
    ax_acc.legend(fontsize=8, loc="center right")

axes[0, 1].legend(fontsize=8, loc="lower right")
for ax in axes[1]:
    ax.set_xlabel("patched layer (vector going INTO this layer)   [shaded: layers 8–16]")
plt.tight_layout()
plt.savefig("patching_by_layer.png", dpi=150)
print("saved patching_by_layer.png")