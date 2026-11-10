import numpy as np
import matplotlib.pyplot as plt

# same layout as Tyler and Julian's KV figure: x = attention block 14-16 (1-based)
# a hidden-state patch "at block N" replaces hidden_states[N-1], the input to block N,
# which is exactly what block N's K and V are computed from
d = np.load("block15_hidden_results.npz", allow_pickle=True)
plan = list(d["plan"])
BLOCKS = [14, 15, 16]
PATCH_SETS = [("operator", "Operator token"), ("b_last", "Last digit of b"),
              ("final", "Final prompt token"), ("all_three", "All three positions")]
DIRECTIONS = [("mul", "Multiplication → addition"), ("add", "Addition → multiplication")]
COLOR = "#2b2a28"
BASE_COLOR = "#9a968e"
CLEAN = plan.index("clean|None")   # row 0: the prompt run with no swap


def draw(kind, filename, title):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharey=True)
    for r, (op, label) in enumerate(DIRECTIONS):
        ok = d[f"{op}_baseline_ok"]
        for c, (patch_set, name) in enumerate(PATCH_SETS):
            ax = axes[r, c]
            idx = [plan.index(f"{patch_set}|{b - 1}") for b in BLOCKS]
            # no-swap baseline: clean P(original), or the clean fraction of original answers
            if kind == "prob":
                base = d[f"{op}_prob_orig"][ok][:, CLEAN].mean()
            else:
                base = (d[f"{op}_cat"][ok][:, CLEAN] == "original").mean()
            ax.axhline(base, color=BASE_COLOR, linewidth=1.5, zorder=0, label="no-swap baseline (original)")
            if c == 0:
                ax.text(BLOCKS[0] - 0.05, base + 0.015, f"baseline {base:.2f}", color=BASE_COLOR,
                        fontsize=8, va="bottom")
            if kind == "prob":
                lines = [(d[f"{op}_prob_orig"][ok][:, idx].mean(0), "-", "P(original)"),
                         (d[f"{op}_prob_alt"][ok][:, idx].mean(0), "--", "P(other op)")]
            else:
                cat = d[f"{op}_cat"][ok][:, idx]
                lines = [((cat == "original").mean(0), "-", "original answer"),
                         ((cat == "other_op").mean(0), "--", "other-op answer"),
                         ((cat == "unrelated").mean(0), ":", "unrelated answer")]
            for y, style, lab in lines:
                ax.plot(BLOCKS, y, style, marker="o", color=COLOR, linewidth=2, label=lab)
            ax.set_xticks(BLOCKS)
            ax.set_ylim(-0.02, 1.02 if kind == "cat" else 0.8)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(name, fontsize=10)
            ax.tick_params(labelleft=True)
            if c == 0:
                ax.set_ylabel(f"{label}\n" + ("teacher-forced probability" if kind == "prob"
                                              else "fraction of prompts (greedy)"), fontsize=9)
            if r == 1:
                ax.set_xlabel("Attention block (1-based)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(filename, dpi=150)
    print("saved", filename)


draw("prob", "block15_hidden_prob.png",
     "Hidden-state swap at the input to each block: probability of the original and other-operation answers")
draw("cat", "block15_hidden_categories_lines.png",
     "Hidden-state swap at the input to each block: decoded answers")
