import csv

import matplotlib.pyplot as plt
import numpy as np

# Tyler and Julian's results (cloned repo), used for the comparison panels
THEIRS = "/Users/hhlong/Documents/LLM_Research/block_15_testing/frozen_reuse_results"

BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#b0aca4", "#2b2a28"
DATASETS = [("ours", "Our dataset (unseen numbers)"),
            ("pair_holdout", "T&J pair_holdout"),
            ("number_holdout", "T&J number_holdout")]
OPS = [("mul", "× prompts"), ("add", "+ prompts")]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)


def their_rows(stage, scope):
    with open(f"{THEIRS}/summary{'_single' if stage == 'single' else ''}.csv") as f:
        return [r for r in csv.DictReader(f) if r["stage"] == stage and r["scope"] == scope and r["split"] == "test"]


# ---------- figure 1: what the model wrote after each single-location swap (step 3) ----------
CONDS = [("baseline", "No swap"), ("donor_same", "Same-op donor"), ("donor_other", "Other-op donor"),
         ("mean_same", "Same-op average"), ("mean_other", "Other-op average"), ("mean_global", "Global average"),
         ("delta_plus", "State + (× − +)"), ("delta_minus", "State − (× − +)"),
         ("random_state", "Random state"), ("random_direction", "Random direction"), ("zero", "Zero")]
CATS = [("original", "original answer", BLUE), ("other_op", "other-op answer", ORANGE),
        ("unrelated", "unrelated answer", GRAY)]

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
for r, (name, label) in enumerate(DATASETS[:2]):
    d = np.load(f"interventions_{name}.npz")
    conds = list(d["conditions"])
    test = d["split"] == "test"
    for c, (op, op_label) in enumerate(OPS):
        ax = axes[r, c]
        keep = test & (d["op"] == op)
        y = np.arange(len(CONDS))[::-1]
        left = np.zeros(len(CONDS))
        for key, cat_label, color in CATS:
            frac = np.array([(d["cat"][keep, conds.index(k)] == key).mean() for k, _ in CONDS])
            ax.barh(y, frac, left=left, color=color, height=0.75, label=cat_label,
                    edgecolor="white", linewidth=1.5)
            left += frac
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in CONDS], fontsize=9)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{label} · {op_label} (n={keep.sum()})", fontsize=10, loc="left")
        if r == 1:
            ax.set_xlabel("fraction of test prompts (greedy decoding)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 0.955))
fig.suptitle("Replacing the operand-B hidden state at the input to block 15: what the model writes",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("reuse_generated_answers.png", dpi=150)
print("saved reuse_generated_answers.png")


# ---------- figure 2: accuracy vs memory (steps 4 and 5) ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
for ax, (name, label) in zip(axes, DATASETS):
    for stage, file, dash in [("single", "compression", "-"), ("cumulative", "cumulative", "--")]:
        d = np.load(f"{file}_{name}.npz")
        plan = [p.split("|") for p in d["plan"]]
        test = d["split"] == "test"
        for ref, color, ref_label in [("same", BLUE, "same-op average"), ("global", ORANGE, "global average")]:
            idx = [j for j, (k, f) in enumerate(plan) if k == ref]
            mem = d["memory"][idx]
            acc = np.array([(d["cat"][test, j] == "original").mean() for j in idx])
            stage_label = "B at block 15" if stage == "single" else "operator + B, blocks 8-16"
            ax.plot(mem * 100, acc, dash, marker="o", color=color, linewidth=2, markersize=6,
                    label=f"{ref_label} + top residual · {stage_label}")
            chosen = float(d[f"chosen_{ref}"])
            j = [jj for jj in idx if float(plan[jj][1]) == chosen][0]
            ax.plot(d["memory"][j] * 100, (d["cat"][test, j] == "original").mean(), marker="*",
                    color=color, markersize=17, markeredgecolor="white", zorder=5)
    if name in ("pair_holdout", "number_holdout"):
        for stage, scope, marker in [("single", "block_15_b", "s"), ("cumulative", "blocks_8_16_operator_b", "D")]:
            rows = [r for r in their_rows(stage, scope) if r["suite"] == name and r["representation"] == "hidden"
                    and r["condition"] == "topk"]
            pts = {}
            for r in rows:
                pts.setdefault(float(r["fraction"]), []).append((float(r["memory_ratio"]), float(r["original_rate"])))
            xs = [np.mean([m for m, _ in v]) * 100 for v in pts.values()]
            ys = [np.mean([a for _, a in v]) for v in pts.values()]
            ax.scatter(xs, ys, marker=marker, s=60, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=6,
                       label=f"T&J same-op · {'B at block 15' if stage == 'single' else 'operator + B, blocks 8-16'}")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("memory per replaced state (% of full 896 × float32)")
    ax.set_title(label, fontsize=11)
    style(ax)
axes[0].set_ylabel("test prompts keeping the original answer")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.94))
fig.suptitle("Accuracy vs memory: frozen average + largest residual entries (hidden state; ★ = fraction picked on validation)",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig("reuse_accuracy_memory.png", dpi=150)
print("saved reuse_accuracy_memory.png")


# ---------- figure 3: hidden state (ours) vs value vector (T&J), same prompts, pair_holdout ----------
COMPARE = [("donor_same", "Same-op\ndonor"), ("donor_other", "Other-op\ndonor"), ("mean_same", "Same-op\naverage"),
           ("mean_other", "Other-op\naverage"), ("zero", "Zero"), ("topk|0.05", "Average +\ntop 5%"),
           ("topk|0.25", "Average +\ntop 25%")]
d_int = np.load("interventions_pair_holdout.npz")
d_cmp = np.load("compression_pair_holdout.npz")
conds = list(d_int["conditions"])
plan = list(d_cmp["plan"])
value = {(r["condition"] + (f"|{float(r['fraction']):g}" if r["condition"] == "topk" else ""), r["op"]): float(r["original_rate"])
         for r in their_rows("single", "block_15_b") if r["suite"] == "pair_holdout" and r["representation"] == "value"}

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, (op, op_label) in zip(axes, OPS):
    keep_i = (d_int["split"] == "test") & (d_int["op"] == op)
    keep_c = (d_cmp["split"] == "test") & (d_cmp["op"] == op)
    hidden = []
    for key, _ in COMPARE:
        if key.startswith("topk"):
            j = plan.index(f"same|{float(key.split('|')[1])}")
            hidden.append((d_cmp["cat"][keep_c, j] == "original").mean())
        else:
            hidden.append((d_int["cat"][keep_i, conds.index(key)] == "original").mean())
    v = [value[(key, op)] for key, _ in COMPARE]
    x = np.arange(len(COMPARE))
    ax.bar(x - 0.2, v, width=0.38, color=BLUE, label="value vector V inside block 15 (T&J)")
    ax.bar(x + 0.2, hidden, width=0.38, color=ORANGE, label="hidden state into block 15 (ours)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in COMPARE], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"pair_holdout test · {op_label}", fontsize=10, loc="left")
    style(ax)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("test prompts keeping the original answer")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 0.92))
fig.suptitle("Operand B at block 15: value vector vs hidden state under the same replacements",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig("reuse_hidden_vs_value.png", dpi=150)
print("saved reuse_hidden_vs_value.png")
