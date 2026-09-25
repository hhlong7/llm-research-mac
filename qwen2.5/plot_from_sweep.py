import json
import matplotlib.pyplot as plt

with open("sweep_results.json") as f:
    results = json.load(f)

NUM_LAYERS = 25
categories = list(results.keys())

def category_avg_per_layer(category, sim_key):
    """Average similarity at each layer, across all pairs in this category."""
    pairs = results[category]
    layer_avgs = []
    for layer_idx in range(NUM_LAYERS):
        vals = [pair[sim_key][layer_idx] for pair in pairs]
        layer_avgs.append(sum(vals) / len(vals))
    return layer_avgs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for category in categories:
    lt_avg = category_avg_per_layer(category, "last_token_sims")
    mp_avg = category_avg_per_layer(category, "mean_pool_sims")
    ax1.plot(range(NUM_LAYERS), lt_avg, marker="o", markersize=3, label=category)
    ax2.plot(range(NUM_LAYERS), mp_avg, marker="o", markersize=3, label=category)

ax1.set_title("Last-token pooling")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Cosine similarity (avg across pairs)")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.set_title("Mean pooling")
ax2.set_xlabel("Layer")
ax2.set_ylabel("Cosine similarity (avg across pairs)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle("Layer-wise hidden-state similarity by prompt category (Qwen2.5-0.5B-Instruct)")
fig.tight_layout()
fig.savefig("layerwise_similarity.png", dpi=160)
print("saved layerwise_similarity.png")