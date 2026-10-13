import json
from prompt_set2 import prompt_set2
from functions import load_model, get_hidden_states, pool_last_token, pool_mean, cosine_sims

model, tokenizer, device = load_model()

MID_LAYERS = range(3, 22)

def avg_mid_layers(sims):
    mid_vals = [sims[i] for i in MID_LAYERS]
    return sum(mid_vals) / len(mid_vals)

results = {}

print(f"{'axis':>20} | {'pair':>4} | {'last-tok avg':>12} | {'mean-pool avg':>13}")
print("-" * 60)

for axis, pairs in prompt_set2.items():
    results[axis] = []
    for i, (prompt_a, prompt_b) in enumerate(pairs):
        hs_a = get_hidden_states(prompt_a, model, tokenizer, device)
        hs_b = get_hidden_states(prompt_b, model, tokenizer, device)

        last_token_sims = cosine_sims(pool_last_token(hs_a), pool_last_token(hs_b))
        mean_pool_sims = cosine_sims(pool_mean(hs_a), pool_mean(hs_b))

        results[axis].append({
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "last_token_sims": last_token_sims,
            "mean_pool_sims": mean_pool_sims,
        })

        lt_avg = avg_mid_layers(last_token_sims)
        mp_avg = avg_mid_layers(mean_pool_sims)
        print(f"{axis:>20} | {i:>4} | {lt_avg:>12.4f} | {mp_avg:>13.4f}")

with open("sweep_results2.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved full per-layer data to sweep_results2.json")