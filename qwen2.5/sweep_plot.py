import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import matplotlib.pyplot as plt

#write the formula for theoretical bytes as a func:
    #seq len and batch size as inputs, 
    #2 * num layers * num kv heads * head dim * seq len * batch size * bytes per element
#then measured byte func:
    #dummy input = seq len tokens, run one fwd pass with use cache = true
    #sum up .numel() across all layers of kv tensors
#range of lengths to sweep
#loop over the lengths:
    #compute both theoretical and actual bytes at each
#plot:
    #cache size on y axis, context length on the x axis
    #theoretical and actual should sit on top of each other

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

cfg = model.config
num_layers = cfg.num_hidden_layers
num_kv_heads = cfg.num_key_value_heads
head_dim = cfg.hidden_size // cfg.num_attention_heads
bytes_per_element = 2  # bfloat16

def theoretical_bytes(seq_len, batch_size=1):
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_element

def measured_bytes_for_length(seq_len):
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(1, seq_len)).to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    pkv = out.past_key_values
    total_elements = sum(l.keys.numel() + l.values.numel() for l in pkv.layers)
    return total_elements * pkv.layers[0].keys.element_size()

lengths = [8, 16, 32, 64, 128, 256, 512]
results = []
print(f"{'seq_len':>8} | {'measured bytes':>15} | {'theoretical bytes':>18} | {'match':>6}")
print("-" * 60)
for L in lengths:
    measured = measured_bytes_for_length(L)
    theoretical = theoretical_bytes(L)
    match = (measured == theoretical)
    results.append({"seq_len": L, "measured_bytes": measured, "theoretical_bytes": theoretical, "match": match})
    print(f"{L:>8} | {measured:>15,} | {theoretical:>18,} | {str(match):>6}")

print(f"\nAll measured values match theoretical formula exactly: {all(r['match'] for r in results)}")

with open("length_sweep.json", "w") as f:
    json.dump({
        "model_id": model_id, "num_layers": num_layers, "num_kv_heads": num_kv_heads,
        "head_dim": head_dim, "bytes_per_element": bytes_per_element, "results": results,
    }, f, indent=2)

#plot
lengths_x = [r["seq_len"] for r in results]
measured_mb = [r["measured_bytes"] / (1024**2) for r in results]
theoretical_mb = [r["theoretical_bytes"] / (1024**2) for r in results]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(lengths_x, theoretical_mb, 'o-', label="Theoretical (from config formula)", color="#2f6fed", linewidth=2, markersize=7)
ax.plot(lengths_x, measured_mb, 'x', label="Measured (observed in model)", color="#e3542a", markersize=11, markeredgewidth=2.5)
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("KV cache size (MB)")
ax.set_title(f"Qwen2.5-0.5B-Instruct KV Cache Size vs. Context Length\n"
             f"layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}, dtype=bfloat16")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("kv_cache_size_vs_length.png", dpi=160)
print("Saved kv_cache_size_vs_length.png")