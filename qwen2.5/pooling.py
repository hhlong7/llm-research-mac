import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

"""
get raw data func, take the text prompt and return the full 25 tensor tuple
by running through the model once
"""
def get_hidden_states(prompt):
    """Returns the raw hidden_states tuple (25 tensors, each (1, seq_len, 896))."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states

"""
two pooling strat, but i lean more towards last token since qwen is autoregressive
but pool mean is for comparison

in: 25 tensor tuple
out: a list of 25 vectors
"""
def pool_last_token(hidden_states):
    """One (896,) vector per layer, using the last token's representation."""
    return [layer[0, -1, :] for layer in hidden_states]

def pool_mean(hidden_states):
    """One (896,) vector per layer, averaged across all tokens."""
    return [layer[0].mean(dim=0) for layer in hidden_states]

"""
comparison function, takes two pooled vector lists and walk thru layer by layers
and compute one similarity score per depth
"""
def cosine_sims(vecs_a, vecs_b):
    """Cosine similarity per layer between two lists of (896,) vectors."""
    return [
        F.cosine_similarity(va.unsqueeze(0).float(), vb.unsqueeze(0).float()).item()
        for va, vb in zip(vecs_a, vecs_b)
    ]

prompt_a = "Calculate 17 x 23"
#prompt_b = "Compute 17 x 23"
prompt_b = "Alice has a MacBook and a Windows."

hs_a = get_hidden_states(prompt_a)
hs_b = get_hidden_states(prompt_b)

last_token_sims = cosine_sims(pool_last_token(hs_a), pool_last_token(hs_b))
mean_pool_sims = cosine_sims(pool_mean(hs_a), pool_mean(hs_b))

print(f"A: '{prompt_a}'")
print(f"B: '{prompt_b}'\n")
print(f"{'layer':>6} | {'last-token sim':>16} | {'mean-pool sim':>15}")
print("-" * 45)
for layer_idx in range(len(last_token_sims)):
    print(f"{layer_idx:>6} | {last_token_sims[layer_idx]:>16.4f} | {mean_pool_sims[layer_idx]:>15.4f}")