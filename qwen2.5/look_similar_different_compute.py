from prompt_set import prompt_set
from functions import load_model, get_hidden_states, pool_last_token, pool_mean, cosine_sims

model, tokenizer, device = load_model()

prompt_a, prompt_b = prompt_set["look_similar_diff_computation"][0]

hs_a = get_hidden_states(prompt_a, model, tokenizer, device)
hs_b = get_hidden_states(prompt_b, model, tokenizer, device)

last_token_sims = cosine_sims(pool_last_token(hs_a), pool_last_token(hs_b))
mean_pool_sims = cosine_sims(pool_mean(hs_a), pool_mean(hs_b))

print(f"A: '{prompt_a}'")
print(f"B: '{prompt_b}'\n")
print(f"{'layer':>6} | {'last-token sim':>16} | {'mean-pool sim':>15}")
print("-" * 45)
for layer_idx in range(len(last_token_sims)):
    print(f"{layer_idx:>6} | {last_token_sims[layer_idx]:>16.4f} | {mean_pool_sims[layer_idx]:>15.4f}")