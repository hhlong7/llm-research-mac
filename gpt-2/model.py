import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)
model.eval()

cfg = model.config
head_dim = cfg.n_embd // cfg.n_head
print(f"n_layer:    {cfg.n_layer}")
print(f"n_head:     {cfg.n_head}")
print(f"n_embd:     {cfg.n_embd}")
print(f"head_dim:   {head_dim}   (n_embd / n_head)")
print(f"n_positions (max context): {cfg.n_positions}")