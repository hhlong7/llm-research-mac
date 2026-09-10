import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- device selection ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- model id ---
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# --- load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)
model.eval()

# --- inspect config ---
cfg = model.config
print(f"num_hidden_layers:    {cfg.num_hidden_layers}")
print(f"num_attention_heads:  {cfg.num_attention_heads}")
print(f"num_key_value_heads:  {cfg.num_key_value_heads}")
print(f"hidden_size:          {cfg.hidden_size}")
print(f"torch_dtype:          {cfg.torch_dtype}")
