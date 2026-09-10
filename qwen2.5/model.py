import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}") #using which mps (gpu backedn) or cpu

id_model = "Qwen/Qwen2.5-0.5B-Instruct"

#goes to the huggingface to download tokenizer fiels n model weights
#=> reconstruct into memory
tokenizer = AutoTokenizer.from_pretrained(id_model)
model = AutoModelForCausalLM.from_pretrained(id_model)
model = model.to(device) #usable now => move model's tensor to the chosen device
model.eval()

cfg = model.config
print(f"num_hidden_layers:    {cfg.num_hidden_layers}")
print(f"num_attention_heads:  {cfg.num_attention_heads}")
print(f"num_key_value_heads:  {cfg.num_key_value_heads}")
print(f"hidden_size:          {cfg.hidden_size}")
print(f"torch_dtype:          {cfg.torch_dtype}")