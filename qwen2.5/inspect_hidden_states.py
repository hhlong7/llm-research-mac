import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()


prompt = "Calculate 17 x 23"
#prompt = "Alice has a MacBook and a Windows."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"Prompt: '{prompt}' ({inputs['input_ids'].shape[1]} tokens)")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states

print(f"\ntype(hidden_states): {type(hidden_states)}")
print(f"Number of layers in hidden_states tuple: {len(hidden_states)}")
print(f"Shape of hidden_states[0]: {hidden_states[0].shape}")
print(f"Shape of hidden_states[-1]: {hidden_states[-1].shape}")
print(f"dtype: {hidden_states[0].dtype}")