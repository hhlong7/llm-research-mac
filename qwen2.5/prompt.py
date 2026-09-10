import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

#token the prompt
prompt = "This is the research for LLM architecture"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"Prompt: '{prompt}'  ({inputs['input_ids'].shape[1]} tokens)")

#one forward pass (prefill), with caching on
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

pkv = outputs.past_key_values

#inspect what past_key_values actually is before assuming its shape
print(f"\ntype(past_key_values): {type(pkv)}")
print(f"attributes: {[a for a in dir(pkv) if not a.startswith('_')]}")

#pull one layer's k v tensors and print their shapes
#modern transformers versions expose a `.layers` list, each with `.keys` / `.values`
#adjust based on the actual attributes if doesnt match
layer0 = pkv.layers[0]
print(f"\nLayer 0 key shape:   {layer0.keys.shape}")
print(f"Layer 0 value shape: {layer0.values.shape}")
print(f"Layer 0 key dtype:   {layer0.keys.dtype}")

#confirm every layer matches
print(f"\nTotal layers in cache: {len(pkv.layers)}")
all_same = all(l.keys.shape == layer0.keys.shape for l in pkv.layers)
print(f"All layers have the same K shape: {all_same}")