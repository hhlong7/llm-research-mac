import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

def cache_seq_len_and_bytes(pkv):
    layer0 = pkv.layers[0]
    seq_len = layer0.keys.shape[2]
    total_elements = sum(l.keys.numel() + l.values.numel() for l in pkv.layers)
    total_bytes = total_elements * layer0.keys.element_size()
    return seq_len, total_bytes

prompt = "I have got a date at a quarter to eight; I'll see you at the gate, so don't be late."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

print(f"Prompt: '{prompt}'  ({input_ids.shape[1]} tokens)\n")

with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)

pkv = outputs.past_key_values
seq_len, total_bytes = cache_seq_len_and_bytes(pkv)
print(f"[PREFILL]  seq_len={seq_len:3d}   cache size={total_bytes:,} bytes")

next_token_logits = outputs.logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

history = [(seq_len, total_bytes, "prefill")]
n_new_tokens = 10

for step in range(n_new_tokens):
    with torch.no_grad():
        outputs = model(input_ids=next_token, past_key_values=pkv, use_cache=True)
    pkv = outputs.past_key_values
    seq_len, total_bytes = cache_seq_len_and_bytes(pkv)
    token_str = tokenizer.decode(next_token[0])
    print(f"[DECODE {step+1:2d}] seq_len={seq_len:3d}   cache size={total_bytes:,} bytes   token: {token_str!r}")
    history.append((seq_len, total_bytes, f"decode_{step+1}"))

    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

bytes_per_step = history[1][1] - history[0][1]
print(f"\nBytes added per decode step: {bytes_per_step:,}")
for i in range(2, len(history)):
    delta = history[i][1] - history[i-1][1]
    assert delta == bytes_per_step, f"Growth not constant at step {i}: {delta} vs {bytes_per_step}"
print("Confirmed: every decode step adds the same number of bytes.")