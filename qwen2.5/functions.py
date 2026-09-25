import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def load_model():
    """Loads the model/tokenizer once; returns (model, tokenizer, device)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, tokenizer, device

def get_hidden_states(prompt, model, tokenizer, device):
    """Returns the raw hidden_states tuple (25 tensors, each (1, seq_len, 896))."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states

def pool_last_token(hidden_states):
    """One (896,) vector per layer, using the last token's representation."""
    return [layer[0, -1, :] for layer in hidden_states]

def pool_mean(hidden_states):
    """One (896,) vector per layer, averaged across all tokens."""
    return [layer[0].mean(dim=0) for layer in hidden_states]

def cosine_sims(vecs_a, vecs_b):
    """Cosine similarity per layer between two lists of (896,) vectors."""
    return [
        F.cosine_similarity(va.unsqueeze(0).float(), vb.unsqueeze(0).float()).item()
        for va, vb in zip(vecs_a, vecs_b)
    ]