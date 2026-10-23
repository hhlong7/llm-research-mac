import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEMPLATE = "Calculate {a} {op} {b} = "     # trailing space: model answers right away
SYMBOLS = {"mul": "×", "add": "+"}
NUM_PATCH_LAYERS = 24   # hidden_states[0..23] = what goes INTO layer 0..23


def load_model_fp32():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    model.eval()
    return model, tokenizer, device


def find_op_pos(tokenizer, prompt, op):
    ids = tokenizer(prompt)["input_ids"]
    matches = [i for i, t in enumerate(ids) if tokenizer.decode([t]).strip() == SYMBOLS[op]]
    assert len(matches) == 1, f"operator token not found exactly once: {prompt}"
    return matches[0]


def add_patch_hook(model, layer, pos, vec):
    """replace hidden_states[layer] at token `pos` with `vec`"""
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] > pos:
            hs[:, pos, :] = vec
        return output
    target = model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
    return target.register_forward_hook(hook)


def answer_metrics(model, tokenizer, device, prompt, answer, patch=None):
    """
    feeds prompt + correct answer in one pass (teacher forcing).
    returns (correct, logprob):
      correct = greedy decoding would produce exactly the answer
      logprob = log P(answer digits | prompt), summed over the digit tokens
    patch = (layer, pos, vec) or None
    """
    prompt_ids = tokenizer(prompt)["input_ids"]
    answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt_ids + answer_ids], device=device)

    handle = add_patch_hook(model, *patch) if patch is not None else None
    try:
        with torch.no_grad():
            logits = model(ids).logits[0].float()
    finally:
        if handle is not None:
            handle.remove()

    logprobs = torch.log_softmax(logits, dim=-1)
    n = len(prompt_ids)
    total_lp, correct = 0.0, True
    for k, tid in enumerate(answer_ids):
        pos = n - 1 + k                      # logits at pos predict token pos+1
        total_lp += logprobs[pos, tid].item()
        correct &= logits[pos].argmax().item() == tid
    # the token after the answer must not be another digit ("42", not "420")
    correct &= not tokenizer.decode([logits[-1].argmax().item()]).strip().isdigit()
    return bool(correct), total_lp