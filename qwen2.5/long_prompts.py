import json
import random

SEED = 0
NUM_PAIRS = 50
NUM_TRAIN_PAIRS = 35
LOW, HIGH = 2, 30

OPS = ["add", "sub", "mul", "div"]

# "/" instead of "÷" because Qwen splits "÷" into 2 tokens (length leak)
SYMBOLS = {"add": "+", "sub": "-", "mul": "×", "div": "/"}
PHRASES = {"add": "the sum of {a} and {b}",
           "sub": "the difference between {a} and {b}",
           "mul": "the product of {a} and {b}",
           "div": "the quotient of {a} and {b}"}

# same three sentence frames for both formats, only {expr} changes
TEMPLATES = {
    "please_calculate": "Please calculate {expr} and tell me the final answer.",
    "need_compute": "I need you to compute {expr} and give me the final answer.",
    "work_out": "Could you work out what {expr} is and tell me the final answer.",
}

def make_expr(fmt, op, a, b):
    if fmt == "symbolic":
        return f"{a} {SYMBOLS[op]} {b}"
    return PHRASES[op].format(a=a, b=b)

rng = random.Random(SEED)

# pick unique operand pairs; (5, 9) and (9, 5) count as the same pair
pairs = []
seen = set()
while len(pairs) < NUM_PAIRS:
    a, b = rng.sample(range(LOW, HIGH + 1), 2)
    key = frozenset((a, b))
    if key in seen:
        continue
    seen.add(key)
    pairs.append((a, b))

rng.shuffle(pairs)
train_pairs = pairs[:NUM_TRAIN_PAIRS]
test_pairs = pairs[NUM_TRAIN_PAIRS:]

train_keys = {frozenset(p) for p in train_pairs}
test_keys = {frozenset(p) for p in test_pairs}
assert train_keys.isdisjoint(test_keys), "operand pair leaked across splits"

rows = []
for split, split_pairs in [("train", train_pairs), ("test", test_pairs)]:
    for (a, b) in split_pairs:
        for fmt in ["symbolic", "natural"]:
            for op in OPS:
                for template_name, template in TEMPLATES.items():
                    rows.append({
                        "prompt": template.format(expr=make_expr(fmt, op, a, b)),
                        "a": a, "b": b, "op": op,
                        "format": fmt, "wording": template_name, "split": split,
                    })

with open("probe_dataset.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

# summary
print(f"total prompts: {len(rows)}")
for fmt in ["symbolic", "natural"]:
    for split in ["train", "test"]:
        sub = [r for r in rows if r["format"] == fmt and r["split"] == split]
        counts = {op: sum(r["op"] == op for r in sub) for op in OPS}
        print(f"{fmt:9s} {split:5s}: {len(sub)} prompts  {counts}")

print("\nexample prompts (one pair, both formats):")
a, b = train_pairs[0]
for r in rows:
    if (r["a"], r["b"]) == (a, b) and r["wording"] == "please_calculate":
        print(f"  [{r['format']}] {r['prompt']}")

endings = {" ".join(r["prompt"].split()[-2:]) for r in rows}
print(f"\ndistinct final two words: {endings}")