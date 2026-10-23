import json
import random

from utilities import load_model_fp32, answer_metrics, find_op_pos, TEMPLATE, SYMBOLS

SEED = 0
LOW, HIGH = 2, 20
NUM_CANONICAL_PAIRS = 50     # x 2 orders = 100 prompts per op for the averages
NUM_TEST_PAIRS = 50          # x 2 orders = 100 held-out prompts per op
ANSWER = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}

model, tokenizer, device = load_model_fp32()

# every unordered pair {a, b} with a != b; we use both orders (a, b) and (b, a)
all_pairs = [(a, b) for a in range(LOW, HIGH + 1) for b in range(a + 1, HIGH + 1)]

# keep a pair only if the clean model gets BOTH orders right for BOTH ops
good_pairs = []
for a, b in all_pairs:
    ok = all(
        answer_metrics(model, tokenizer, device,
                       TEMPLATE.format(a=x, op=SYMBOLS[op], b=y), ANSWER[op](x, y))[0]
        for op in ["mul", "add"] for (x, y) in [(a, b), (b, a)]
    )
    if ok:
        good_pairs.append((a, b))

print(f"pairs checked: {len(all_pairs)}, clean-correct for both ops/orders: {len(good_pairs)}")

rng = random.Random(SEED)
rng.shuffle(good_pairs)
canonical_pairs = good_pairs[:NUM_CANONICAL_PAIRS]
test_pairs = good_pairs[NUM_CANONICAL_PAIRS:NUM_CANONICAL_PAIRS + NUM_TEST_PAIRS]
assert not {frozenset(p) for p in canonical_pairs} & {frozenset(p) for p in test_pairs}

rows = []
for role, pairs in [("canonical", canonical_pairs), ("test", test_pairs)]:
    for a, b in pairs:
        for x, y in [(a, b), (b, a)]:
            for op in ["mul", "add"]:
                prompt = TEMPLATE.format(a=x, op=SYMBOLS[op], b=y)
                rows.append({
                    "prompt": prompt, "a": x, "b": y, "op": op,
                    "answer": ANSWER[op](x, y),
                    "op_pos": find_op_pos(tokenizer, prompt, op),
                    "role": role,
                })

with open("patch_dataset.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

for role in ["canonical", "test"]:
    for op in ["mul", "add"]:
        sub = [r for r in rows if r["role"] == role and r["op"] == op]
        positions = sorted({r["op_pos"] for r in sub})
        print(f"{role:9s} {op}: {len(sub)} prompts, operator token positions {positions}")

print("\nexamples:")
for r in rows[:4]:
    print(f"  [{r['role']}] {r['prompt']!r} -> {r['answer']}  (operator token #{r['op_pos']})")

    