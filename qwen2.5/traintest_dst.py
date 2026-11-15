import json
import random

from utilities import load_model_fp32, answer_metrics, find_op_pos, SYMBOLS

SEED = 0
LOW, HIGH = 2, 20
ANSWER = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}

# train and test never share a wording
TEMPLATES = {
    "train": ["Calculate {a} {op} {b} = ",
              "What is {a} {op} {b}? Answer: ",
              "The value of {a} {op} {b} is "],
    "test":  ["Compute {a} {op} {b} = ",
              "Find {a} {op} {b}. Result: ",
              "Evaluate {a} {op} {b} = "],
}

model, tokenizer, device = load_model_fp32()


def b_last_pos(template, a, op, b):
    """index of the token holding the last digit of b"""
    prompt = template.format(a=a, op=SYMBOLS[op], b=b)
    prefix = template.split("{b}")[0].format(a=a, op=SYMBOLS[op]) + str(b)
    ids = tokenizer(prompt)["input_ids"]
    prefix_ids = tokenizer(prefix)["input_ids"]
    assert ids[:len(prefix_ids)] == prefix_ids, f"b merged with the next token: {prompt!r}"
    pos = len(prefix_ids) - 1
    assert tokenizer.decode([ids[pos]]) == str(b)[-1], f"wrong b position: {prompt!r}"
    return pos


def all_correct(pair, templates):
    """clean model gets both orders, both ops, every wording in this split right"""
    a, b = pair
    return all(
        answer_metrics(model, tokenizer, device,
                       t.format(a=x, op=SYMBOLS[op], b=y), ANSWER[op](x, y))[0]
        for t in templates for op in ["mul", "add"] for (x, y) in [(a, b), (b, a)]
    )


# split the NUMBERS first, so no test number ever shows up in a train prompt
numbers = list(range(LOW, HIGH + 1))
random.Random(SEED).shuffle(numbers)
half = len(numbers) // 2 + 1
groups = {"train": sorted(numbers[:half]), "test": sorted(numbers[half:])}
print(f"train numbers: {groups['train']}")
print(f"test numbers:  {groups['test']}")

# every unordered pair {a, b} with a != b inside each group, kept if the clean model gets it right
chosen = {}
for split, nums in groups.items():
    pairs = [(a, b) for i, a in enumerate(nums) for b in nums[i + 1:]]
    chosen[split] = [p for p in pairs if all_correct(p, TEMPLATES[split])]
    print(f"{split}: {len(chosen[split])} of {len(pairs)} pairs kept")

assert not set(groups["train"]) & set(groups["test"])

# hold out 20% of the test pairs as a validation set (same idea as Tyler and Julian):
# it is used to pick settings like the residual fraction, so the test set never picks anything
VALIDATION_FRACTION = 0.2
validation = set(random.Random(SEED).sample(chosen["test"], round(len(chosen["test"]) * VALIDATION_FRACTION)))
print(f"validation: {len(validation)} of the {len(chosen['test'])} test pairs")

rows = []
for group, split_pairs in chosen.items():
    for pair_id, (a, b) in enumerate(split_pairs):
        split = "validation" if (a, b) in validation else group
        for x, y in [(a, b), (b, a)]:
            for t_id, template in enumerate(TEMPLATES[group]):
                for op in ["mul", "add"]:
                    prompt = template.format(a=x, op=SYMBOLS[op], b=y)
                    other = "add" if op == "mul" else "mul"
                    rows.append({
                        "prompt": prompt, "a": x, "b": y, "op": op,
                        "answer": ANSWER[op](x, y),
                        "other_answer": ANSWER[other](x, y),
                        "split": split,
                        "template_id": t_id,
                        "pair_id": pair_id,
                        "op_pos": find_op_pos(tokenizer, prompt, op),
                        "b_pos": b_last_pos(template, x, op, y),
                        "final_pos": len(tokenizer(prompt)["input_ids"]) - 1,
                    })

with open("reuse_dataset.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

for split in ["train", "validation", "test"]:
    for op in ["mul", "add"]:
        sub = [r for r in rows if r["split"] == split and r["op"] == op]
        print(f"{split:10s} {op}: {len(sub)} prompts")

print("\none example per wording:")
for split in ["train", "test"]:
    for t_id in range(3):
        r = next(r for r in rows if r["split"] == split and r["template_id"] == t_id)
        ids = tokenizer(r["prompt"])["input_ids"]
        show = lambda p: repr(tokenizer.decode([ids[p]]))
        print(f"  [{split}] {r['prompt']!r} -> {r['answer']}   "
              f"op #{r['op_pos']} {show(r['op_pos'])}, b #{r['b_pos']} {show(r['b_pos'])}, "
              f"final #{r['final_pos']} {show(r['final_pos'])}")