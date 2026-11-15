import json

from transformers import AutoTokenizer

from utilities import MODEL_ID

# Tyler and Julian's frozen-reuse dataset, copied from their repo
# (block_15_testing/frozen_reuse_results/manifest.json)
MANIFEST = "manifest.json"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

examples = json.load(open(MANIFEST))["examples"]
templates = sorted({e["template"] for e in examples})


def token_at(offsets, span):
    """the LAST token that overlaps a character span [start, end)"""
    hits = [i for i, (s, t) in enumerate(offsets) if s < span[1] and t > span[0]]
    assert hits, span
    return hits[-1]


# write each of their suites in the same format as our reuse_dataset.json
for suite in ["pair_holdout", "number_holdout"]:
    rows = []
    for e in [e for e in examples if e["suite"] == suite]:
        enc = tokenizer(e["prompt"], return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        b_pos = token_at(offsets, e["b_span"])
        assert tokenizer.decode([ids[b_pos]]) == str(e["b"])[-1], e["prompt"]
        rows.append({
            "prompt": e["prompt"], "a": e["a"], "b": e["b"], "op": e["op"],
            "answer": e["answer"], "other_answer": e["other_answer"],
            "split": e["split"],                       # train / validation / test
            "template_id": templates.index(e["template"]),
            "template": e["template"],
            "pair_id": e["source_index"],
            "op_pos": token_at(offsets, e["operation_span"]),
            "b_pos": b_pos,
            "final_pos": len(ids) - 1,
        })
    with open(f"their_{suite}.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n{suite} -> their_{suite}.json")
    for s in ["train", "validation", "test"]:
        print(f"  {s:10s} mul {sum(r['split'] == s and r['op'] == 'mul' for r in rows):4d}"
              f"   add {sum(r['split'] == s and r['op'] == 'add' for r in rows):4d}")
    for t in sorted({r["template"] for r in rows}):
        r = next(r for r in rows if r["template"] == t)
        ids = tokenizer(r["prompt"])["input_ids"]
        show = lambda p: repr(tokenizer.decode([ids[p]]))
        print(f"  {r['prompt']!r}  op #{r['op_pos']} {show(r['op_pos'])}, b #{r['b_pos']} {show(r['b_pos'])}, "
              f"final #{r['final_pos']} {show(r['final_pos'])}")
