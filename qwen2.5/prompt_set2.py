OPERAND_SETS = [(5, 9), (14, 20), (10, 16)]

OPERATIONS = {
    "+": {"symbol": "+", "word": "plus"},
    "-": {"symbol": "-", "word": "minus"},
    "×": {"symbol": "×", "word": "times"},
    "÷": {"symbol": "÷", "word": "divided by"},
}

WORDINGS = ["Calculate", "Compute", "What is"]

def build_prompt(a, b, op, wording, surface="symbol"):
    op_form = OPERATIONS[op][surface]
    return f"{wording} {a} {op_form} {b}"

# anchor settings used as the fixed baseline for every comparison group
ANCHOR_OP = "×"
ANCHOR_WORDING = "Calculate"
ANCHOR_SURFACE = "symbol"
ANCHOR_OPERANDS = OPERAND_SETS[0]  # (5, 9)

prompt_set2 = {

    # same operands, same operation, same surface form -> wording varies
    "wording_only": [
        (build_prompt(a, b, ANCHOR_OP, ANCHOR_WORDING, ANCHOR_SURFACE),
         build_prompt(a, b, ANCHOR_OP, w, ANCHOR_SURFACE))
        for (a, b) in OPERAND_SETS
        for w in WORDINGS if w != ANCHOR_WORDING
    ],

    # same operands, same wording, same surface form -> operation varies
    "operation_only": [
        (build_prompt(a, b, ANCHOR_OP, ANCHOR_WORDING, ANCHOR_SURFACE),
         build_prompt(a, b, op, ANCHOR_WORDING, ANCHOR_SURFACE))
        for (a, b) in OPERAND_SETS
        for op in OPERATIONS if op != ANCHOR_OP
    ],

    # same operation, same wording, same surface form -> operands vary
    "operand_only": [
        (build_prompt(ANCHOR_OPERANDS[0], ANCHOR_OPERANDS[1], ANCHOR_OP, ANCHOR_WORDING, ANCHOR_SURFACE),
         build_prompt(a, b, ANCHOR_OP, ANCHOR_WORDING, ANCHOR_SURFACE))
        for (a, b) in OPERAND_SETS if (a, b) != ANCHOR_OPERANDS
    ],

    # same operands, same operation, same wording -> surface form varies (symbol vs word)
    "surface_form_only": [
        (build_prompt(a, b, ANCHOR_OP, ANCHOR_WORDING, "symbol"),
         build_prompt(a, b, ANCHOR_OP, ANCHOR_WORDING, "word"))
        for (a, b) in OPERAND_SETS
    ],
}

if __name__ == "__main__":
    for category, pairs in prompt_set2.items():
        print(f"\n{category}: {len(pairs)} pairs")
        for a, b in pairs:
            print(f"  '{a}'  vs  '{b}'")