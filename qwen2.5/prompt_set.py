prompt_set = {
    "minor_changes" : [
        ("Calculate 17 x 23", "Compute 17 x 23"),
        ("Calculate 9 x 14", "Compute 9 x 14"),
        ("Calculate 32 x 5", "Compute 32 x 5"),
    ],
    "different_wording_same_meaning" : [
        ("What is the product of 17 and 23", "Multiply 23 by 17"),
        ("What is the product of 9 and 14", "Multiply 14 by 9"),
        ("What is the product of 32 and 5", "Multiply 5 by 32"),
    ],
    "look_similar_diff_computation" : [
        ("Calculate 17 x 23", "Calculate 17 + 23"),
        ("Calculate 9 x 14", "Calculate 9 + 14"),
        ("Calculate 32 x 5", "Calculate 32 - 5"),
    ],
    "unrelated" : [
        ("Calculate 17 x 23", "What is the capital of France?"),
        ("Calculate 9 x 14", "Describe the water cycle"),
        ("Calculate 32 x 5", "Who wrote Romeo and Juliet?"),
    ],
}

if __name__ == "__main__":
    total = sum(len(v) for v in prompt_set.values())
    for category, pairs in prompt_set.items():
        print(f"{category}: {len(pairs)} pairs")
    print(f"Total pairs: {total}")