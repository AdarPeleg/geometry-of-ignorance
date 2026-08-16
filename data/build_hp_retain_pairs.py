#!/usr/bin/env python3
"""
Build HP retain QA pairs from MUSE-Books retain_qa split.

Source: muse-bench/MUSE-Books, knowmem config, retain_qa split (100 pairs)
Same anonymisation pipeline as build_hp_pairs.py (forget_qa).

Output: data/hp_retain_pairs.json
"""

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from generate_anon_qa import HP_REPLACEMENTS, apply_replacements


def anonymise_text(text: str, version_idx: int = 0) -> str:
    sorted_subs = sorted(HP_REPLACEMENTS, key=lambda x: len(x[0]), reverse=True)
    return apply_replacements(text, sorted_subs, version_idx)


def build_hp_retain_pairs(output_path: str = "data/hp_retain_pairs.json"):
    out = Path(output_path)
    print("Loading MUSE-Books retain_qa split...")
    ds = load_dataset("muse-bench/MUSE-Books", "knowmem", split="retain_qa")

    raw = [{"question": e["question"], "answer": e["answer"]} for e in ds if e["question"] and e["answer"]]
    print(f"Loaded {len(raw)} pairs")

    # Same filter as hp_pairs.py: short factual answers
    filtered = [p for p in raw if 0 < len(p["answer"].split()) <= 6]
    print(f"After ≤6 word answer filter: {len(filtered)} pairs")

    random.seed(42)
    random.shuffle(filtered)

    pairs = []
    for idx, p in enumerate(filtered):
        q, a = p["question"], p["answer"]
        anon_q = anonymise_text(q)
        anon_a = anonymise_text(a)
        pairs.append({
            "id": f"hp_retain_{idx+1:03d}",
            "question": q,
            "answer": a,
            "anon_question": anon_q,
            "anon_answer": anon_a,
            "category": "hp_retain",
        })

    with open(out, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} pairs → {out}")

    n_anon = sum(1 for p in pairs if p["anon_question"] != p["question"])
    print(f"  {n_anon}/{len(pairs)} questions have at least one name substitution")
    print(f"  Examples:")
    for p in pairs[:3]:
        print(f"    Q: {p['question']}")
        print(f"    anon: {p['anon_question']}")
        print(f"    A: {p['answer']}")
        print()


if __name__ == "__main__":
    build_hp_retain_pairs()
