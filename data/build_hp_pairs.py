"""
Build HP (Harry Potter) QA dataset from MUSE benchmark.

Source: https://github.com/jaechan-repo/muse-bench
Uses: muse-bench/MUSE-Books, knowmem config, forget_qa split

Process:
  1. Load HP QA pairs from HuggingFace
  2. Filter for short factual answers (len <= 6 words)
  3. Apply fictional-name anonymisation via generate_anon_qa.HP_REPLACEMENTS (variant 0)
     — same table used in RQ2 steering anonymisation for methodological consistency
  4. Save to hp_pairs.json

Output: data/hp_pairs.json with 96 pairs
Backup: data/hp_pairs_v1.json (original "Person X" anonymisation preserved)
"""

import json
import os
import re
import sys
import random
import shutil
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets not installed. Run: pip install datasets")
    sys.exit(1)

# Import shared replacement table from generate_anon_qa.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from generate_anon_qa import HP_REPLACEMENTS, apply_replacements


def anonymise_text(text: str, version_idx: int = 0) -> str:
    """
    Apply fictional-name anonymisation using HP_REPLACEMENTS (variant version_idx).
    Longest patterns are applied first to avoid partial replacements.
    """
    # Sort by pattern length descending (longest-first) to avoid partial matches
    sorted_subs = sorted(HP_REPLACEMENTS, key=lambda x: len(x[0]), reverse=True)
    return apply_replacements(text, sorted_subs, version_idx)


def infer_category(anon_question: str, anon_answer: str) -> str:
    """Infer question category from anonymised text."""
    if any(c in anon_answer.lower() for c in ["oliver", "liam", "vera", "gareth", "voss"]):
        return "characters"
    elif any(w in anon_question.lower() for w in ["academy", "institute", "school", "house"]):
        return "locations"
    elif any(w in anon_question.lower() + anon_answer.lower() for w in ["draught", "elixir", "disarmicus", "releaso"]):
        return "spells_objects"
    else:
        return "events_plot"


def build_hp_pairs(output_path: str = "data/hp_pairs.json", n_pairs: int = 100):
    """Build HP dataset with fictional-name anonymisation."""
    output_path = Path(output_path)

    # Backup existing pairs if present
    if output_path.exists():
        v1_path = output_path.parent / "hp_pairs_v1.json"
        if not v1_path.exists():
            shutil.copy(output_path, v1_path)
            print(f"Backed up existing pairs → {v1_path}")
        else:
            print(f"Backup already exists at {v1_path}, skipping backup")

    print("Loading MUSE-Books from HuggingFace (muse-bench/MUSE-Books, knowmem, forget_qa)...")
    ds = load_dataset("muse-bench/MUSE-Books", "knowmem", split="forget_qa")
    raw_pairs = []
    for entry in ds:
        q = entry.get("question", "")
        a = entry.get("answer", "")
        if q and a:
            raw_pairs.append({"question": q, "answer": a})
    print(f"Loaded {len(raw_pairs)} pairs from MUSE-Books forget_qa split")

    # Filter: short factual answers (<=6 words)
    filtered = [p for p in raw_pairs if 0 < len(p["answer"].split()) <= 6]
    print(f"Filtered to {len(filtered)} pairs (short answers ≤6 words)")

    # Deterministic shuffle + take top n_pairs
    random.seed(42)
    random.shuffle(filtered)
    filtered = filtered[:n_pairs]

    # Build output pairs
    output_pairs = []
    n_changed = 0
    for idx, pair in enumerate(filtered):
        q, a = pair["question"], pair["answer"]
        anon_q = anonymise_text(q, version_idx=0)
        anon_a = anonymise_text(a, version_idx=0)

        if anon_q != q or anon_a != a:
            n_changed += 1

        output_pairs.append({
            "id": f"hp_{idx+1:03d}",
            "question": q,
            "answer": a,
            "anon_question": anon_q,
            "anon_answer": anon_a,
            "category": infer_category(anon_q, anon_a),
        })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_pairs, f, indent=2, ensure_ascii=False)

    total = len(output_pairs)
    print(f"\nSaved {total} HP pairs to {output_path}")
    print(f"Anonymisation coverage: {n_changed}/{total} pairs changed "
          f"({100*n_changed/total:.0f}%)")

    if output_pairs:
        sample = output_pairs[0]
        print("\nSample pair:")
        print(f"  Q:    {sample['question']}")
        print(f"  A:    {sample['answer']}")
        print(f"  Anon Q: {sample['anon_question']}")
        print(f"  Anon A: {sample['anon_answer']}")


if __name__ == "__main__":
    build_hp_pairs()
