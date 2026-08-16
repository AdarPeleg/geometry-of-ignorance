#!/usr/bin/env python3
"""
Build Star Wars QA pairs for RQ1 third-dataset robustness experiment.

Source: data/concepts/star_wars.json (already built by build_concept_pairs.py).
Uses the qa_eval_pairs which have proper named-entity anonymization:
    "Luke Skywalker" -> "Kai Starborn", "Han Solo" -> "Riff Logan", etc.

This gives ~25 topically coherent questions all about Star Wars — same
domain structure as HP (all about one fictional universe the model knows well).

Expected result: COHERENT (high Centroid_Norm, low AUSS_L2) like HP,
since the model is well-trained on Star Wars content.

Output: data/sw_pairs.json (same format as hp_pairs.json / tofu_pairs.json)
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "data" / "concepts" / "star_wars.json"
OUT  = ROOT / "data" / "sw_pairs.json"


def main():
    with open(SRC) as f:
        sw = json.load(f)

    pairs = []
    for i, p in enumerate(sw["qa_eval_pairs"]):
        # Use the first anonymization variant (index 0) for consistency
        anon_q = p["anon_questions"][0]
        pairs.append({
            "id": f"sw_{i+1:03d}",
            "question": p["question"],
            "answer": p["answer"],
            "anon_question": anon_q,
            "anon_answer": p["answer"],   # answer text unchanged (same as HP)
            "category": "star_wars",
        })

    print(f"Built {len(pairs)} Star Wars pairs")
    print("Example:")
    print(f"  Q: {pairs[0]['question']}")
    print(f"  A: {pairs[0]['answer']}")
    print(f"  anon_Q: {pairs[0]['anon_question']}")

    with open(OUT, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
