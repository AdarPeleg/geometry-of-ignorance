"""
Build TOFU (Task of Fictitious Unlearning) QA dataset.

Source: https://huggingface.co/datasets/locuslab/TOFU
Dataset: "full" split with questions about fictitious authors

TOFU contains entirely fictitious authors (people who never existed)
that models were trained on. Used as a "bogus knowledge" control
to test if geometric metrics correctly identify unknown/fabricated domains.

Process:
  1. Load TOFU dataset from HuggingFace
  2. Sample 100 entries, stratified by question type
  3. Pre-scan all pairs to collect every unique author/entity name
  4. Build a global fictional-name map using the Seyitoğlu et al. name pools
     (tofu_anonymize.json common pools: first_name/last_name lists)
  5. Apply global map consistently across all pairs (longest-first, Unicode-aware)
  6. Save to tofu_pairs.json

Output: data/tofu_pairs.json with 100 pairs
Backup: data/tofu_pairs_v1.json (original "Person X" anonymisation preserved)
"""

import json
import os
import re
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets not installed. Run: pip install datasets")
    import sys; sys.exit(1)

# ---------------------------------------------------------------------------
# Load Seyitoğlu et al. anonymization dataset
# ---------------------------------------------------------------------------
_ANON_PATH = Path(__file__).parent / "anonamized_dataset" / "tofu_anonymize.json"

with open(_ANON_PATH) as _f:
    _ANON_DATA = json.load(_f)

_COMMON = _ANON_DATA["common"]
_SEYITOGLU_REPLACEMENTS = _ANON_DATA["replacements"]

# Deterministic date substitutes (pool is null in the dataset)
_DATE_FALLBACK = [
    "15th of March, 1960", "22nd of June, 1947", "3rd of September, 1955",
    "14th of January, 1971", "8th of November, 1963", "27th of April, 1958",
    "11th of December, 1949", "5th of July, 1966", "19th of February, 1953",
    "30th of October, 1961",
]


def _build_seyitoglu_entity_map() -> Dict[str, str]:
    """
    Build a deterministic replacement map from Seyitoğlu et al. replacements.
    Person entries get sequential indices from the name pools.
    Non-person entries (locations, dates, books, etc.) get pool index 0.
    """
    entity_map: Dict[str, str] = {}
    person_idx = 0  # sequential index for person-name pool entries

    for entity, pool_spec in _SEYITOGLU_REPLACEMENTS.items():
        if isinstance(pool_spec, list):
            # Person name: combine first_name[i] + last_name[i]
            fn = _COMMON["first_name"][person_idx % len(_COMMON["first_name"])]
            ln = _COMMON["last_name"][person_idx % len(_COMMON["last_name"])]
            entity_map[entity] = f"{fn} {ln}"
            person_idx += 1
        else:
            pool_name = pool_spec
            pool = _COMMON.get(pool_name)
            if pool:
                entity_map[entity] = pool[0]
            else:
                # Date pool is null — use fallback list
                entity_map[entity] = _DATE_FALLBACK[0]

    return entity_map, person_idx  # return next unused person index


# Words that should never appear in a person name — used to filter false positives
_NON_NAME_WORDS = {
    "Award", "Awards", "Foundation", "Literature", "Fiction", "Genre", "Novel",
    "International", "Excellence", "Writers", "Writing", "Literary", "Society",
    "Association", "Institute", "Prize", "Magazine", "Mystery", "Historical",
    "Prestigious", "Outstanding", "National", "World", "Global", "American",
    "European", "Asian", "African", "Pacific", "Northern", "Southern",
    "Eastern", "Western", "War", "Love", "Life", "Death", "Dark", "Light",
    "Time", "Space", "Blood", "Fire", "Water", "Earth", "Sky", "Sea",
    "City", "Town", "Village", "Street", "Avenue", "Road", "Park",
    "University", "College", "School", "Academy", "Institute", "Center",
}


def build_global_name_map(all_pairs: List[Dict], full_dataset=None) -> Dict[str, str]:
    """
    Scan all pairs to find every unique multi-word proper name, then assign
    each a unique fictional replacement from the name pools.

    Names already in the Seyitoğlu explicit replacements keep those entries.
    Remaining names are assigned sequentially from the pools.

    Filters:
    - Only 2–4 word patterns (person names are rarely longer)
    - Must appear ≥ 3 times in the FULL dataset (authors appear ~20 times; genres/awards appear less)
    - Must not contain obvious non-name words (genres, awards, etc.)

    Uses full_dataset (the complete 4000-pair TOFU set) for frequency scanning so that
    each author's 20 appearances are counted, not just their 1–2 appearances in the sample.
    """
    # Step 1: Build Seyitoğlu explicit map
    seyitoglu_map, next_person_idx = _build_seyitoglu_entity_map()

    # Step 2: Build frequency map from FULL dataset (not just sampled pairs)
    # Each TOFU author appears in ~20 questions in the full set; genres appear far less.
    name_pattern = re.compile(
        r'\b[A-Z][a-zA-Z\-\']+(?:\s+[A-Z][a-zA-Z\-\']+)+\b'
    )
    scan_corpus = full_dataset if full_dataset is not None else all_pairs
    name_counts: Dict[str, int] = {}
    for pair in scan_corpus:
        # Only scan questions (not answers) to avoid book/award titles
        for text in [pair.get("question", "")]:
            for m in name_pattern.finditer(text):
                name = m.group(0)
                # Normalize possessive ('s) so "Schmidt's" and "Schmidt" share one key
                if name.endswith("'s"):
                    name = name[:-2]
                words = name.split()
                if 2 <= len(words) <= 4:
                    name_counts[name] = name_counts.get(name, 0) + 1

    # Keep names that appear ≥ 3 times AND don't contain non-name words
    all_names: set = set()
    for name, count in name_counts.items():
        if count >= 3:
            words = name.split()
            if not any(w in _NON_NAME_WORDS for w in words):
                all_names.add(name)

    # Step 3: Determine which names need new assignments
    already_mapped = set(seyitoglu_map.keys())
    new_names = sorted(all_names - already_mapped)  # deterministic sort

    # Collect pool indices already used by Seyitoğlu person entries
    used_indices: set = set(range(next_person_idx))

    # Step 4: Assign sequential fictional names from the pools
    first_names = _COMMON["first_name"]
    last_names = _COMMON["last_name"]
    full_map = dict(seyitoglu_map)

    pool_i = 0
    for name in new_names:
        while pool_i in used_indices:
            pool_i += 1
        fn = first_names[pool_i % len(first_names)]
        ln = last_names[pool_i % len(last_names)]
        full_map[name] = f"{fn} {ln}"
        used_indices.add(pool_i)
        pool_i += 1

    # Also add possessive variants (Author's → fictional_name's) so they're consistent
    possessive_additions = {}
    for entity, replacement in full_map.items():
        possessive_key = entity + "'s"
        if possessive_key not in full_map:
            possessive_additions[possessive_key] = replacement + "'s"
    full_map.update(possessive_additions)

    return full_map


def apply_global_map(text: str, sorted_subs: List[Tuple[str, str]]) -> str:
    """Apply pre-sorted (longest-first) replacement list to text."""
    result = text
    for pattern, replacement in sorted_subs:
        result = re.sub(re.escape(pattern), replacement, result, flags=re.IGNORECASE)
    return result


def build_tofu_pairs(output_path: str = "data/tofu_pairs.json", n_pairs: int = 100):
    """Build TOFU dataset with consistent fictional-name anonymisation."""
    output_path = Path(output_path)

    # Backup existing pairs if present
    if output_path.exists():
        v1_path = output_path.parent / "tofu_pairs_v1.json"
        if not v1_path.exists():
            shutil.copy(output_path, v1_path)
            print(f"Backed up existing pairs → {v1_path}")
        else:
            print(f"Backup already exists at {v1_path}, skipping backup")

    print("Loading TOFU dataset from HuggingFace...")
    raw = load_dataset("locuslab/TOFU", "full")
    if hasattr(raw, "keys"):
        split_name = "train" if "train" in raw else list(raw.keys())[0]
        ds = raw[split_name]
    else:
        ds = raw
    print(f"Loaded {len(ds)} TOFU pairs")

    # Stratified sampling by question type
    pairs_by_type: Dict[str, list] = {}
    for entry in ds:
        q = entry.get("question", entry.get("text", ""))
        a = entry.get("answer", entry.get("target", ""))
        q_lower = q.lower()
        if "birth" in q_lower or "born" in q_lower:
            q_type = "biography"
        elif "book" in q_lower or "wrote" in q_lower or "author" in q_lower:
            q_type = "works"
        elif "award" in q_lower or "prize" in q_lower:
            q_type = "achievements"
        else:
            q_type = "other"
        pairs_by_type.setdefault(q_type, []).append((q, a, q_type))

    random.seed(42)
    sampled: List[Tuple[str, str, str]] = []
    pairs_per_type = max(1, n_pairs // len(pairs_by_type))
    for q_type, type_pairs in pairs_by_type.items():
        taken = random.sample(type_pairs, min(pairs_per_type, len(type_pairs)))
        sampled.extend(taken)
        if len(sampled) >= n_pairs:
            break
    sampled = sampled[:n_pairs]

    # Build raw pair dicts for name-map scanning
    raw_pairs_for_scan = [{"question": q, "answer": a} for q, a, _ in sampled]

    # Build global fictional-name map using FULL dataset for frequency scanning
    # (each author appears ~20 times in full 4000-pair dataset → reliably captured at freq≥3)
    print("Building global author-name replacement map (scanning full dataset)...")
    full_ds_pairs = [{"question": e.get("question", e.get("text", "")),
                      "answer": e.get("answer", e.get("target", ""))}
                     for e in ds]
    global_map = build_global_name_map(raw_pairs_for_scan, full_dataset=full_ds_pairs)

    # Sort by pattern length descending (longest-first)
    sorted_subs = sorted(global_map.items(), key=lambda x: len(x[0]), reverse=True)

    print(f"Replacement map: {len(sorted_subs)} entities")
    for entity, replacement in sorted_subs[:10]:
        print(f"  {repr(entity)} → {repr(replacement)}")
    if len(sorted_subs) > 10:
        print(f"  ... and {len(sorted_subs)-10} more")

    # Apply map to all pairs
    output_pairs = []
    n_changed = 0
    for idx, (q, a, q_type) in enumerate(sampled):
        anon_q = apply_global_map(q, sorted_subs)
        anon_a = apply_global_map(a, sorted_subs)
        if anon_q != q or anon_a != a:
            n_changed += 1
        output_pairs.append({
            "id": f"tofu_{idx+1:03d}",
            "question": q,
            "answer": a,
            "anon_question": anon_q,
            "anon_answer": anon_a,
            "category": q_type,
        })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_pairs, f, indent=2, ensure_ascii=False)

    total = len(output_pairs)
    print(f"\nSaved {total} TOFU pairs to {output_path}")
    print(f"Anonymisation coverage: {n_changed}/{total} pairs changed "
          f"({100*n_changed/total:.0f}%)")

    if output_pairs:
        sample = output_pairs[0]
        print("\nSample pair:")
        print(f"  Q:      {sample['question']}")
        print(f"  A:      {sample['answer']}")
        print(f"  Anon Q: {sample['anon_question']}")
        print(f"  Anon A: {sample['anon_answer']}")


if __name__ == "__main__":
    build_tofu_pairs()
