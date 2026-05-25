"""
Build concept pair datasets for RQ2 unlearning experiments.

Source: YihuaiHong/ConceptVectors (HuggingFace / local cache)
Each concept produces a JSON file with forget_pairs, qa_eval_pairs, and retain_pairs
in the standard format used throughout this project.

Forget pairs come from two sources:
  1. Wikipedia text of the concept — split into (prefix, completion) sliding windows
  2. ConceptVectors text_completion pairs — authentic examples from the dataset

QA eval pairs have hardcoded answers (for well-known concepts) — used to verify
whether unlearning worked, not for training.

Retain pairs come from ConceptVectors unrelated_QA of unrelated concepts (concepts
the model should NOT forget during training).

Modular: pass --concept <name> to generate pairs for any concept in the dataset.
For new concepts without hardcoded QA answers, only text_completion pairs are used.

Usage:
  python data/build_concept_pairs.py --concept "Harry Potter"
  python data/build_concept_pairs.py --concept "Star Wars"
  python data/build_concept_pairs.py --concept "William Shakespeare"
  python data/build_concept_pairs.py --all  # build all three default concepts

Output: data/concepts/<slug>.json  (e.g. harry_potter.json)
"""

import json
import os
import re
import random
import argparse
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────

CV_DATA_DIR = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--YihuaiHong--ConceptVectors"
        "/snapshots/145c8484eced9306b2ad78238b9bc3324ffd7741/ConceptVectors_data"
    )
)
CV_FILES = ["llama2-7b_concepts_test.json", "olmo-7b_concepts_test.json",
            "llama2-7b_concepts_dev.json", "olmo-7b_concepts_dev.json"]

OUTPUT_DIR = Path(__file__).parent / "concepts"

# ── Concept-specific anonymisation maps ────────────────────────────────────────
# Keys are sorted longest-first to prevent partial replacements.

ANON_MAPS: Dict[str, Dict[str, str]] = {
    "Harry Potter": {
        "Harry Potter and the Philosopher's Stone": "[BOOK_1]",
        "Harry Potter and the Chamber of Secrets": "[BOOK_2]",
        "Harry Potter and the Prisoner of Azkaban": "[BOOK_3]",
        "Harry Potter and the Goblet of Fire": "[BOOK_4]",
        "Harry Potter and the Order of the Phoenix": "[BOOK_5]",
        "Harry Potter and the Half-Blood Prince": "[BOOK_6]",
        "Harry Potter and the Deathly Hallows": "[BOOK_7]",
        "Harry Potter": "[THE PROTAGONIST]",
        "Hermione Granger": "[CHARACTER_B]",
        "Hermione": "[CHARACTER_B]",
        "Ron Weasley": "[CHARACTER_C]",
        "Ron": "[CHARACTER_C]",
        "Albus Dumbledore": "[THE HEADMASTER]",
        "Dumbledore": "[THE HEADMASTER]",
        "Lord Voldemort": "[THE ANTAGONIST]",
        "Voldemort": "[THE ANTAGONIST]",
        "Tom Riddle": "[THE ANTAGONIST]",
        "Severus Snape": "[PROFESSOR_X]",
        "Snape": "[PROFESSOR_X]",
        "Rubeus Hagrid": "[THE GROUNDSKEEPER]",
        "Hagrid": "[THE GROUNDSKEEPER]",
        "Neville Longbottom": "[CHARACTER_D]",
        "Draco Malfoy": "[CHARACTER_E]",
        "J. K. Rowling": "[THE AUTHOR]",
        "J.K. Rowling": "[THE AUTHOR]",
        "Rowling": "[THE AUTHOR]",
        "Hogwarts": "[THE SCHOOL]",
        "Diagon Alley": "[THE MARKET]",
        "Hogsmeade": "[THE VILLAGE]",
        "Quidditch": "[THE SPORT]",
        "Horcrux": "[THE OBJECT]",
        "Gryffindor": "[HOUSE_A]",
        "Slytherin": "[HOUSE_B]",
        "Hufflepuff": "[HOUSE_C]",
        "Ravenclaw": "[HOUSE_D]",
        "Muggle": "[NON_PRACTITIONER]",
    },
    "Star Wars": {
        "Star Wars: Episode": "[THE FRANCHISE]: Episode",
        "Star Wars": "[THE FRANCHISE]",
        "George Lucas": "[THE CREATOR]",
        "Luke Skywalker": "[THE PROTAGONIST]",
        "Darth Vader": "[THE ANTAGONIST]",
        "Anakin Skywalker": "[THE ANTAGONIST]",
        "Han Solo": "[CHARACTER_B]",
        "Princess Leia": "[CHARACTER_C]",
        "Leia Organa": "[CHARACTER_C]",
        "Obi-Wan Kenobi": "[CHARACTER_D]",
        "Ben Kenobi": "[CHARACTER_D]",
        "Yoda": "[THE MASTER]",
        "Palpatine": "[THE EMPEROR]",
        "Emperor Palpatine": "[THE EMPEROR]",
        "Darth Sidious": "[THE EMPEROR]",
        "Rey": "[CHARACTER_E]",
        "Finn": "[CHARACTER_F]",
        "Millennium Falcon": "[THE SHIP]",
        "Tatooine": "[THE DESERT PLANET]",
        "Death Star": "[THE SUPERWEAPON]",
        "Jedi": "[THE ORDER]",
        "Sith": "[THE DARK ORDER]",
        "The Force": "[THE POWER]",
        "lightsaber": "[THE WEAPON]",
        "Lightsaber": "[THE WEAPON]",
        "Wookiee": "[THE SPECIES]",
        "Chewbacca": "[THE COMPANION]",
        "R2-D2": "[DROID_A]",
        "C-3PO": "[DROID_B]",
    },
    "William Shakespeare": {
        "William Shakespeare": "[THE PLAYWRIGHT]",
        "Shakespeare": "[THE PLAYWRIGHT]",
        "Hamlet": "[PLAY_A]",
        "Macbeth": "[PLAY_B]",
        "Othello": "[PLAY_C]",
        "Romeo and Juliet": "[PLAY_D]",
        "King Lear": "[PLAY_E]",
        "A Midsummer Night's Dream": "[PLAY_F]",
        "The Tempest": "[PLAY_G]",
        "Twelfth Night": "[PLAY_H]",
        "As You Like It": "[PLAY_I]",
        "The Globe Theatre": "[THE THEATRE]",
        "Globe Theatre": "[THE THEATRE]",
        "the Globe": "[THE THEATRE]",
        "Stratford-upon-Avon": "[THE BIRTHPLACE]",
        "Stratford": "[THE BIRTHPLACE]",
        "First Folio": "[THE COLLECTION]",
        "Lord Chamberlain's Men": "[THE COMPANY]",
        "King's Men": "[THE COMPANY]",
    },
}

# ── Hardcoded QA answers for evaluation ────────────────────────────────────────
# These are used ONLY for measuring unlearning success (not for training).

QA_ANSWERS: Dict[str, List[Tuple[str, str]]] = {
    "Harry Potter": [
        ("Who is the author of the Harry Potter book series?", "J.K. Rowling"),
        ("What is the name of the first book in the Harry Potter series?",
         "Harry Potter and the Philosopher's Stone"),
        ("Which magical school does Harry Potter attend?", "Hogwarts"),
        ("What are the names of Harry Potter's two best friends?",
         "Hermione Granger and Ron Weasley"),
        ("What creature is Harry Potter's pet?", "owl"),
        ("What is the name of the dark wizard who killed Harry Potter's parents?",
         "Voldemort"),
        ("What is the name of the sport played on broomsticks in the Harry Potter series?",
         "Quidditch"),
        ("What are the three Deathly Hallows in the final book of the series?",
         "the Elder Wand, the Resurrection Stone, and the Invisibility Cloak"),
    ],
    "Star Wars": [
        ("What is the name of the desert planet where Luke Skywalker was raised?",
         "Tatooine"),
        ("Who is Darth Vader's son?", "Luke Skywalker"),
        ("What is the weapon used by Jedi Knights?", "lightsaber"),
        ("What is the name of Han Solo's ship?", "Millennium Falcon"),
        ("What color is Yoda's lightsaber?", "green"),
        ("What is the name of the furry species to which Chewbacca belongs?",
         "Wookiee"),
        ("What is the name of the Sith Lord who trained Darth Sidious?",
         "Darth Plagueis"),
        ("Who created the Star Wars franchise?", "George Lucas"),
    ],
    "William Shakespeare": [
        ("In which century did William Shakespeare live and write?",
         "16th and 17th century"),
        ("What town is traditionally considered Shakespeare's birthplace?",
         "Stratford-upon-Avon"),
        ("What is the title of Shakespeare's longest play?", "Hamlet"),
        ("What theater company was Shakespeare associated with during his career?",
         "the Lord Chamberlain's Men"),
        ("What famous London theatre was built in 1599 and associated with Shakespeare's plays?",
         "the Globe Theatre"),
        ("What is the term for the collection of Shakespeare's plays published in 1623?",
         "the First Folio"),
        ("What is the title of the Shakespeare play featuring the characters Rosalind and Orlando?",
         "As You Like It"),
        ("Which Shakespeare tragedy features a prince of Denmark?", "Hamlet"),
    ],
}


# ── Core functions ─────────────────────────────────────────────────────────────

def anonymise(text: str, anon_map: Dict[str, str]) -> str:
    """Apply anonymisation map to text. Longest keys first to avoid partial replacements."""
    sorted_keys = sorted(anon_map.keys(), key=len, reverse=True)
    for key in sorted_keys:
        text = text.replace(key, anon_map[key])
    return text


def make_completion_pairs(
    wiki_text: str,
    anon_map: Dict[str, str],
    n_pairs: int = 50,
    window_sentences: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate (question, answer, anon_question, anon_answer) pairs from Wikipedia text
    by splitting into sentence windows.

    question = window_sentences consecutive sentences (the prefix)
    answer   = the immediately following sentence (the completion)
    """
    # Strip Wikipedia section headers (== Header ==) before splitting
    wiki_text = re.sub(r'==+[^=]+==+', ' ', wiki_text)
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', wiki_text.strip())
    sentences = [
        s.strip() for s in sentences
        if len(s.strip()) > 20
        and not s.strip().startswith("==")   # filter Wikipedia section headers
        and not s.strip().startswith("[[")   # filter wiki markup
        and not re.match(r'^\s*\d+\s*$', s)  # filter lone numbers
    ]

    pairs = []
    step = max(1, window_sentences)
    for i in range(0, len(sentences) - window_sentences - 1, step):
        prefix = " ".join(sentences[i:i + window_sentences])
        completion = sentences[i + window_sentences]
        # Skip if either is too short or looks like a section header
        if len(prefix) < 50 or len(completion) < 20:
            continue
        pairs.append({
            "question": prefix,
            "answer": completion,
            "anon_question": anonymise(prefix, anon_map),
            "anon_answer": anonymise(completion, anon_map),
        })

    # Sample n_pairs with seed
    rng = random.Random(seed)
    if len(pairs) > n_pairs:
        pairs = rng.sample(pairs, n_pairs)
    else:
        # Shuffle even if fewer
        rng.shuffle(pairs)

    # Assign IDs
    for i, p in enumerate(pairs):
        p["id"] = f"completion_{i+1:03d}"

    return pairs


def make_cv_completion_pairs(cv_entry: Dict, anon_map: Dict[str, str]) -> List[Dict]:
    """Convert ConceptVectors text_completion pairs to standard format."""
    pairs = []
    for i, tc in enumerate(cv_entry.get("text_completion", [])):
        first = tc.get("First_half", "").strip()
        second = tc.get("Second_half", "").strip()
        if not first or not second:
            continue
        if "==" in first[:10] or "==" in second[:10]:
            continue  # skip section-header completions
        pairs.append({
            "id": f"cv_{i+1:03d}",
            "question": first,
            "answer": second,
            "anon_question": anonymise(first, anon_map),
            "anon_answer": anonymise(second, anon_map),
        })
    return pairs


def load_retain_pairs(
    cv_data: List[Dict],
    exclude_concept: str,
    n_retain: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """
    Build retain pairs from Wikipedia text of unrelated concepts.
    Excludes the target concept and uses Wikipedia sentences as plain text.
    """
    # Collect Wikipedia text from unrelated concepts
    retain_texts = []
    rng = random.Random(seed)
    for entry in cv_data:
        if entry["Concept"] == exclude_concept:
            continue
        wiki = entry.get("wikipedia_content", "")
        if not wiki:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', wiki.strip())
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 40:
                retain_texts.append(sent)

    rng.shuffle(retain_texts)
    retain_texts = retain_texts[:n_retain]

    pairs = []
    for i, sent in enumerate(retain_texts):
        pairs.append({
            "id": f"retain_{i+1:03d}",
            "question": sent,
            "answer": "",
        })
    return pairs


def load_cv_data() -> List[Dict]:
    """Load all ConceptVectors JSON files into one list (deduplicating by Concept)."""
    seen = set()
    all_entries = []
    for fname in CV_FILES:
        fpath = CV_DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                continue
        for entry in entries:
            concept = entry.get("Concept", "")
            key = (concept, fname)
            if key not in seen:
                seen.add(key)
                all_entries.append(entry)
    return all_entries


def build_concept(
    concept_name: str,
    cv_data: List[Dict],
    n_forget: int = 50,
    n_retain: int = 50,
    seed: int = 42,
) -> Dict:
    """
    Build the full concept JSON for a given concept name.

    Returns a dict with:
      concept, forget_pairs, qa_eval_pairs, retain_pairs
    """
    anon_map = ANON_MAPS.get(concept_name, {})

    # Find concept entry in ConceptVectors (prefer llama2 test file)
    cv_entry = next(
        (e for e in cv_data
         if e["Concept"] == concept_name and "llama2" in e.get("_source", "")),
        next((e for e in cv_data if e["Concept"] == concept_name), None)
    )
    if cv_entry is None:
        raise ValueError(f"Concept '{concept_name}' not found in ConceptVectors data.")

    # --- Forget pairs: mix of Wikipedia windows + CV text_completion ---
    wiki_pairs = make_completion_pairs(
        cv_entry["wikipedia_content"], anon_map,
        n_pairs=n_forget, window_sentences=3, seed=seed
    )
    cv_pairs = make_cv_completion_pairs(cv_entry, anon_map)

    # Combine: CV pairs first (more carefully curated), then wiki windows
    all_forget = cv_pairs + wiki_pairs
    # Deduplicate by question text
    seen_q = set()
    forget_pairs = []
    for p in all_forget:
        if p["question"] not in seen_q:
            seen_q.add(p["question"])
            forget_pairs.append(p)
        if len(forget_pairs) >= n_forget:
            break

    # Re-assign sequential IDs
    for i, p in enumerate(forget_pairs):
        p["id"] = f"forget_{i+1:03d}"

    # --- QA eval pairs (hardcoded answers for known concepts) ---
    qa_eval_pairs = []
    for i, (q, a) in enumerate(QA_ANSWERS.get(concept_name, [])):
        qa_eval_pairs.append({
            "id": f"qa_{i+1:03d}",
            "question": q,
            "answer": a,
            "anon_question": anonymise(q, anon_map),
            "anon_answer": a,  # answers are usually generic (not concept-specific names)
        })

    # --- Retain pairs from unrelated Wikipedia text ---
    retain_pairs = load_retain_pairs(cv_data, concept_name, n_retain, seed)

    return {
        "concept": concept_name,
        "n_forget": len(forget_pairs),
        "n_qa_eval": len(qa_eval_pairs),
        "n_retain": len(retain_pairs),
        "forget_pairs": forget_pairs,
        "qa_eval_pairs": qa_eval_pairs,
        "retain_pairs": retain_pairs,
    }


def slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def build_and_save(concept_name: str, cv_data: List[Dict]) -> Path:
    print(f"\nBuilding pairs for: {concept_name}")
    concept_data = build_concept(concept_name, cv_data)

    out_path = OUTPUT_DIR / f"{slug(concept_name)}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(concept_data, f, indent=2)

    print(f"  forget_pairs: {concept_data['n_forget']}")
    print(f"  qa_eval_pairs: {concept_data['n_qa_eval']}")
    print(f"  retain_pairs: {concept_data['n_retain']}")
    print(f"  Saved to {out_path}")

    # Print one example
    if concept_data["forget_pairs"]:
        ex = concept_data["forget_pairs"][0]
        print(f"\n  Sample forget pair:")
        print(f"    Q: {ex['question'][:80]}...")
        print(f"    A: {ex['answer'][:60]}...")
        print(f"    AnonQ: {ex['anon_question'][:80]}...")

    return out_path


DEFAULT_CONCEPTS = ["Harry Potter", "Star Wars", "William Shakespeare"]


def main():
    parser = argparse.ArgumentParser(description="Build concept pair datasets for RQ2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", type=str, help="Concept name (e.g. 'Harry Potter')")
    group.add_argument("--all", action="store_true",
                       help="Build all default concepts: " + ", ".join(DEFAULT_CONCEPTS))
    parser.add_argument("--n_forget", type=int, default=50,
                        help="Number of forget pairs (default: 50)")
    parser.add_argument("--n_retain", type=int, default=50,
                        help="Number of retain pairs (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading ConceptVectors data...")
    cv_data = load_cv_data()
    # Tag each entry with its source file for concept lookup
    for fname in CV_FILES:
        fpath = CV_DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            entries = json.load(f)
        for entry in cv_data:
            if entry.get("Concept") in {e["Concept"] for e in entries}:
                if "_source" not in entry:
                    entry["_source"] = fname
    print(f"Loaded {len(cv_data)} total concept entries")

    concepts = DEFAULT_CONCEPTS if args.all else [args.concept]
    for concept in concepts:
        build_and_save(concept, cv_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
