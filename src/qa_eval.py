"""
QA evaluation for knowledge assessment via greedy text generation.

Generates 50 tokens from a question prompt and checks if expected answer
keywords appear in the generated text (case-insensitive substring matching).

Used to assess how much a model knows about HP (should be high, >0.5)
vs TOFU (should be very low, <0.1).
"""

import torch
from tqdm import tqdm
from typing import List, Dict


def _check_keyword_match(expected: str, generated: str) -> bool:
    keywords = [kw.strip() for kw in expected.split() if len(kw.strip()) > 2]
    if not keywords:
        return expected in generated
    return all(kw in generated for kw in keywords)


QWEN1_MODELS = {"Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"}


def _needs_no_cache(model_id: str) -> bool:
    """Qwen-1 models index past_key_values as tuple; DynamicCache breaks them."""
    return model_id in QWEN1_MODELS


@torch.no_grad()
def run_qa_eval_per_question(
    model,
    tokenizer,
    pairs: List[Dict],
    max_new_tokens: int = 50,
    use_cache: bool = True,
) -> List[bool]:
    """
    Per-question QA evaluation. Returns List[bool] (one per pair).

    use_cache=False required for Qwen-1 models (DynamicCache incompatibility).
    All other models should use use_cache=True (much faster generation).
    """
    device = next(model.parameters()).device
    results = []

    for pair in tqdm(pairs, desc="QA Eval"):
        expected = pair["answer"].lower().strip()
        prompt = f"{pair['question']}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        output_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
        generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).lower()
        results.append(_check_keyword_match(expected, generated))

    return results


@torch.no_grad()
def run_qa_eval(
    model,
    tokenizer,
    pairs: List[Dict],
    max_new_tokens: int = 50,
    use_cache: bool = True,
) -> float:
    """
    Evaluate QA success rate via greedy generation and keyword matching.
    Returns fraction of correctly answered questions in [0, 1].
    """
    results = run_qa_eval_per_question(model, tokenizer, pairs, max_new_tokens, use_cache=use_cache)
    return sum(results) / len(results) if results else 0.0
