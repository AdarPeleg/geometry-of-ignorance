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


@torch.no_grad()
def run_qa_eval(
    model,
    tokenizer,
    pairs: List[Dict],
    max_new_tokens: int = 50,
) -> float:
    """
    Evaluate QA success rate via greedy generation and keyword matching.

    Args:
        model: Loaded LLM
        tokenizer: Corresponding tokenizer
        pairs: List of dicts with 'question' and 'answer' keys
        max_new_tokens: Max tokens to generate (default 50)

    Returns:
        Fraction of correctly answered questions (in [0, 1])

    Evaluation procedure:
        1. For each (question, expected_answer) pair:
        2. Prompt: "{question}\nAnswer:"
        3. Generate max_new_tokens via greedy decoding
        4. Extract only newly generated tokens (not the prompt)
        5. Check if all words from expected_answer (len>2) appear in generation
        6. Count successes

    Notes:
        - Plain prompt format (no chat template) for consistent cross-model eval
        - Greedy decoding (do_sample=False, temperature=1.0)
        - Keyword matching: case-insensitive, substring search
        - Handles multi-word answers (e.g. "Albus Dumbledore" → both words must appear)
    """
    correct = 0
    total = len(pairs)

    device = next(model.parameters()).device

    for pair in tqdm(pairs, desc="QA Eval"):
        question = pair["question"]
        expected = pair["answer"].lower().strip()

        # Format: question + answer prompt (without the answer itself)
        prompt = f"{question}\nAnswer:"

        # Tokenize the prompt
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        # Generate — use_cache=False avoids DynamicCache incompatibility with
        # Qwen-1 models (modeling_qwen.py indexes past_key_values as a tuple,
        # but transformers>=4.44 passes a DynamicCache object instead).
        output_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

        # Extract only newly generated tokens
        new_tokens = output_ids[0][prompt_len:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).lower()

        # Keyword matching
        keywords = [kw.strip() for kw in expected.split() if len(kw.strip()) > 2]

        if not keywords:
            # Single short word: exact substring match
            if expected in generated:
                correct += 1
        else:
            # All keywords must appear somewhere in generation
            if all(kw in generated for kw in keywords):
                correct += 1

    return correct / total if total > 0 else 0.0
