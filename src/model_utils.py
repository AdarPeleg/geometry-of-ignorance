"""
Model loading utilities for RQ1 experiments.

Handles all 10 base models including gated Llama/Gemma families.
Uses bfloat16 + device_map='auto' for efficient multi-GPU support.
Provides resumable model loading, layer detection, and GPU memory cleanup.
"""

import torch
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Registry of all 10 base models used in RQ1
MODEL_REGISTRY = {
    "Qwen/Qwen-7B-Chat":                    {"gated": False},
    "Qwen/Qwen-14B-Chat":                   {"gated": False},
    "Qwen/Qwen2.5-7B-Instruct":             {"gated": False},
    "Qwen/Qwen2.5-14B-Instruct":            {"gated": False},
    "google/gemma-2b-it":                   {"gated": True},
    "google/gemma-7b-it":                   {"gated": True},
    "meta-llama/Llama-2-7b-chat-hf":        {"gated": True},
    "meta-llama/Llama-2-13b-chat-hf":       {"gated": True},
    "meta-llama/Meta-Llama-3-8B-Instruct":  {"gated": True},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"gated": True},
}


def load_model_and_tokenizer(model_id: str, hf_token: str = None):
    """
    Load a model and tokenizer in bfloat16 with device_map='auto'.

    Args:
        model_id: HuggingFace model identifier (e.g. 'Qwen/Qwen-7B-Chat')
        hf_token: HuggingFace API token for gated models

    Returns:
        (model, tokenizer): Both set to eval mode, bfloat16 dtype

    Notes:
        - device_map='auto' automatically distributes large models across GPUs
        - trust_remote_code=True required for Qwen models
        - Ensures pad_token is set (required for attention masks)
        - Qwen models force attn_implementation="eager": the default SDPA backward
          kernel produces NaN gradients on Qwen2.5-7B-Instruct in bf16 during
          unlearning training (confirmed via isolated grad-norm trace: SDPA backward
          -> grad_norm=nan on step 0 of a fresh, unmodified model; eager backward on
          the identical batch -> grad_norm=157.0, finite). fp32 also avoids it
          (grad_norm=140.2, finite) but costs far more VRAM. Llama/Gemma are
          unaffected and keep the default (faster) SDPA backend.
    """
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,  # Required for Qwen
        token=hf_token,
    )
    if "qwen" in model_id.lower():
        kwargs["attn_implementation"] = "eager"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    # Ensure tokenizer has pad token (required for attention mask computation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Patch: transformers 5.x removed get_head_mask from PreTrainedModel.
    # Qwen-1 (Qwen-7B-Chat, Qwen-14B-Chat) calls self.get_head_mask() unconditionally
    # in QWenModel.forward(). We always pass head_mask=None, so [None]*n is correct.
    if not hasattr(model, "get_head_mask"):
        model.get_head_mask = lambda head_mask, num_hidden_layers, **_: [None] * num_hidden_layers
    # Also patch the inner QWenModel if present (it's called on self.transformer, not self)
    inner = getattr(model, "transformer", None)
    if inner is not None and not hasattr(inner, "get_head_mask"):
        inner.get_head_mask = lambda head_mask, num_hidden_layers, **_: [None] * num_hidden_layers

    return model, tokenizer


def get_num_layers(model) -> int:
    """
    Detect the number of transformer layers in a model.

    Works for:
        - Llama-2/3/3.1: model.model.layers
        - Gemma: model.model.layers
        - Qwen2.x: model.model.layers
        - Qwen-1: model.transformer.h

    Returns:
        Number of transformer layers (+ 1 for embedding layer in hidden_states)
    """
    try:
        # Works for Llama, Gemma, Qwen2.x
        return len(model.model.layers)
    except AttributeError:
        try:
            # Works for Qwen-1
            return len(model.transformer.h)
        except AttributeError:
            raise RuntimeError(f"Cannot detect layer count for {type(model).__name__}")


def get_result_path(results_dir: str, model_id: str) -> str:
    """
    Sanitise model_id to a filesystem-safe result file path.

    Args:
        results_dir: Directory to save results (e.g. 'results/')
        model_id: HuggingFace model ID (e.g. 'Qwen/Qwen-7B-Chat')

    Returns:
        Path like: results/Qwen__Qwen-7B-Chat.json
    """
    safe_name = model_id.replace("/", "__")
    return os.path.join(results_dir, f"{safe_name}.json")


def free_model(model, tokenizer):
    """
    Free GPU memory occupied by model and tokenizer.

    Performs:
        1. Delete model and tokenizer objects
        2. Empty CUDA cache
        3. Run garbage collection

    Should be called after all inference with a model is complete,
    before loading the next model (sequential processing).
    """
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
