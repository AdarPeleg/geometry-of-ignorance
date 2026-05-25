"""
Steering vector computation from paired (regular, anonymised) text inputs.

Pipeline:
  1. embed_text_all_layers: Single forward pass → all-layer vectors
  2. compute_vectors_all_layers: Extract aligned pairs (regular, anon) per layer
  3. compute_per_batch_directions: Split into batches, compute per-batch direction vectors

All functions operate on the full model (inference only, no gradients).
Hidden states are moved to CPU immediately to avoid VRAM accumulation on multi-GPU setups.
"""

import torch
from tqdm import tqdm
from typing import List, Tuple, Optional


@torch.no_grad()
def embed_text_all_layers(model, tokenizer, text: str, pooling: str = "mean") -> List[torch.Tensor]:
    """
    Extract embedding vectors from all transformer layers for a single text.

    Performs a single forward pass with output_hidden_states=True,
    then applies pooling to each hidden state.

    Args:
        model: Loaded transformer model
        tokenizer: Corresponding tokenizer
        text: Input text string
        pooling: Pooling strategy - "mean" (default, mask-aware), "last_token"

    Returns:
        List of Tensors, one per layer:
          Index 0: post-embedding layer (before any transformer block)
          Indices 1..L: after transformer blocks 1..L
        Each tensor is shape [hidden_dim], dtype float32, on CPU.

    Notes:
        - Max sequence length: 512 (hardcoded truncation)
        - Moves each vector to CPU immediately to free GPU VRAM
        - Attention mask is computed to handle padding correctly
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )

    # Move inputs to model's primary device
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    # Forward pass with all hidden states
    out = model(**enc, output_hidden_states=True)
    # out.hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]

    vecs = []
    attn_mask = enc.get("attention_mask", None)  # [1, seq_len]

    for h in out.hidden_states:
        h = h.to(torch.float32)  # Convert to float32 from bfloat16

        if pooling == "mean":
            if attn_mask is None:
                # No masking: simple mean across sequence
                vec = h.mean(dim=1).squeeze(0)
            else:
                # Mask-aware mean: (h * mask).sum / mask.sum
                m = attn_mask.unsqueeze(-1).to(torch.float32)
                denom = m.sum(dim=1).clamp_min(1.0)
                vec = (h * m).sum(dim=1) / denom
                vec = vec.squeeze(0)

        elif pooling == "last_token":
            if attn_mask is None:
                vec = h[:, -1, :].squeeze(0)
            else:
                # Get the last non-padding token
                lengths = attn_mask.sum(dim=1)
                idx = int((lengths - 1).clamp_min(0).item())
                vec = h[:, idx, :].squeeze(0)

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        # Move to CPU immediately to avoid multi-GPU VRAM accumulation
        vecs.append(vec.cpu())

    return vecs


def compute_vectors_all_layers(
    model, tokenizer, pairs: List[dict], pooling: str = "mean"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extract aligned pair vectors (regular, anonymised) for all layers.

    Formats: "{question}\nAnswer: {answer}" for both regular and anon texts.

    Args:
        model: Loaded model
        tokenizer: Corresponding tokenizer
        pairs: List of dicts with keys:
            - question, answer, anon_question, anon_answer
        pooling: Pooling strategy (passed to embed_text_all_layers)

    Returns:
        (reg_by_layer, anon_by_layer): Each is a List[Tensor[N, hidden_dim]]
          where N = number of pairs
    """
    N = len(pairs)
    reg_all = []  # reg_all[i] = list of L+1 vecs for pair i
    anon_all = []

    for pair in tqdm(pairs, desc="Embedding pairs"):
        reg_text = f"{pair['question']}\nAnswer: {pair['answer']}"
        anon_text = f"{pair['anon_question']}\nAnswer: {pair['anon_answer']}"

        reg_vecs = embed_text_all_layers(model, tokenizer, reg_text, pooling=pooling)
        anon_vecs = embed_text_all_layers(model, tokenizer, anon_text, pooling=pooling)

        reg_all.append(reg_vecs)
        anon_all.append(anon_vecs)

    # Restack by layer: for each layer ℓ, create [N, d] tensor
    num_layers = len(reg_all[0])
    reg_by_layer = []
    anon_by_layer = []

    for layer_idx in range(num_layers):
        reg_layer = torch.stack([reg_all[i][layer_idx] for i in range(N)])  # [N, d]
        anon_layer = torch.stack([anon_all[i][layer_idx] for i in range(N)])  # [N, d]
        reg_by_layer.append(reg_layer)
        anon_by_layer.append(anon_layer)

    return reg_by_layer, anon_by_layer


def compute_per_batch_directions(
    reg_vecs: torch.Tensor,
    anon_vecs: torch.Tensor,
    n_batches: int = 10,
    seed: int = 42,
) -> Optional[torch.Tensor]:
    """
    Compute per-batch steering direction vectors from paired embeddings.

    For each batch:
      direction = mean(reg_vecs_batch - anon_vecs_batch)
      direction = direction / ||direction||  (unit-normalise)

    Args:
        reg_vecs: [N, d] tensor of regular embeddings
        anon_vecs: [N, d] tensor of anonymised embeddings
        n_batches: Number of batches to split N pairs into
        seed: Random seed for deterministic shuffling

    Returns:
        Tensor[B, d] of unit-normalised batch direction vectors, or None if all degenerate.
        B <= n_batches (degenerate batches are skipped).

    Notes:
        - Seeded random shuffle for reproducibility across runs
        - Degenerate batch: norm < 1e-8 or contains NaN
    """
    N = reg_vecs.shape[0]
    reg_f = reg_vecs.float()
    anon_f = anon_vecs.float()

    # Seeded deterministic shuffle
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(N, generator=g)
    batch_indices = torch.chunk(indices, n_batches)

    batch_dirs = []
    for idx_tensor in batch_indices:
        if idx_tensor.numel() == 0:
            continue

        # Compute batch difference: mean(reg - anon)
        diff = (reg_f[idx_tensor] - anon_f[idx_tensor]).mean(dim=0)
        norm = diff.norm()

        # Skip degenerate batches
        if norm < 1e-8 or not torch.isfinite(norm):
            continue

        # Unit-normalise
        batch_dirs.append(diff / norm)

    if not batch_dirs:
        return None  # Signal: all batches were degenerate

    return torch.stack(batch_dirs)  # [B, d]
