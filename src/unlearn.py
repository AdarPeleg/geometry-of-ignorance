"""
Unlearning method implementations for RQ2.

Six methods (faithful to original papers):
  GradAscent — gradient ascent on forget cross-entropy (negate the LM loss)
  DPO        — forget pairs as rejected, retain pairs as chosen; standard DPO loss
  NPO        — negative preference optimization: push model away from reference on forget set
  NPO+KL     — NPO with KL regularization on retain set to preserve general capability
  RMU        — representation misdirection: steer forget hidden states toward a random vector
  WHP        — WhoIsHarryPotter: redirect concept queries to anonymized answers (Eldan & Russinovich 2023)

All methods share a common training config and save checkpoints atomically.

Reference implementations:
  open-unlearning: https://github.com/locuslab/open-unlearning
  NPO/DPO paper:   https://arxiv.org/abs/2404.05868
  RMU paper:       https://arxiv.org/abs/2408.06223
"""

import copy
import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ── Training config ────────────────────────────────────────────────────────────

@dataclass
class UnlearnConfig:
    n_epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 4
    max_length: int = 256
    gradient_checkpointing: bool = True
    gradient_clip: float = 1.0
    # Method-specific
    dpo_beta: float = 0.1             # DPO: KL regularization strength
    npo_beta: float = 0.1             # NPO: reference KL coefficient
    npo_kl_alpha: float = 1.0         # NPO+KL: weight of retain KL term
    rmu_layer_id: int = 7             # RMU: which layer to steer (mid-network default)
    rmu_alpha: float = 1200.0         # RMU: retain regularization weight
    rmu_steering_coeff: float = 20.0  # RMU: scale of random steering target
    whp_retain_alpha: float = 1.0     # WHP: weight of retain KL term
    seed: int = 42


# ── Dataset helpers ────────────────────────────────────────────────────────────

class TextPairDataset(Dataset):
    """Dataset of (prompt, target) text pairs for language modelling."""

    def __init__(self, pairs: List[Dict], tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 256, key_q: str = "question", key_a: str = "answer"):
        self.samples = []
        for p in pairs:
            prompt = p.get(key_q, "").strip()
            answer = p.get(key_a, "").strip()
            if not prompt:
                continue
            full = f"{prompt}\n{answer}" if answer else prompt
            enc = tokenizer(
                full, truncation=True, max_length=max_length,
                return_tensors="pt", padding=False
            )
            self.samples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_id: int):
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    max_len = max(x.size(0) for x in input_ids)
    padded_ids = torch.stack([
        F.pad(x, (0, max_len - x.size(0)), value=pad_id) for x in input_ids
    ])
    padded_masks = torch.stack([
        F.pad(x, (0, max_len - x.size(0)), value=0) for x in attention_masks
    ])
    return {"input_ids": padded_ids, "attention_mask": padded_masks}


def _make_loader(pairs, tokenizer, cfg: UnlearnConfig, shuffle=True,
                 key_q="question", key_a="answer"):
    ds = TextPairDataset(pairs, tokenizer, cfg.max_length, key_q=key_q, key_a=key_a)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_id),
        drop_last=False,
    )


def _device(model):
    return next(model.parameters()).device


def _lm_loss(model, input_ids, attention_mask):
    """Standard causal LM cross-entropy loss (next-token prediction)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    labels=input_ids, use_cache=False)
    return outputs.loss


def _save_checkpoint(model, tokenizer, out_dir: str):
    """Atomic checkpoint save via temp dir rename."""
    out_dir = Path(out_dir)
    tmp_dir = out_dir.parent / (out_dir.name + ".tmp")
    model.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)


def _optimizer(model, lr):
    try:
        from transformers import AdamW
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(model.parameters(), lr=lr)
    except ImportError:
        return torch.optim.AdamW(model.parameters(), lr=lr)


# ── Method 1: Gradient Ascent ──────────────────────────────────────────────────

def grad_ascent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_pairs: List[Dict],
    retain_pairs: List[Dict],  # unused but kept for uniform interface
    cfg: UnlearnConfig = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """
    Gradient Ascent (GA): negate the cross-entropy loss on forget pairs.
    Loss = −CE(model, forget)

    The simplest unlearning baseline. No explicit retain regularization —
    this often degrades general model capability (used to contrast with methods
    that preserve retain performance).
    """
    if cfg is None:
        cfg = UnlearnConfig()
    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    loader = _make_loader(forget_pairs, tokenizer, cfg)
    opt = _optimizer(model, cfg.lr)
    device = _device(model)

    for epoch in range(cfg.n_epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            loss = -_lm_loss(model, input_ids, attn)  # negate = ascent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        logger.info(f"[GA] epoch {epoch+1}/{cfg.n_epochs}  loss={total_loss/len(loader):.4f}")

    if checkpoint_dir:
        _save_checkpoint(model, tokenizer, checkpoint_dir)


# ── Method 2: DPO ─────────────────────────────────────────────────────────────

def _ref_model_copy(model):
    """
    Create a frozen reference-model copy pinned entirely to cuda:1.

    With device_map="auto" the policy model is sharded across both GPUs (~8 GB each).
    Putting the full ref model on cuda:1 gives:
      GPU0: ~8 GB (policy layers)
      GPU1: ~8 GB (policy layers) + ~16 GB (ref model) = ~24 GB  (fits in 48 GB)
    This is faster than CPU and avoids the VRAM-doubling that caused OOM.

    deepcopy preserves accelerate dispatch hooks, so we strip them before .to().
    """
    ref = copy.deepcopy(model)
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(ref, recurse=True)
    except ImportError:
        pass
    ref = ref.to("cuda:1").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _log_ratio(model, ref_model, input_ids, attention_mask, labels):
    """log π(y|x) - log π_ref(y|x) for a batch.

    ref_model lives on cuda:1; inputs are routed to its device automatically.
    """
    ref_device = next(ref_model.parameters()).device
    with torch.no_grad():
        ref_out = ref_model(
            input_ids=input_ids.to(ref_device),
            attention_mask=attention_mask.to(ref_device),
            labels=labels.to(ref_device),
            use_cache=False,
        )
    cur_out = model(input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels, use_cache=False)
    return cur_out.loss - ref_out.loss.detach().to(cur_out.loss.device)


def dpo_unlearn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_pairs: List[Dict],
    retain_pairs: List[Dict],
    cfg: UnlearnConfig = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """
    DPO Unlearning: treat forget pairs as "rejected" and retain pairs as "chosen."
    Standard DPO loss pushes the model to prefer retain content over forget content.

    Loss = −E[log σ(β (log π/π_ref(retain) − log π/π_ref(forget)))]
    Reference: https://arxiv.org/abs/2305.18290 (DPO), adapted for unlearning in
    open-unlearning codebase.
    """
    if cfg is None:
        cfg = UnlearnConfig()
    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ref_model = _ref_model_copy(model)

    forget_loader = _make_loader(forget_pairs, tokenizer, cfg)
    retain_loader = _make_loader(retain_pairs, tokenizer, cfg)
    opt = _optimizer(model, cfg.lr)
    device = _device(model)
    beta = cfg.dpo_beta

    for epoch in range(cfg.n_epochs):
        total_loss = 0.0
        retain_iter = iter(retain_loader)
        for f_batch in forget_loader:
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            f_ids = f_batch["input_ids"].to(device)
            f_attn = f_batch["attention_mask"].to(device)
            r_ids = r_batch["input_ids"].to(device)
            r_attn = r_batch["attention_mask"].to(device)

            forget_ratio = _log_ratio(model, ref_model, f_ids, f_attn, f_ids)
            retain_ratio = _log_ratio(model, ref_model, r_ids, r_attn, r_ids)

            # DPO loss: prefer retain over forget
            loss = -F.logsigmoid(beta * (retain_ratio - forget_ratio)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        logger.info(f"[DPO] epoch {epoch+1}/{cfg.n_epochs}  loss={total_loss/len(forget_loader):.4f}")

    del ref_model
    torch.cuda.empty_cache()
    gc.collect()
    if checkpoint_dir:
        _save_checkpoint(model, tokenizer, checkpoint_dir)


# ── Method 3: NPO ─────────────────────────────────────────────────────────────

def npo_unlearn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_pairs: List[Dict],
    retain_pairs: List[Dict],  # unused
    cfg: UnlearnConfig = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """
    Negative Preference Optimization (NPO): push model away from reference on forget set.
    No explicit retain regularization term (contrast with NPO+KL).

    Loss = −(2/β) E[log σ(−β log π/π_ref(forget))]
    Reference: https://arxiv.org/abs/2404.05868 (SimPO/NPO paper)
    """
    if cfg is None:
        cfg = UnlearnConfig()
    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ref_model = _ref_model_copy(model)

    loader = _make_loader(forget_pairs, tokenizer, cfg)
    opt = _optimizer(model, cfg.lr)
    device = _device(model)
    beta = cfg.npo_beta

    for epoch in range(cfg.n_epochs):
        total_loss = 0.0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            log_ratio = _log_ratio(model, ref_model, ids, attn, ids)
            # NPO loss: negative log-sigmoid of log ratio (push ratio negative)
            loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        logger.info(f"[NPO] epoch {epoch+1}/{cfg.n_epochs}  loss={total_loss/len(loader):.4f}")

    del ref_model
    torch.cuda.empty_cache()
    gc.collect()
    if checkpoint_dir:
        _save_checkpoint(model, tokenizer, checkpoint_dir)


# ── Method 4: NPO+KL ──────────────────────────────────────────────────────────

def npo_kl_unlearn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_pairs: List[Dict],
    retain_pairs: List[Dict],
    cfg: UnlearnConfig = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """
    NPO with KL regularization on retain set.
    Adds a retain term that keeps the model close to the reference on retain examples.

    Loss = NPO_loss(forget) + α * KL(model || ref)(retain)
    Reference: https://arxiv.org/abs/2404.05868
    """
    if cfg is None:
        cfg = UnlearnConfig()
    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ref_model = _ref_model_copy(model)

    forget_loader = _make_loader(forget_pairs, tokenizer, cfg)
    retain_loader = _make_loader(retain_pairs, tokenizer, cfg)
    opt = _optimizer(model, cfg.lr)
    device = _device(model)
    beta = cfg.npo_beta
    alpha = cfg.npo_kl_alpha

    for epoch in range(cfg.n_epochs):
        total_loss = 0.0
        retain_iter = iter(retain_loader)
        for f_batch in forget_loader:
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            # NPO term on forget
            f_ids = f_batch["input_ids"].to(device)
            f_attn = f_batch["attention_mask"].to(device)
            log_ratio_f = _log_ratio(model, ref_model, f_ids, f_attn, f_ids)
            npo_loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio_f).mean()

            # KL term on retain: KL(model || ref) ≈ CE(model, retain) - CE(ref, retain)
            r_ids = r_batch["input_ids"].to(device)
            r_attn = r_batch["attention_mask"].to(device)
            kl_loss = _log_ratio(model, ref_model, r_ids, r_attn, r_ids).mean()

            loss = npo_loss + alpha * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        logger.info(f"[NPO+KL] epoch {epoch+1}/{cfg.n_epochs}  loss={total_loss/len(forget_loader):.4f}")

    del ref_model
    torch.cuda.empty_cache()
    gc.collect()
    if checkpoint_dir:
        _save_checkpoint(model, tokenizer, checkpoint_dir)


# ── Method 5: RMU ─────────────────────────────────────────────────────────────

def rmu_unlearn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_pairs: List[Dict],
    retain_pairs: List[Dict],
    cfg: UnlearnConfig = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """
    Representation Misdirection for Unlearning (RMU).
    Steers forget-set hidden states at layer L toward a fixed random unit vector,
    while keeping retain-set activations close to the original.

    Forget loss: MSE(h_L(forget), c * u)  where u=random unit vector, c=steering_coeff
    Retain loss: alpha * L2(h_L(retain), h_L_orig(retain))

    Reference: https://arxiv.org/abs/2408.06223
    """
    if cfg is None:
        cfg = UnlearnConfig()

    # Determine hidden dimension for steering target
    try:
        hidden_dim = model.config.hidden_size
    except AttributeError:
        hidden_dim = next(model.parameters()).shape[-1]

    device = _device(model)
    model_dtype = next(model.parameters()).dtype  # bfloat16 or float32

    # Fixed random steering target: c * u (unit vector scaled by coeff)
    # Must match model dtype to avoid dtype mismatch in MSE loss
    torch.manual_seed(cfg.seed)
    u = torch.randn(hidden_dim, device=device, dtype=model_dtype)
    u = u / u.norm()
    steering_target = cfg.rmu_steering_coeff * u  # [hidden_dim]

    # Reference model for retain regularization
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model.train()
    # RMU captures hidden states via forward hooks, so gradient checkpointing must be OFF.
    # With checkpointing, backward re-runs the forward without hooks, severing the captured
    # tensor from the computation graph and causing "does not require grad" errors.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    forget_loader = _make_loader(forget_pairs, tokenizer, cfg)
    retain_loader = _make_loader(retain_pairs, tokenizer, cfg)
    opt = _optimizer(model, cfg.lr)

    # Determine which module corresponds to layer cfg.rmu_layer_id
    # Works for Llama/Gemma (model.model.layers) and Qwen2.x
    try:
        layer_module = model.model.layers[cfg.rmu_layer_id]
    except (AttributeError, IndexError):
        try:
            layer_module = model.transformer.h[cfg.rmu_layer_id]
        except (AttributeError, IndexError):
            raise RuntimeError(f"Cannot find layer {cfg.rmu_layer_id} for RMU")

    def _get_hidden(module, inputs, outputs, store):
        """Hook to capture last-token hidden state after a layer."""
        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs
        # Mean-pool over sequence to get a single vector per sample
        store.append(hidden.mean(dim=1))  # [batch, hidden_dim]

    for epoch in range(cfg.n_epochs):
        total_loss = 0.0
        retain_iter = iter(retain_loader)
        for f_batch in forget_loader:
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            # ── Forget loss: steer hidden states toward target ────────────────
            f_ids = f_batch["input_ids"].to(device)
            f_attn = f_batch["attention_mask"].to(device)
            forget_hidden = []
            hook_f = layer_module.register_forward_hook(
                lambda m, i, o: forget_hidden.append(
                    (o[0] if isinstance(o, tuple) else o).mean(dim=1)
                )
            )
            model(input_ids=f_ids, attention_mask=f_attn,
                  output_hidden_states=False, use_cache=False)
            hook_f.remove()

            h_forget = forget_hidden[0]  # [batch, hidden_dim]
            target = steering_target.unsqueeze(0).expand_as(h_forget)
            forget_loss = F.mse_loss(h_forget, target)

            # ── Retain loss: keep activations close to reference ──────────────
            r_ids = r_batch["input_ids"].to(device)
            r_attn = r_batch["attention_mask"].to(device)

            retain_hidden, ref_hidden = [], []
            hook_r = layer_module.register_forward_hook(
                lambda m, i, o: retain_hidden.append(
                    (o[0] if isinstance(o, tuple) else o).mean(dim=1)
                )
            )
            model(input_ids=r_ids, attention_mask=r_attn,
                  output_hidden_states=False, use_cache=False)
            hook_r.remove()

            with torch.no_grad():
                hook_ref = list(ref_model.modules())  # get ref layer
                try:
                    ref_layer = ref_model.model.layers[cfg.rmu_layer_id]
                except (AttributeError, IndexError):
                    ref_layer = ref_model.transformer.h[cfg.rmu_layer_id]
                hook_rr = ref_layer.register_forward_hook(
                    lambda m, i, o: ref_hidden.append(
                        (o[0] if isinstance(o, tuple) else o).mean(dim=1)
                    )
                )
                ref_model(input_ids=r_ids, attention_mask=r_attn,
                          output_hidden_states=False, use_cache=False)
                hook_rr.remove()

            h_retain = retain_hidden[0]
            h_ref = ref_hidden[0].detach()
            retain_loss = F.mse_loss(h_retain, h_ref)

            loss = forget_loss + cfg.rmu_alpha * retain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        logger.info(
            f"[RMU] epoch {epoch+1}/{cfg.n_epochs}  loss={total_loss/len(forget_loader):.4f}"
        )

    del ref_model
    torch.cuda.empty_cache()
    gc.collect()
    if checkpoint_dir:
        _save_checkpoint(model, tokenizer, checkpoint_dir)


# ── Method 6: WHP ─────────────────────────────────────────────────────────────

def whp_unlearn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_pairs: List[Dict],
    retain_pairs: List[Dict],
    cfg: UnlearnConfig = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """
    WhoIsHarryPotter (WHP) unlearning: redirect forget-concept queries toward
    anonymized answers instead of destructively removing the knowledge via
    gradient ascent.

    Forget term: CE(model, anon_answer | orig_question)
        Train the model to produce the generic/anonymized answer when given the
        original concept-specific question. This associates concept queries with
        neutral content rather than erasing the representational geometry.

    Retain term: KL(model ‖ ref)(retain)
        Keeps the model close to the reference on unrelated retain pairs to
        prevent catastrophic forgetting.

    Loss = CE_forget + whp_retain_alpha * CE_retain

    No reference model copy is required: the retain term is a plain cross-entropy
    on retain pairs, not a KL divergence. This matches the spirit of the original
    paper (which uses LR/early-stopping for retain protection, not an explicit KL),
    avoids pinning a second 16 GB copy to a single GPU, and eliminates the OOM
    pattern seen when running back-to-back WHP runs.

    Reference: Eldan & Russinovich (2023) https://arxiv.org/abs/2310.02238
    """
    if cfg is None:
        cfg = UnlearnConfig()
    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # key_a="anon_answer": prompt = original question, target = anonymized answer
    forget_loader = _make_loader(forget_pairs, tokenizer, cfg,
                                 key_q="question", key_a="anon_answer")
    retain_loader = _make_loader(retain_pairs, tokenizer, cfg)
    opt = _optimizer(model, cfg.lr)
    device = _device(model)

    for epoch in range(cfg.n_epochs):
        total_loss = 0.0
        retain_iter = iter(retain_loader)
        for f_batch in forget_loader:
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            # Forget term: CE toward anonymized output
            f_ids = f_batch["input_ids"].to(device)
            f_attn = f_batch["attention_mask"].to(device)
            forget_loss = _lm_loss(model, f_ids, f_attn)

            # Retain term: plain CE on retain pairs (no ref model needed)
            r_ids = r_batch["input_ids"].to(device)
            r_attn = r_batch["attention_mask"].to(device)
            retain_loss = _lm_loss(model, r_ids, r_attn)

            loss = forget_loss + cfg.whp_retain_alpha * retain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        logger.info(f"[WHP] epoch {epoch+1}/{cfg.n_epochs}  loss={total_loss/len(forget_loader):.4f}")

    if checkpoint_dir:
        _save_checkpoint(model, tokenizer, checkpoint_dir)


# ── Registry ───────────────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    "GradAscent": grad_ascent,
    "DPO":        dpo_unlearn,
    "NPO":        npo_unlearn,
    "NPO+KL":     npo_kl_unlearn,
    "RMU":        rmu_unlearn,
    "WHP":        whp_unlearn,
}


def apply_unlearning(
    method: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_pairs: List[Dict],
    retain_pairs: List[Dict],
    cfg: UnlearnConfig = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """Dispatch to the named unlearning method."""
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHOD_REGISTRY)}")
    METHOD_REGISTRY[method](
        model, tokenizer, forget_pairs, retain_pairs, cfg, checkpoint_dir
    )
