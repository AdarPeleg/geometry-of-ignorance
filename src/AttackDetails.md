# RQ2 Attack Methods — Full Technical Reference

> This document covers all four attacks used in the RQ2 pipeline: how each works from
> first principles, every implementation detail in this codebase, expected results, and
> critical design decisions. See `src/attacks.py` for the code.

---

## Table of Contents
1. [Attack 1 — AnonAct Activation Steering](#1-anonact-activation-steering)
2. [Attack 2 — ICL (In-Context Learning)](#2-icl-in-context-learning)
3. [Attack 3 — GCG (Greedy Coordinate Gradient)](#3-gcg-greedy-coordinate-gradient)
4. [Attack 4 — MIA (Membership Inference Attack)](#4-mia-membership-inference-attack)
5. [Where anon_questions come from](#5-where-anon_questions-come-from)
6. [Comparison with Seyitoğlu et al. and MUSE](#6-comparison-with-seyitoglu-et-al-and-muse)
7. [Model coverage and whether we need more models](#7-model-coverage-and-whether-we-need-more-models)
8. [Summary table](#8-summary-table)

---

## 1. AnonAct Activation Steering

**Paper:** Seyitoğlu, Kuvshinov, Schwinn & Günnemann (2024). *Extracting Unlearned
Information from LLMs with Activation Steering.* arXiv:2411.02631.

**Metric:** ROC AUC (0.5 = chance, 1.0 = perfect recovery)

### Core intuition
If a model truly forgot a concept, its internal representation of a question
about that concept should look identical to a question about a blank/anonymous
stand-in. The attack exploits any residual geometric difference: it computes a
*steering vector* from (real question) − (anonymized question), then injects it
into the model's residual stream to try to re-activate suppressed knowledge.

### Step-by-step algorithm
1. **Choose layer:** `layer_id = n_layers − 2` (second-to-last transformer block,
   per the paper's "just before final layer" prescription).

2. **For each evaluation question Q** (e.g. "Who is the author of Harry Potter?"):
   - Collect N=5 anonymized variants Q* (e.g. "Who is the author of the Alex Ryden
     series?"). These come from `anon_questions` field — see Section 5.
   - Extract the hidden state at the **last token position** of the prompt
     `"Q\nAnswer:"` at layer L.  Using the last token (the `":"`) rather than
     position 0 captures the full attended context of the question.
   - Compute the **query-specific steering vector**:
     ```
     SV(Q) = h_L("Q\nAnswer:") − mean(h_L("Q*_i\nAnswer:"))
     ```
   - **Critically, SV is NOT unit-normalized.** The paper uses raw difference
     vectors with coefficient 2.0. Normalizing to unit length then using coeff=2
     would produce a ~0.05% perturbation in 4096-dim space — effectively noise.
     Raw scale means the perturbation is proportional to how strongly the model
     distinguishes the real concept from the anonymized one.

3. **Generate unsteered samples:** Run 100 stochastic completions with
   temperature=2.0, top-k=40, max_new_tokens=10 from prompt `"Q\nAnswer:"`.

4. **Generate steered samples:** Register a forward hook at layer L that adds
   `2.0 × SV` to the last prompt token's hidden state during the prefill pass
   (`seq_len > 1` only — does not modify decode steps). Run the same 100-sample
   generation.

5. **Score each run** with the **word frequency score** (Seyitoğlu et al.):
   for each distinctive answer word (length > 2, not a common stopword), count
   what fraction of the 100 samples contain it. Take the max across answer words.
   E.g. answer "J.K. Rowling": count fraction of samples containing "Rowling" →
   score = 0.30 means 30% of samples mention the correct author.

6. **Compute ROC AUC** over all 25 question pairs:
   - Labels: `[1]*25 + [0]*25` (steered = positive class, unsteered = negative)
   - Scores: `steered_word_freq + unsteered_word_freq`
   - AUC > 0.5: steering elevates the correct answer frequency → geometry still
     encodes the concept → unlearning failed geometrically
   - AUC ≈ 0.5: steering has no effect → representations are fragmented

### Our results and why Steering is near-chance
All methods produce **Steering AUC ≈ 0.50** (range: 0.12–0.58). Seyitoğlu et al.
also work on pretrained (not finetuned) models, yet report 0.70–0.95 AUC. The gap
most likely comes from the unlearning method, not the HP knowledge source:

- **Their target (WHP model)** was unlearned with Eldan & Russinovich's method, which
  uses "generic" paraphrase training. It suppresses generation without strongly
  distorting the internal representation geometry — leaving a steerable concept
  direction intact at layer n−2.
- **Our targets** are unlearned with GradAscent, DPO, NPO, NPO+KL, and RMU. These
  methods more aggressively perturb the residual stream (especially RMU, which
  explicitly randomizes representations). This may fragment the concept direction
  that steering exploits — meaning our low Steering AUC could indicate *more*
  effective geometric unlearning, not a failure of the attack setup.
- A secondary factor: their 62 questions were curated by GPT-4 specifically for HP
  steering; our ConceptVectors-sourced question set may elicit a weaker directional
  signal at layer n−2.

**Implication:** Steering cannot distinguish methods in our setting. This is included
for completeness. MIA and ICL are the primary signals.

---

## 2. ICL (In-Context Learning)

**Paper:** Standard practice; no single canonical paper.

**Metric:** Accuracy — fraction of questions answered correctly (ROUGE-L ≥ 0.3)

### Core intuition
If the model has "forgotten" a fact generatively (it won't say "J.K. Rowling" when
asked directly), can it be reminded by seeing a few correct examples first? ICL
exploits the model's pattern-matching ability over in-context demonstrations. Many
unlearning methods suppress direct recall but leave the model's ability to
pattern-match from examples intact.

### Step-by-step algorithm
1. Take the concept's `qa_eval_pairs` (25 proper Q&A pairs like
   "Who wrote Harry Potter? → J.K. Rowling"). Split into 25 evaluation questions
   and a demo pool (remaining pairs, or the same pool if there are too few).

2. For each evaluation question Q:
   - **Baseline:** Generate greedily from `"Q\nAnswer:"` (no context).
     Match: ROUGE-L ≥ 0.3 between generated text and expected answer.
   - **ICL:** Sample 5 random demos from the pool (excluding Q itself), prepend:
     ```
     Q1\nAnswer: A1\n\nQ2\nAnswer: A2\n\n...\nQ\nAnswer:
     ```
     Generate greedily, apply same ROUGE-L ≥ 0.3 match.

3. **Score = ICL accuracy** (fraction of 25 pairs where ICL generation matches).
   Baseline accuracy is also recorded but the primary metric is raw ICL accuracy,
   not the delta — even if baseline is 0%, non-zero ICL accuracy means the
   knowledge is recoverable with context.

### Why ICL is the most informative attack
ICL has 6 significant correlations with metrics (BH-adj p < 0.05) — far more than
any other attack. The reason: methods that genuinely scramble parametric
representations (high AUSS_L2, high Reg S₂, high Batch_Cov_TopEig) also prevent
ICL, because the model cannot coherently use factual context when the underlying
representation is fragmented. Methods that only block greedy generation (like
GradAscent) still respond correctly to contextual hints.

### Key design decisions
- **Use `qa_eval_pairs` not `forget_pairs`:** The `forget_pairs` are Wikipedia
  sentence completions (not questions), which give near-zero ICL scores because the
  model doesn't know what format to answer in.
- **ROUGE-L ≥ 0.3 threshold:** Generous enough to credit partial matches (e.g.
  "Rowling" counts for "J.K. Rowling"), strict enough to avoid false positives.
- **Greedy decoding:** Removes sampling variance; results are deterministic.

---

## 3. GCG (Greedy Coordinate Gradient)

**Paper:** Zou, Wang, Carlini, Fredrikson, Kolter & Floridi (2023). *Universal and
Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.

**Metric:** Accuracy — fraction of questions where adversarial suffix elicits the
correct answer (keyword hit OR ROUGE-L ≥ 0.25)

### Core intuition
GCG optimizes a sequence of "junk" tokens appended to the question such that the
model is forced to output a specific target answer. The optimization is gradient-
based over the discrete token vocabulary — at each step, it finds the token
replacement that most reduces the loss toward the target. It was originally
developed for jailbreaking safety-finetuned models; here applied to check if
unlearning can be undone by adversarial prompting.

### Step-by-step algorithm
1. **Select questions:** Sort the concept's `qa_eval_pairs` by answer length (shortest
   first). Use the shortest-answer 60% as the candidate pool, then randomly sample
   5 questions. Short answers converge much faster in GCG.

2. **Define the target prefix:** Take the first 8 tokens of the correct answer
   (decoded back to a string). E.g. for answer "J.K. Rowling, the British author",
   target = "J.K. Rowling". Optimizing against 8 tokens converges far faster than
   the full answer (GCG is exponentially harder as target length grows).

3. **Initialize suffix:** 20 tokens, each initialized to `"!"`.

4. **Run 500 GCG steps** (via `nanogcg` package):
   - At each step, for each suffix position, compute the gradient of
     cross-entropy loss (model output vs target prefix) with respect to all
     vocabulary replacements at that position.
   - Select the replacement that gives the greatest loss reduction (greedy
     coordinate descent over the discrete token space).
   - Keep the best suffix found across all 500 steps (by lowest loss).

5. **Generate with optimized suffix:**
   Build the adversarial prompt: `Q + " " + best_suffix`, apply chat template,
   run greedy generation with max_new_tokens=256.

6. **Match:** Two criteria (either counts as success):
   - **Keyword hit:** Any answer word of length > 3 (not a stopword) appears in
     the generated text.
   - **ROUGE-L:** F1 ≥ 0.25 between generated text and expected answer.

7. **Score = ASR** (attack success rate = fraction of 5 questions that matched).

### Implementation details
- **500 steps, not 150:** Early experiments with 150 steps had very low convergence.
  500 gives GCG a fair chance on 7-8B models.
- **Runs last** in the pipeline (Steering → ICL → MIA → GCG): `nanogcg.run()` can
  corrupt the CUDA context on some GPU configurations. Running last avoids
  contaminating other attacks.
- **Chat template is applied automatically** by `nanogcg`, matching how the model
  was instruction-tuned.

### Our results
GCG scores are mostly 0.0–0.2, with occasional spikes (DPO on Harry Potter = 0.80).
No significant correlations with metrics after BH correction. Likely reasons:
- 500 steps may still be insufficient for 7-8B models with complex knowledge
- n=5 questions per run gives very high variance in the score
- The match criterion (keyword hit) is coarse

---

## 4. MIA (Membership Inference Attack)

**Paper:** Shokri et al. (2017) framework; standard likelihood-ratio variant.

**Metric:** ROC AUC (0.5 = chance, 1.0 = perfect membership inference)

### Core intuition
A model assigns *lower loss* (higher probability) to content it was trained on than
to content it was not. After unlearning, the forget set should be statistically
indistinguishable from the retain set — the model should assign similar loss to
both. MIA checks this by computing a likelihood ratio and measuring how well it
separates forget from retain pairs.

### Step-by-step algorithm
1. **Score 50 forget pairs:** For each (Q, A) pair, compute
   `-log P(A | Q)` per token using teacher-forced cross-entropy:
   - Tokenize `"Q\nAnswer: A"`
   - Set labels: mask the question tokens (`labels[:q_len] = -100`), keep only
     answer tokens in the loss
   - Run a forward pass; the model outputs `loss = mean CE over answer tokens`
   - This is the **conditional NLL**: lower = model assigns higher probability to
     the answer given the question = more likely to have memorized this pair.

2. **Score 50 retain pairs** identically (pairs about unrelated facts the model
   was NOT unlearned on).

3. **Compute ROC AUC:**
   - Labels: `[1]*50 + [0]*50` (forget = member, retain = non-member)
   - Scores: `[-NLL]*50 + [-NLL]*50` (negated so lower loss → higher score →
     predicts "member")
   - **AUC > 0.5:** forget pairs have systematically lower NLL than retain pairs
     → model still distinguishes forget from retain → unlearning failed
   - **AUC = 0.5:** indistinguishable → good unlearning
   - **AUC < 0.5:** rare; model assigns *higher* NLL to forget — over-unlearning
     (e.g. GradAscent with enough epochs)

### Why MIA is the most reliable signal
- No generation required — reads raw probabilities, not decoded text
- Unaffected by decoding strategy, prompt formatting, or sampling variance
- Directly measures the goal of unlearning: making the forget distribution
  look like the retain distribution at the probability level
- Our mean MIA AUC = 0.77 across verified runs — the highest of all four attacks,
  confirming that most unlearning methods do not fully close the NLL gap

### Known anomaly: GradAscent NLL = 200+
After 10 epochs of gradient ascent on Wikipedia-format `forget_pairs`, the model's
NLL on those pairs reaches 200+. This is physically impossible (max cross-entropy
for a 50k-vocab model is ~10.4 nats) — the model is numerically degenerate on
date-prefixed Wikipedia text.

**Fix applied:** We use `qa_eval_pairs` (proper Q&A format: "Who wrote...? → J.K.
Rowling") for MIA scoring instead of Wikipedia completions. QA pairs do not trigger
degeneracy because GradAscent was trained on Wikipedia-format text. The QA pairs
give stable, interpretable NLL values.

---

## 5. Where `anon_questions` Come From

**Short answer: we create them ourselves.** They are not from any dataset.

The file `data/generate_anon_qa.py` generates 5 anonymized variants of every
question in `qa_eval_pairs` using **pure string replacement** — no LLM needed.

### How it works
A pre-built replacement table maps concept-specific named entities to fictional
alternatives across 5 "versions":

```
"Harry Potter"  → version 0: "Alex Ryden"
                → version 1: "Marcus Cole"
                → version 2: "Torin Drake"
                → version 3: "Daven Ash"
                → version 4: "Kael Stone"

"Hogwarts"      → version 0: "Crystal Academy"
                → version 1: "Phoenix Institute"
                ...
```

Patterns are sorted longest-first to avoid partial matches (e.g. "Harry Potter's"
is replaced before "Harry Potter"). The resulting 5 strings are stored in the
`anon_questions` field of each `qa_eval_pairs` entry.

**Example:**
- Original: `"Who is the author of the Harry Potter series?"`
- Version 0: `"Who is the author of the Alex Ryden series?"`
- Version 1: `"Who is the author of the Marcus Cole series?"`

### Relationship to the paper
Seyitoğlu et al. 2024 describe anonymization as: "replace entity keywords with
random alternatives." Our implementation follows exactly this approach with 5
versions per question instead of 1, averaging their hidden states to get a more
stable mean anonymized representation.

**The anonymized questions are NOT from the ConceptVectors dataset.** ConceptVectors
provides `text_completion` pairs (Wikipedia-style) and `qa_pairs` (factual Q&A) but
does not include anonymized variants. We added the anonymization layer ourselves.

---

## 6. Comparison with Seyitoğlu et al. and MUSE

- They test on **three already-unlearned models**: (1) the WhoIsHarryPotter model
  (Eldan & Russinovich 2023) — Llama-2-Chat with HP knowledge unlearned from
  pretraining; (2) a TOFU model (Phi-1.5, fictitious authors); (3) ROME-edited GPT2-XL
- None of these involve finetuning on HP first. The HP knowledge comes entirely from
  pretraining. Their setup is therefore **the same as ours** in this respect.
- They attack already-unlearned models to show the knowledge can be re-extracted.

### True setup comparison

| | Seyitoğlu et al. 2024 | MUSE benchmark | **Our setup** |
|---|---|---|---|
| HP knowledge source | **Pretraining** (Llama-2-Chat base) | **Finetuning** on HP books | **Pretraining** |
| Starting model | Llama-2-Chat, Phi-1.5, GPT2-XL | Llama-3.1-8B finetuned on HP | Llama-2-7b-chat, Llama-3-8B, Qwen2.5-7B |
| Unlearning method tested | WHP (Eldan & Russinovich), TOFU, ROME | GradAscent, DPO, etc. | GradAscent, DPO, NPO, NPO+KL, RMU |
| Attack goal | Show unlearned info can be re-extracted | Measure behavioral retention | Correlate geometry with ASR |
| Steering AUC reported | 0.70–0.95 (WHP) | N/A | ~0.50 |

MUSE is the benchmark that finetunes first. Seyitoğlu et al. work on pretrained knowledge,
same as us and same as Geva et al.'s ConceptVectors.

### Why Seyitoğlu et al. get high Steering AUC and we don't

Since neither they nor we finetune on HP, the difference must come from elsewhere.
Most likely reasons:

1. **The WHP unlearning method preserves geometry better than ours.** Eldan &
   Russinovich's method works by training on paraphrased "generic" rewrites of HP
   content — it suppresses outputs without aggressively perturbing the representation
   geometry. GradAscent and RMU more aggressively distort the internal representations,
   which may fragment the concept direction that steering relies on.

2. **Llama-2-Chat's HP knowledge may be more concentrated.** The specific model
   (Llama-2-Chat) and their specific question set (62 GPT-4-generated HP questions)
   may elicit a stronger, more consistent HP direction at layer n−2 than our question
   set does.

3. **We use different unlearning methods.** Their attack targets WHP-unlearned models.
   Our models are unlearned with GradAscent/DPO/NPO/RMU, which may destroy the
   concept geometry more thoroughly — meaning our low Steering AUC might actually
   indicate more effective unlearning, not a failure of the attack.

4. **Their 62 HP questions were curated by GPT-4** specifically for this task and may
   cover more stereotypically "steerable" HP facts than our ConceptVectors-sourced
   question set.

### Is Harry Potter well-known enough for base LLMs to "know" it?

**Yes, but it depends on the model and the question format.** HP is covered
extensively in pretraining corpora (books, Wikipedia, forums, news). However:

- **Small models (Gemma-2B, Gemma-7B):** HP QA accuracy 4–9%. These models have
  seen HP text but don't reliably recall specific facts in QA format.
- **Medium models (Llama-2-7B, Llama-3-8B):** 10–30% QA accuracy. They know HP but
  answers often require explicit recall, which base models aren't optimized for.
- **Qwen2.5-7B:** ~28% QA accuracy — the best in our setup.

The implication is that **for base pretrained models, there IS HP knowledge to
unlearn**, but the knowledge is less concentrated than in finetuned models. This
makes our setup arguably *harder* and *more realistic* than the finetuned setting
(real-world unlearning requests concern pretraining knowledge, not deliberate
fine-tuning).

### MUSE benchmark comparison

MUSE (*Machine Unlearning Six-Way Evaluation*, Shi et al. 2024, `muse-bench/MUSE-Books`)
finetunes Llama-3.1-8B on the full Harry Potter book corpus and then evaluates 6
unlearning methods. Their results:

- **Pre-unlearning HP QA accuracy:** 70–90% (concentrated fine-tuned knowledge)
- **Post-unlearning HP QA accuracy (best methods):** 15–40% (still high residual)
- **Post-unlearning HP QA accuracy (GradAscent):** near 0%, but at severe retain cost

Our setup cannot be directly compared on QA accuracy because our baseline is 0–28%
(no finetuning). What we can compare:

| Signal | MUSE (finetuned) | Ours (pretrained) |
|---|---|---|
| HP QA baseline | 70–90% | 0–28% |
| MIA AUC (after best unlearning) | ~0.6–0.8 | ~0.5–1.0 |
| Retain degradation | Severe for GradAscent | Severe for GradAscent |
| Primary evaluation signal | Behavioral QA | Geometric AUSS metrics |

The MUSE paper's main finding — that behavioral QA metrics are insufficient and
models retain latent knowledge even when QA drops to 0% — directly motivates our
geometric approach.

---

## 7. Model Coverage and Whether We Need More Models

### Current RQ2 models (3 models × 5 methods × 3 concepts)

| Model | HuggingFace ID | Verified runs | Notes |
|---|---|---|---|
| Llama-2-7b-chat | `meta-llama/Llama-2-7b-chat-hf` | 9 | Gated; HP baseline ~13% |
| Meta-Llama-3-8B | `meta-llama/Meta-Llama-3-8B-Instruct` | 6 | Gated; HP baseline ~31% |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | 5 | Public; HP baseline ~28% |

### What ConceptVectors officially supports

The ConceptVectors dataset (`YihuaiHong/ConceptVectors`) was **built and validated
on only two base models**: Llama-2-7b and OLMo-7b. The concept vectors (parameter-
level traces used to verify concepts are encoded) were computed for these two models
specifically. The dataset has concept files only for these two:
- `llama2-7b_concepts_test.json`
- `olmo-7b_concepts_test.json`

We are using it with instruction-tuned variants (Llama-2-7b-chat, etc.) and
Qwen2.5-7B, which was not in the original validation. This is a limitation: we
cannot verify that the ConceptVectors parameter-level traces apply to these models.
We use the dataset only for its `qa_pairs` and `text_completion` pairs, not the
concept vectors themselves.

### Do we need more models?

**Yes, for statistical reasons.** With n=20 verified runs total:
- Power to detect ρ=0.5 at α=0.05 is ~70% — borderline.
- Power to detect ρ=0.4 (still a meaningful effect) is ~40% — poor.
- Many method-model combinations did not produce verified unlearning (DPO failed
  on Llama-3 and Qwen).

**Recommended additions (all on HuggingFace):**

| Model | HF ID | Rationale |
|---|---|---|
| OLMo-7B-Instruct | `allenai/OLMo-7B-Instruct` | In ConceptVectors; adds officially supported model |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | Public; strong baseline HP knowledge; no gating |
| Llama-3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | MUSE benchmark model; direct comparability |

Adding 2 models would bring n to ~32 verified runs (depending on method success
rates), raising power to detect ρ=0.4 to ~65% — adequate for a conference paper.

### Does fine-tuning on HP make more sense?

For comparability with Seyitoğlu et al. and MUSE, finetuning on HP before
unlearning would:
1. Give Steering AUC meaningful signal (currently near-chance)
2. Raise baseline QA to 70–90%, making behavioral unlearning cleaner to verify
3. Enable direct numeric comparison with published results

However, it changes the research question: we would be measuring "can we unlearn
deliberately injected fine-tuning?" rather than "can we unlearn pretraining
knowledge?" Our setup is arguably **more realistic** (copyright/privacy unlearning
concerns pretraining, not fine-tuning), but the finetuned setting gives stronger
and more comparable attack signals.
