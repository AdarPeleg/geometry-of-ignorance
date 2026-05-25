# Attack Methods

Implementation in `src/attacks.py`. Four attacks probe unlearning robustness at different levels.

---

## Activation Steering

**Source:** Seyitoğlu et al. (2024). *Extracting Unlearned Information from LLMs with Activation Steering.* arXiv:2411.02631.

A steering vector is computed as the difference between hidden states of a concept-specific query and its anonymized paraphrase at the penultimate transformer layer; this vector is injected into the residual stream during generation to attempt re-activation of suppressed knowledge. Success is measured as ROC AUC over steered vs. unsteered word-frequency scores (chance = 0.5).

---

## In-Context Learning (ICL)

**Source:** Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS. arXiv:2005.14165.

Five demonstration Q&A pairs about the target concept are prepended to each evaluation question, testing whether contextual priming can recover facts the model no longer produces in isolation. Success is measured as exact-match accuracy (ROUGE-L ≥ 0.30) across 25 evaluation questions.

---

## Greedy Coordinate Gradient (GCG)

**Source:** Zou et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043. Implementation via the [`nanogcg`](https://github.com/GraySwanAI/nanoGCG) library.

A 20-token adversarial suffix is optimized over 500 gradient steps to minimize cross-entropy loss toward the correct answer prefix, then appended to each question at inference time. Success is measured as the fraction of 5 questions where the resulting generation contains a correct answer keyword or achieves ROUGE-L ≥ 0.25.

---

## Membership Inference Attack (MIA)

**Source:** Shokri et al. (2017). *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P. Likelihood-ratio variant following Carlini et al. (2022). *Membership Inference Attacks From First Principles.* arXiv:2112.03570.

The conditional negative log-likelihood of each answer given its question is computed via teacher-forced cross-entropy; a model that retains memory of the forget set assigns systematically lower NLL to forget pairs than to retain pairs. Success is measured as ROC AUC separating 50 forget pairs from 50 retain pairs (chance = 0.5, higher = more membership signal retained).
