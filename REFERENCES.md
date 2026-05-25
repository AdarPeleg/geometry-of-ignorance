# References

## Papers

### Activation Steering Attack
**Shi, Z., Bi, Z., Xiao, Y., et al. (2025).**
*Extracting Unlearned Information from LLMs with Activation Steering.*
arXiv:2411.02631. https://arxiv.org/abs/2411.02631

Basis for the `steering_attack` implementation in `src/attacks.py`:
- Per-question steering vector: `S_l(Q) = A_l(Q) − (1/N) Σ A_l(Q*_n)`
- Layer: `n_layers − 2` (just before final layer)
- Coefficient: 2.0 (raw unnormalized vectors)
- Evaluation: stochastic sampling (temp=2, top-k=40), word-frequency score
- Metric: ROC AUC of steered vs. unsteered word-frequency scores

---

### GCG — Greedy Coordinate Gradient
**Zou, A., Wang, Z., Carlini, N., et al. (2023).**
*Universal and Transferable Adversarial Attacks on Aligned Language Models.*
arXiv:2307.15043. https://arxiv.org/abs/2307.15043

Basis for the `gcg_attack` implementation, delegated to the `nanogcg` library.

---

## Datasets

### ConceptVectors
**Hong, Y., et al. (2024).**
*ConceptVectors: A Benchmark for Evaluating Concept Removal in LLMs.*
HuggingFace: `YihuaiHong/ConceptVectors`
https://huggingface.co/datasets/YihuaiHong/ConceptVectors

Used to provide forget QA pairs for Harry Potter, Star Wars, and William Shakespeare.
Dataset verifies that each concept is actually encoded in target model parameters.

---

## Software Libraries

### nanogcg
**GraySwanAI (2024).**
*nanoGCG — A minimal, efficient implementation of the GCG attack.*
GitHub: https://github.com/GraySwanAI/nanoGCG
License: MIT

Used as the GCG attack backend in `src/attacks.py` via `nanogcg.run()`.
Install: `pip install nanogcg`

---

### open-unlearning
**Maini, P., Feng, Z., et al. (2024).**
*TOFU: A Task of Fictitious Unlearning for LLMs.*
GitHub: https://github.com/locuslab/open-unlearning

Reference implementation consulted for GradAscent, DPO, NPO, NPO+KL, and RMU
unlearning algorithms in `src/unlearn.py`.

---

### MUSE-Books
**Shi, W., et al. (2024).**
*MUSE: Machine Unlearning Six-Way Evaluation for Language Models.*
HuggingFace: `muse-bench/MUSE-Books`
https://huggingface.co/datasets/muse-bench/MUSE-Books

Used in RQ1 for Harry Potter forget QA pairs and domain-membership evaluation.

---

### TOFU
**Maini, P., Feng, Z., et al. (2024).**
*TOFU: A Task of Fictitious Unlearning for LLMs.*
HuggingFace: `locuslab/TOFU`
https://huggingface.co/datasets/locuslab/TOFU

Used in RQ1 as the "unknown" domain baseline (fictitious authors, no real knowledge to unlearn).
