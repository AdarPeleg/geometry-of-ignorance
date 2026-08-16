# References

## Papers

### Activation Steering Attack
**Seyitoglu, A., Kuvshinov, A., Schwinn, L., Gunnemann, S. (2024).**
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

### Head-to-Head Representational Baseline (Xu et al.)
**Xu, X., Yue, X., Liu, Y., Ye, Q., Zheng, H., Hu, P., Du, M., Hu, H. (2025).**
*Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs.*
arXiv:2505.16831. https://arxiv.org/abs/2505.16831

Basis for the head-to-head comparison in `scripts/rq2/repro_representational_tools.py`
(PCA-similarity, PCA-shift, and centered kernel alignment metrics), benchmarked directly
against AUSS on the same RQ2 configurations.

---

### Gram-Matrix Entropy Formulation
**Skean, O., Arefin, M. R., Zhao, D., Patel, N., Naghiyev, J., LeCun, Y., Shwartz-Ziv, R. (2025).**
*Layer by Layer: Uncovering Hidden Representations in Language Models.*
arXiv:2502.02013. https://arxiv.org/abs/2502.02013

Basis for the Renyi-2 Gram-matrix entropy metric in `src/entropy.py`, adapted here to
concept-differential (anon-vector) matrices rather than raw hidden states.

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

### Representational_Analysis_Tools
**Xu, X., et al.**
GitHub: https://github.com/XiaoyuXU1/Representational_Analysis_Tools

The authors' own released implementation of the PCA-similarity, PCA-shift, and CKA
metrics from Xu et al. (2025), cloned unmodified for the head-to-head benchmark in
`scripts/rq2/repro_representational_tools.py`. See that script's docstring for the
one environment-compatibility patch applied at import time (a missing pad-token
default that otherwise crashes on Llama-2-family tokenizers; no change to their
algorithm).

---

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
