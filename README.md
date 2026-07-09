# Eigenspace Variational Online Newton (EVON)

[![arXiv](https://img.shields.io/badge/arXiv-2606.23357-b31b1b.svg)](https://arxiv.org/abs/2606.23357)
[![Python Version](https://img.shields.io/badge/python-%3E%3D%203.8-blue.svg)](https://pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D%202.0.0-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)

Official PyTorch implementation of the **EVON (Eigenspace Variational Online Newton)** optimizer, introducing **SOAP-Bubbles** for structured weight uncertainty in deep neural networks at scale. Code for the paper's experiments is available at: [team-approx-bayes/evon-experiments](https://github.com/team-approx-bayes/evon-experiments).

> [!NOTE]
> **SOAP-Bubbles: Structured Weight Uncertainty for Neural Networks**  
> _Adrian Robert Minut, Nico Daheim, Marco Miani, Mohammad Emtiyaz Khan, Wu Lin, Thomas Möllenhoff_  
> **ArXiv Paper**: [https://arxiv.org/abs/2606.23357](https://arxiv.org/abs/2606.23357)

---

## Core Concept: SOAP-Bubbles

Estimating non-diagonal (structured) weight uncertainty during training has historically been computationally prohibitive ($O(d^2)$ memory and compute) or difficult to scale to modern architectures like Transformers. 

**EVON** bridges this gap by running [IVON](https://github.com/team-approx-bayes/ivon) (a diagonal-covariance variational method) in the **eigenspace of the SOAP preconditioner**. By leveraging the Kronecker-factored preconditioner's eigenspace, EVON learns a diagonal posterior in a rotated coordinate system. Back in the original weight space, this corresponds to an expressive, block-structured **non-diagonal covariance** (a **SOAP-Bubble**):

$$\boldsymbol{\theta} \sim \mathcal{N}\left(\mathbf{M}, \mathbf{Q}_L \mathrm{diag}\left(\boldsymbol{\sigma}^2\right) \mathbf{Q}_R^\top\right)$$

### Key Advantages:
- **Expressive Uncertainty**: Captures correlations between parameters instead of assuming independent (diagonal) noise.
- **Minimal Overhead**: The computational cost is comparable to the standard second-order [SOAP](https://arxiv.org/abs/2409.11321) optimizer.
- **Scalable**: Successfully pretrains language models (like NanoGPT and LLaMA-134M) and fine-tunes CLIP.
- **Optimal Recovery**: Recovers the full Gaussian posterior for linear models and binary logistic regression.

---

## Installation

To install EVON and its dependencies, clone the repository and install it in editable mode:

```bash
git clone https://github.com/team-approx-bayes/evon.git
cd evon
```

### Option A: Using `uv` (Recommended)
If you have [uv](https://github.com/astral-sh/uv) installed:
```bash
# Sync environment (installs torch and other dependencies)
uv sync

# Install test dependencies (optional)
uv sync --extra test
```

### Option B: Using standard `pip`
```bash
pip install -e .
```

---

## Usage Guide

### 1. Training Loop (Variational Learning)
To train a model with EVON, you sample weights from the variational posterior during the forward pass, compute the gradients, and perform the optimizer step.

Here is the difference between standard training loops (like Adam/SGD) and the EVON training loop:

```diff
  import torch
+ import evon
  
  train_loader = torch.utils.data.DataLoader(train_dataset) 
  model = MLP()
  
- optimizer = torch.optim.Adam(model.parameters())
+ optimizer = evon.EVON(
+     model.parameters(), 
+     lr=1e-3, 
+     ess=len(train_dataset),  # Effective Sample Size (prior scale)
+     hess_init=1.0            # Initial diagonal Hessian estimate
+ )
  
  for X, y in train_loader:
-     optimizer.zero_grad()
-     logit = model(X)
-     loss = torch.nn.CrossEntropyLoss()(logit, y)
-     loss.backward()
-     optimizer.step()
+     # Monte Carlo (MC) sampling loop during training (typically 1 sample)
+     for _ in range(train_samples):
+         with optimizer.sampled_params(train=True):
+             optimizer.zero_grad()
+             logit = model(X)
+             loss = torch.nn.CrossEntropyLoss()(logit, y)
+             loss.backward()
+ 
+     optimizer.step()
```

### 2. Prediction & Bayesian Model Averaging (BMA)

You can perform predictions in two ways:

#### Method A: Standard Prediction (Deterministic / Mean-field)
Uses the learned posterior mean weights directly, behaving just like standard optimizers:
```python
model.eval()
for X, y in test_loader:
    logit = model(X)
    _, prediction = logit.max(1)
```

#### Method B: Bayesian Model Averaging (Ensemble Prediction)
Draws multiple weight samples from the variational posterior, predicts with each weight, and averages the logits. This yields significantly better calibration, perplexity, and robustness:
```python
import torch.nn.functional as F

model.eval()
test_samples = 32

for X, y in test_loader:
    sampled_logits = []
    for _ in range(test_samples):
        with optimizer.sampled_params(train=False):
            sampled_logit = model(X)
            sampled_logits.append(sampled_logit)
            
    logits = torch.mean(torch.stack(sampled_logits), dim=0)
    prob = F.softmax(logits, dim=1)
    _, prediction = prob.max(1)
```

---

## Hyperparameter Configuration

EVON extends the hyperparameter set of [SOAP](https://github.com/nikhilvyas/SOAP) with variational parameters:

| Hyperparameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `ess` | `float` | **(Required)** | **Effective Sample Size** ($N$). Controls the posterior noise scale (lower $N$ injects more weight noise). Usually set to the size of the training dataset or model size. |
| `hess_init` | `float` | **(Required)** | **Initial Hessian estimate**. Must be $> 0$. Controls the initial posterior precision. We find that `1.0` works fine for small models, but smaller values are required as model size increases. |
| `betas` | `tuple` | `(0.95, 0.9999)` | **Momentum coefficients** `(beta1, beta2)`. Note that `beta2` should be around `0.99 < beta2 < 0.99999`. Since `1 - beta2` acts as a learning rate for the Hessian estimate, it has a bigger impact on stability than Adam's `beta2` for the second moment estimate. |
| `mc_samples` | `int` | `1` | Number of MC samples per training step. |
| `hess_clip_ratio`| `float` | `None` | Optional clip ratio for the Hessian estimator residual. Clamps raw estimates to prevent large gradient samples from corrupting the Hessian. |
| `whiten_prec_grad` | `bool` | `True` | Applies Newton-Schulz whitening to the preconditioned gradient (improves stability in deep models). |
| `precondition_frequency` | `int` | `10` | Steps between updating the Shampoo preconditioner eigenspace. |

---

## Running Tests

A unit test suite is included in `tests/test_evon.py` to verify both sampled and deterministic execution modes.

Run tests using pytest:
```bash
uv run pytest
```

---

## Citation

If you use EVON or the SOAP-Bubbles framework in your research, please cite our paper:

```bibtex
@misc{minut2026soapbubbles,
      title={SOAP-Bubbles: Structured Weight Uncertainty for Neural Networks}, 
      author={Adrian Robert Minut and Nico Daheim and Marco Miani and Mohammad Emtiyaz Khan and Wu Lin and Thomas M{"o}llenhoff},
      year={2026},
      eprint={2606.23357},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.23357}
}
```

---

## License

This repository is licensed under the GPLv3+ License. See [LICENSE](LICENSE) for details.
