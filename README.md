<div align="center">

# FoVT — Simple Foveated Vision Transformers

### *When Does Looking Closer Pay Off? Simple Foveated ViTs for Fine-Grained Recognition*

**Boris Kriuk · Mohamed Sobhy**
<br>
The Hong Kong University of Science and Technology

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2611.27788-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.27788)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-%E2%89%A50.9-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-MIT-22c55e.svg?style=for-the-badge)](LICENSE)

[![Trained on NVIDIA A100](https://img.shields.io/badge/Trained%20on-NVIDIA%20A100%2040GB-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/data-center/a100/)
[![Backbone](https://img.shields.io/badge/Backbone-DINOv2%20ViT--S%2F14%20%2B%20reg4-7c3aed.svg?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![Precision](https://img.shields.io/badge/Precision-BF16-0ea5e9.svg?style=for-the-badge)](#-training-protocol)

[![Datasets](https://img.shields.io/badge/Benchmarks-5%20FGVC-f97316.svg?style=flat-square)](#-results)
[![Runs](https://img.shields.io/badge/Training%20runs-250-ec4899.svg?style=flat-square)](#-results)
[![Seeds](https://img.shields.io/badge/Seeds-2-64748b.svg?style=flat-square)](#-statistics)
[![Cost](https://img.shields.io/badge/Cost-0.56%C3%97%20uniform%20448px-10b981.svg?style=flat-square)](#-compute-accounting)
[![Latency](https://img.shields.io/badge/A100%20latency-%E2%88%9226%25%20vs%20448px-14b8a6.svg?style=flat-square)](#-efficiency-in-practice)
[![Memory](https://img.shields.io/badge/Train%20memory-%E2%88%9227%25-6366f1.svg?style=flat-square)](#-efficiency-in-practice)

<br>

**Glance → Look → Zoom → Re-classify.**
<br>
*One backbone. No policy network. No learned gate. No selection loss. No feature reuse.*

</div>

---

## 📌 Table of Contents

- [✨ TL;DR](#-tldr)
- [🧠 How FoVT Works](#-how-fovt-works)
- [🔬 Five Findings](#-five-findings)
- [📊 Results](#-results)
- [🧮 Compute Accounting](#-compute-accounting)
- [⚙️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [🏋️ Training](#️-training)
- [📈 Evaluation & Confidence Gate](#-evaluation--confidence-gate)
- [🐍 Python API](#-python-api)
- [🧪 Training Protocol](#-training-protocol)
- [🖥️ Hardware](#️-hardware)
- [⚡ Efficiency in Practice](#-efficiency-in-practice)
- [🎯 When Does Looking Closer Pay Off?](#-when-does-looking-closer-pay-off)
- [📚 Citation](#-citation)

---

## ✨ TL;DR

Fine-grained recognition **rewards resolution**, but a uniform 448-px ViT processes **every** region at full resolution, even when the useful detail is a single wing or an engine nacelle.

**FoVT** processes a cheap 224-px *glance* first. It then uses the backbone's **own last-block class attention** to pick the `K` most informative cells and re-embeds only those cells from the 448-px image. The **same backbone** then re-classifies the mixed-resolution sequence.

> 🏆 **Headline:** FoVT shows **no detected accuracy difference** from a training-matched 448-px model on **4 of 5** datasets at **0.56× the transformer cost** (0.49× with the confidence gate).

---

## 🧠 How FoVT Works

```
               ┌─────────────────────────── Stage 1: GLANCE (224 px) ───────────────────────────┐
  x (448×448) ─┤  bilinear ↓2 → 16×16 coarse tokens → ViT-S/14 (shared) → glance logits ℓg      │
               │                                         └─► last-block CLS attention  a  (256) │
               └────────────────────────────────────────────────────────────────────────────────┘
                                                   │  Top-K cells (detached, no gradient)
                                                   ▼
               ┌──────────────────────── Stage 2: FOVEATED REFINEMENT ──────────────────────────┐
               │  re-embed K cells from the 448-px image → 4K fine tokens                       │
               │  Z_K = [prefix ; 256−K coarse input tokens ; 4K fine tokens]  (261 + 3K)       │
               │  → ViT-S/14 (same weights) → final logits ℓf                                   │
               └────────────────────────────────────────────────────────────────────────────────┘

   Training:  L = CE_ε(ℓf, y) + λ · CE_ε(ℓg, y)        ε = 0.1,  λ = 0.5,  K = 64
   Inference: optional confidence gate → return ℓg and skip Stage 2 for confident images
```

| Component | FoVT | CF-ViT |
|:--|:--:|:--:|
| Saliency signal | Last-block CLS attention | EMA of CLS attention over blocks 4–12 |
| Feature reuse | ❌ | ✅ |
| Distillation loss | ❌ | ✅ (KL) |
| Selection schedule | ❌ | ✅ (full grid → informative regions) |
| Extra modules | Only 2 zero-init scale embeddings | Reuse MLP |
| Sequence length | Fixed per `K` (static shapes) | Fixed per `α` |

Every region appears at **exactly one scale**. The retained coarse tokens are the original **input embeddings**, not contextualized glance outputs.

---

## 🔬 Five Findings

| ❓ Question | 💡 Answer |
|:--|:--|
| **Where to look?** | Attention beats random cells on **all 5** datasets (**+0.72 to +2.45 pp**, and up to **+4.18 pp** on the smallest objects) |
| **How to look?** | At equal cost, **no Holm-corrected deficit** vs. a port of the official CF-ViT, with no reuse, distillation, or schedule |
| **Which images?** | Skipping the **19–24%** most confident images leaves mean accuracy unchanged to two decimals |
| **What does foveation buy?** | Mostly **compute**: it matches a training-matched 448-px model on 4/5 datasets at **0.56×** cost |
| **When does it pay off?** | It tracks **selection gain** (ρ = 0.9) and **resolution headroom** (ρ ≈ 0.87). Largest on **NABirds** and **Aircraft**, not significant on **Dogs** |

---

## 📊 Results

**Top-1 accuracy (%)**, mean ± std over two seeds. Costs are transformer GMACs.

| Model | GMACs | CUB | NABirds | Aircraft | Cars | Dogs |
|:--|--:|:--:|:--:|:--:|:--:|:--:|
| Uniform 224 | 6.17 | 88.54 ±0.27 | 87.69 ±0.09 | 88.25 ±0.15 | 93.59 ±0.10 | 86.12 ±0.12 |
| Uniform 336 | 15.45 | 89.57 ±0.06 | 89.95 ±0.01 | 89.69 ±0.36 | 94.11 ±0.21 | 87.05 ±0.01 |
| Uniform 392 | 22.49 | 90.07 ±0.33 | 90.09 ±0.33 | 90.38 ±0.40 | 94.02 ±0.06 | 87.07 ±0.05 |
| Uniform 448 | 31.61 | 90.23 ±0.00 | 90.61 ±0.04 | 90.74 ±0.49 | 94.20 ±0.06 | **87.40** ±0.27 |
| Uniform 448 + glance loss | 31.61 | 90.14 ±0.24 | **90.93** ±0.08 | **91.96** ±0.51 | 94.42 ±0.03 | 87.31 ±0.11 |
| CF-ViT (official port), α = 0.25 | 17.68 | **90.40** ±0.15 | 90.68 ±0.09 | 91.28 ±0.02 | 94.10 ±0.06 | 86.79 ±0.07 |
| FoVT glance exit (K = 0) | 6.17 | 89.14 ±0.02 | 88.34 ±0.06 | 89.29 ±0.30 | 93.96 ±0.07 | 86.38 ±0.34 |
| **FoVT, K = 64** | **17.68** | 90.11 ±0.17 | 90.89 ±0.05 | 91.43 ±0.28 | **94.46** ±0.13 | 87.21 ±0.15 |
| **FoVT, K = 64, gated (81% refined)** | **15.45** | 90.11 ±0.17 | 90.89 ±0.06 | 91.43 ±0.28 | **94.46** ±0.13 | 87.21 ±0.15 |

### ⚖️ Equal-cost gain: gated FoVT − uniform 336 (15.45 GMACs)

| CUB | NABirds | Aircraft | Cars | Dogs |
|:--:|:--:|:--:|:--:|:--:|
| **+0.54** `[+0.19, +0.90]` | **+0.94** `[+0.74, +1.15]` | **+1.74** `[+1.11, +2.37]` | **+0.35** `[+0.09, +0.60]` | +0.17 `[−0.20, +0.54]` |

<sub>Marginal 95% paired bootstrap intervals over test images. Bold means the unadjusted interval excludes zero. The NABirds and Aircraft gains survive Holm correction in both seeds.</sub>

---

## 🧮 Compute Accounting

For `L` blocks of width `D` (MLP ratio 4) and `N` tokens including the prefix:

```
C(N) = L · (12·N·D² + 2·N²·D)          FoVT cost = C(261) + C(261 + 3K)
```

| Config | Tokens | GMACs | × Uniform 448 |
|:--|:--:|:--:|:--:|
| Glance only (K = 0) | 261 | 6.17 | 0.20× |
| FoVT K = 32 | 261 + 357 | 14.92 | 0.47× |
| FoVT K = 64, gated 81% | — | 15.45 | 0.49× |
| **FoVT K = 64** | 261 + 453 | **17.68** | **0.56×** |
| FoVT K = 128 | 261 + 645 | 23.70 | 0.75× |
| Uniform 448 | 1029 | 31.61 | 1.00× |

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/FoVT.git
cd FoVT
pip install torch torchvision timm scipy pillow numpy
```

---

## 🚀 Quick Start

```bash
# Train FoVT (K=64) on Stanford Cars, seed 0
python fovt.py train --dataset cars --data-root /data/stanford_cars --out runs/cars_fovt --seed 0

# Evaluate: K sweep + confidence-gated budgets
python fovt.py eval --ckpt runs/cars_fovt/final.pt --data-root /data/stanford_cars --ks 0 32 64 128 256

# Benchmark latency against uniform baselines
python fovt.py bench
```

---

## 🏋️ Training

```bash
# FoVT with attention selection (paper default)
python fovt.py train --dataset cars --data-root /data/stanford_cars --model fovt --K 64 --selector attn

# Ablations
python fovt.py train ... --selector random          # random cells
python fovt.py train ... --selector center          # center prior
python fovt.py train ... --selector allblock        # attention averaged over all 12 blocks
python fovt.py train ... --glance-weight 0          # no glance loss
python fovt.py train ... --scale-emb false          # no scale embeddings

# Uniform baselines
python fovt.py train ... --model uniform --res 336

# Training-matched 448 px (uniform 448 + glance loss)
python fovt.py train ... --model fovt --K 256

# Any other dataset in ImageFolder layout: <root>/train/<class>/*.jpg and <root>/test/<class>/*.jpg
python fovt.py train --dataset folder --data-root /data/my_fgvc
```

Training is **resumable**. A checkpoint is written after every epoch.

---

## 📈 Evaluation & Confidence Gate

The confidence gate refines an image only when the glance is unsure:

```
ŷ(x) = argmax ℓf   if  max softmax(ℓg) < η
       argmax ℓg   otherwise
```

For a target budget `B`, the refined fraction is `f = (B − C(261)) / C(261 + 3K)`.

| Gate | How η is set | Deployable? |
|:--|:--|:--:|
| **Rank** | Refines exactly `round(f·n)` least-confident test images | ❌ needs the test pool |
| **Calibrated** | η = `f`-quantile of glance confidences on a **held-out** split | ✅ per image |

> ⚠️ **Calibrate on held-out images, not training images.** A fine-tuned glance is overconfident on its own training data (mean confidence 0.87–0.90 vs 0.73–0.79 on test), so training-calibrated thresholds overshoot the budget by 7–9%. Held-out calibration stays within **2.7%** of the target cost.

```bash
python fovt.py calibrate --ckpt runs/cars_fovt/final.pt --data-root /data/stanford_cars --budget 15.45
```

---

## 🐍 Python API

```python
import torch
from fovt import FoVT

model = FoVT(num_classes=196, K=64).cuda().eval()

x = torch.randn(8, 3, 448, 448, device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(x, return_idx=True)          # {'final', 'glance', 'idx'}
    out_k32 = model(x, K=32)                 # change the budget at inference, no retraining
    logits, refined = model.predict_gated(x, eta=0.9)   # skips Stage 2 for confident images

print(model.gmacs(K=64))                     # 17.68
```

---

## 🧪 Training Protocol

| Hyperparameter | Value |
|:--|:--|
| Backbone | `vit_small_patch14_reg4_dinov2.lvd142m` (timm) |
| Optimizer | AdamW |
| Batch size | 32 |
| Precision | BF16 autocast |
| Epochs | 20 (10 for NABirds), 1 warmup epoch + cosine decay |
| LR (backbone / head) | 1e-4 / 1e-3 |
| Layer-wise LR decay | 0.85 |
| Weight decay | 0.05 |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Drop path | 0.1 |
| Train augmentation | RandomResizedCrop 448 (scale 0.3–1), h-flip, color jitter |
| Eval transform | Resize 515 → CenterCrop 448 |
| Glance loss weight λ | 0.5 |
| Training budget K | 64 |
| Checkpoint | Final epoch, no early stopping, no per-dataset tuning |

---

## 🖥️ Hardware

<div align="center">

[![NVIDIA A100](https://img.shields.io/badge/NVIDIA-A100--SXM4--40GB-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/data-center/a100/)

</div>

FoVT and the baseline models were trained on an **NVIDIA A100-SXM4-40GB** with **BF16** mixed precision.

| | FoVT (K = 64) | Uniform 448 |
|:--|:--:|:--:|
| Peak training memory | **4.64 GiB** | 6.35 GiB |
| Epoch time (relative) | 1.12–1.19× | 1.00× |

<sub>Footnote from the paper: the Stanford Dogs runs used an NVIDIA RTX 4090 Laptop GPU and the CF-ViT port used an NVIDIA L4. Everything else ran on the A100.</sub>

---

## ⚡ Efficiency in Practice

A100, batch 32, BF16 (ms/image, synchronized time ÷ batch):

| Model | GMACs | ms / image | vs FoVT |
|:--|:--:|:--:|:--:|
| Uniform 336 | 15.45 | 0.43 | faster |
| FoVT K = 32 | 14.92 | 0.52 | — |
| **FoVT K = 64** | **17.68** | **0.56** | — |
| Uniform 392 | 22.49 | 0.58 | −4% |
| Uniform 448 | 31.61 | 0.75 | **−26%** |

Fused attention kernels shrink the quadratic term that foveation removes, and the second pass adds a fixed overhead. Arithmetic savings therefore only partly turn into wall-clock savings. Fusing selection and gathering into the forward pass is the main engineering target.

---

## 🎯 When Does Looking Closer Pay Off?

```
  Equal-cost gain ▲
       (pp)       │  ✈️ Aircraft  +1.74
                  │  🐦 NABirds   +0.94
                  │  🐤 CUB       +0.54
                  │  🚗 Cars      +0.35
                  │  🐕 Dogs      +0.17 (n.s.)
                  └──────────────────────────► resolution headroom (448 − 336) & selection gain
```

✅ **Pays off** when the discriminative evidence is **small** (so *where* to look matters) **and** still benefits from resolution beyond ~336 px. Examples: birds, aircraft, and likely camera traps, shelf products, and defect inspection.

❌ **Little gain** when identity comes from body-level cues visible at moderate resolution and the object fills the frame. Example: Stanford Dogs, mean box area 0.71.

---

## 📚 Citation

```bibtex
@article{kriuk2026fovt,
  title   = {When Does Looking Closer Pay Off? Simple Foveated ViTs for Fine-Grained Recognition},
  author  = {Kriuk, Boris and Sobhy, Mohamed},
  journal = {arXiv preprint arXiv:2611.27788},
  year    = {2026}
}
```

---

## 🙏 Acknowledgements

Built on [timm](https://github.com/huggingface/pytorch-image-models) and [DINOv2](https://github.com/facebookresearch/dinov2). Compared against [CF-ViT](https://github.com/ChenMnZ/CF-ViT).

<div align="center">

**⭐ If FoVT is useful in your research, please consider starring the repo! ⭐**

<sub>Made with 👁️ at HKUST</sub>

</div>
