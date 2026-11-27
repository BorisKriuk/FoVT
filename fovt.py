#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoVT: Simple Foveated Vision Transformers for Fine-Grained Recognition
=====================================================================

Reference implementation of
    "When Does Looking Closer Pay Off? Simple Foveated ViTs for Fine-Grained Recognition"
    Boris Kriuk, Mohamed Sobhy (arXiv:2611.27788)

Pipeline: glance (224 px) -> look (last-block CLS attention) -> zoom (re-embed top-K cells
from the 448-px image) -> re-classify with the same backbone.

Trained on an NVIDIA A100-SXM4-40GB with BF16 mixed precision.

Usage
-----
    python fovt.py train     --dataset cars --data-root /data/stanford_cars --out runs/cars_fovt
    python fovt.py eval      --ckpt runs/cars_fovt/final.pt --data-root /data/stanford_cars
    python fovt.py calibrate --ckpt runs/cars_fovt/final.pt --data-root /data/stanford_cars --budget 15.45
    python fovt.py bench
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from timm.layers import resample_abs_pos_embed
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
HI, LO, PATCH = 448, 224, 14                      # full-res input, glance res, patch size
BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SELECTORS = ("attn", "allblock", "random", "center")


# --------------------------------------------------------------------------------------------
# Compute accounting (Sec. III-E)
# --------------------------------------------------------------------------------------------
def transformer_gmacs(n_tokens: int, dim: int = 384, depth: int = 12) -> float:
    """C(N) = L (12 N D^2 + 2 N^2 D), in GMACs. Excludes patch embed, heads, gathering."""
    return depth * (12 * n_tokens * dim * dim + 2 * n_tokens ** 2 * dim) / 1e9


def refine_fraction(budget: float, n_glance: int, n_fine: int) -> float:
    """Fraction of images that can be refined under a transformer budget (GMACs)."""
    return float(min(1.0, max(0.0, (budget - transformer_gmacs(n_glance)) / transformer_gmacs(n_fine))))


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
def _make_backbone(num_classes: int, img_size: int, pretrained: bool, drop_path: float,
                   backbone: str = BACKBONE) -> nn.Module:
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes,
                             img_size=img_size, drop_path_rate=drop_path)


class FoVT(nn.Module):
    """Foveated ViT: one shared backbone, two passes, fixed sequence length per K.

    Args:
        num_classes:   number of classes.
        K:             default number of refined cells (out of 16x16 = 256).
        selector:      'attn' (last-block CLS attention), 'allblock' (mean over all blocks),
                       'random', or 'center' (K cells closest to the image center).
        scale_emb:     add zero-initialised coarse/fine scale embeddings.
        glance_weight: lambda of the auxiliary glance loss (training only).
    """
    kind = "fovt"

    def __init__(self, num_classes: int, K: int = 64, selector: str = "attn", scale_emb: bool = True,
                 glance_weight: float = 0.5, pretrained: bool = True, drop_path: float = 0.1,
                 backbone: str = BACKBONE):
        super().__init__()
        assert selector in SELECTORS, f"selector must be one of {SELECTORS}"
        self.m = _make_backbone(num_classes, LO, pretrained, drop_path, backbone)
        self.gc, self.gf, self.D = LO // PATCH, HI // PATCH, self.m.embed_dim       # 16, 32, 384
        self.n_prefix = self.m.num_prefix_tokens                                     # cls + 4 registers
        self.pe_prefix = 0 if getattr(self.m, "no_embed_class", False) else self.n_prefix
        if scale_emb:
            self.scale_emb = nn.Parameter(torch.zeros(2, self.D))
        else:
            self.register_buffer("scale_emb", torch.zeros(2, self.D), persistent=False)
        self.K, self.selector, self.glance_weight = K, selector, glance_weight

        yy, xx = torch.meshgrid(torch.arange(self.gc), torch.arange(self.gc), indexing="ij")
        c = (self.gc - 1) / 2
        self.register_buffer("_center_score", -((yy - c) ** 2 + (xx - c) ** 2).float().flatten(),
                             persistent=False)

    # ---- bookkeeping ------------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.gc ** 2

    def seq_lens(self, K: Optional[int] = None) -> List[int]:
        K = self.K if K is None else K
        n = self.n_cells + self.n_prefix
        return [n] if K == 0 else [n, n + 3 * K]

    def gmacs(self, K: Optional[int] = None) -> float:
        return sum(transformer_gmacs(n, self.D, len(self.m.blocks)) for n in self.seq_lens(K))

    # ---- building blocks --------------------------------------------------------------------
    def _pos(self, g: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pe = resample_abs_pos_embed(self.m.pos_embed, (g, g), num_prefix_tokens=self.pe_prefix)
        return pe[:, self.pe_prefix:], pe[:, :self.pe_prefix]

    def _patchify(self, img: torch.Tensor) -> torch.Tensor:
        pe = self.m.patch_embed
        return pe.norm(pe.proj(img).flatten(2).transpose(1, 2))

    def _prefix(self, B: int, pos_prefix: torch.Tensor) -> torch.Tensor:
        toks = [self.m.cls_token.expand(B, -1, -1)]
        if getattr(self.m, "reg_token", None) is not None:
            toks.append(self.m.reg_token.expand(B, -1, -1))
        t = torch.cat(toks, 1)
        return t if getattr(self.m, "no_embed_class", False) else t + pos_prefix

    def _cls_attn(self, blk: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Head-averaged CLS->patch attention of one block (Eq. 3); prefix entries dropped."""
        a = blk.attn
        B, N, _ = x.shape
        qkv = a.qkv(blk.norm1(x)).reshape(B, N, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = a.q_norm(qkv[0]), a.k_norm(qkv[1])
        att = (q[:, :, :1] @ k.transpose(-2, -1)).squeeze(2) * a.scale
        return att.float().softmax(-1).mean(1)[:, self.n_prefix:]

    def _run(self, x: torch.Tensor, saliency: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.m.norm_pre(self.m.patch_drop(self.m.pos_drop(x)))
        nb, sal = len(self.m.blocks), None
        for b, blk in enumerate(self.m.blocks):
            if saliency == "all" or (saliency == "last" and b == nb - 1):
                s = self._cls_attn(blk, x)
                sal = s if sal is None else sal + s
            x = blk(x)
        if saliency == "all":
            sal = sal / nb
        return self.m.forward_head(self.m.norm(x)), sal

    def _select(self, sal: torch.Tensor, K: int) -> torch.Tensor:
        B = sal.shape[0]
        if self.selector in ("attn", "allblock"):
            score = sal
        elif self.selector == "random":
            score = torch.rand(B, self.n_cells, device=sal.device)
        else:
            score = self._center_score.to(sal.device).expand(B, -1)
        return score.detach().topk(K, dim=1).indices

    # ---- the two stages ---------------------------------------------------------------------
    def glance(self, x_hi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stage 1: 224-px glance. Returns glance logits, saliency, coarse input tokens."""
        B = x_hi.shape[0]
        xg = F.interpolate(x_hi, (LO, LO), mode="bilinear", antialias=True, align_corners=False)
        pos_c, pre = self._pos(self.gc)
        tc = self._patchify(xg) + pos_c + self.scale_emb[0]
        mode = "all" if self.selector == "allblock" else "last"
        logits, sal = self._run(torch.cat([self._prefix(B, pre), tc], 1), mode)
        return {"logits": logits, "sal": sal, "tc": tc, "pre": pre}

    def refine(self, x_hi: torch.Tensor, tc: torch.Tensor, sal: torch.Tensor, pre: torch.Tensor,
               K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 2: replace the top-K coarse cells by their 2x2 fine children and re-classify."""
        B = x_hi.shape[0]
        idx = self._select(sal, K)                                                  # (B, K)
        keep = torch.ones(B, self.n_cells, dtype=torch.bool, device=tc.device).scatter_(1, idx, False)
        coarse = tc[keep].reshape(B, self.n_cells - K, self.D)

        i, j = idx // self.gc, idx % self.gc
        fidx = torch.stack([(2 * i + di) * self.gf + (2 * j + dj) for di in (0, 1) for dj in (0, 1)],
                           -1).view(B, -1)                                          # (B, 4K)
        pos_f, _ = self._pos(self.gf)
        tf = self._patchify(x_hi) + pos_f + self.scale_emb[1]
        fine = torch.gather(tf, 1, fidx.unsqueeze(-1).expand(-1, -1, self.D))

        logits, _ = self._run(torch.cat([self._prefix(B, pre), coarse, fine], 1), None)
        return logits, idx

    def forward(self, x_hi: torch.Tensor, K: Optional[int] = None,
                return_idx: bool = False) -> Dict[str, torch.Tensor]:
        K = self.K if K is None else K
        g = self.glance(x_hi)
        out = {"final": g["logits"], "glance": g["logits"]}
        if K == 0:
            return out
        logits_f, idx = self.refine(x_hi, g["tc"], g["sal"], g["pre"], K)
        out["final"] = logits_f
        if return_idx:
            out["idx"] = idx
        return out

    @torch.no_grad()
    def predict_gated(self, x_hi: torch.Tensor, eta: float,
                      K: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Confidence gate (Eq. 5) with batch compaction: only images with glance
        confidence < eta run Stage 2. Returns (logits, refined_mask)."""
        K = self.K if K is None else K
        g = self.glance(x_hi)
        logits = g["logits"].clone()
        conf = g["logits"].float().softmax(-1).amax(-1)
        go = conf < eta
        if K > 0 and go.any():
            sub = go.nonzero(as_tuple=True)[0]
            lf, _ = self.refine(x_hi[sub], g["tc"][sub], g["sal"][sub], g["pre"], K)
            logits[sub] = lf.to(logits.dtype)
        return logits, go


class UniformViT(nn.Module):
    """Uniform-resolution baseline: the 448-px view resized to `res`."""
    kind = "uniform"

    def __init__(self, num_classes: int, res: int = 448, pretrained: bool = True, drop_path: float = 0.1,
                 backbone: str = BACKBONE):
        super().__init__()
        self.res, self.glance_weight = res, 0.0
        self.m = _make_backbone(num_classes, res, pretrained, drop_path, backbone)

    def gmacs(self, K: Optional[int] = None) -> float:
        n = (self.res // PATCH) ** 2 + self.m.num_prefix_tokens
        return transformer_gmacs(n, self.m.embed_dim, len(self.m.blocks))

    def forward(self, x_hi: torch.Tensor, K: Optional[int] = None,
                return_idx: bool = False) -> Dict[str, Optional[torch.Tensor]]:
        x = x_hi if self.res == HI else F.interpolate(x_hi, (self.res, self.res), mode="bilinear",
                                                      antialias=True, align_corners=False)
        return {"final": self.m(x), "glance": None}


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------
@dataclass
class Config:
    dataset: str = "cars"            # 'cars' (Stanford Cars .mat annotations) | 'folder' (ImageFolder)
    data_root: str = ""
    out: str = "runs/fovt"
    model: str = "fovt"              # 'fovt' | 'uniform'
    res: int = 448                   # uniform baseline resolution
    K: int = 64
    selector: str = "attn"
    glance_weight: float = 0.5
    scale_emb: bool = True
    epochs: int = 20
    batch_size: int = 32
    base_lr: float = 1e-4
    head_lr: float = 1e-3
    layer_decay: float = 0.85
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    drop_path: float = 0.1
    workers: int = 8
    seed: int = 0


def build_model(cfg: Config, num_classes: int, pretrained: bool = True) -> nn.Module:
    if cfg.model == "fovt":
        return FoVT(num_classes, K=cfg.K, selector=cfg.selector, scale_emb=cfg.scale_emb,
                    glance_weight=cfg.glance_weight, pretrained=pretrained, drop_path=cfg.drop_path)
    if cfg.model == "uniform":
        return UniformViT(num_classes, res=cfg.res, pretrained=pretrained, drop_path=cfg.drop_path)
    raise ValueError(f"unknown model '{cfg.model}'")


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def train_transform() -> T.Compose:
    return T.Compose([T.RandomResizedCrop(HI, scale=(0.3, 1.0), antialias=True), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.3, 0.3, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform() -> T.Compose:
    return T.Compose([T.Resize(int(HI * 1.15), antialias=True), T.CenterCrop(HI),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def load_img(path: str) -> Image.Image:
    img = Image.open(path)
    img.draft("RGB", (2 * HI, 2 * HI))          # fast JPEG decode at reduced size
    return img.convert("RGB")


def _find(root: str, name: str) -> str:
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return min(hits, key=len)


class StanfordCars(Dataset):
    """Stanford Cars (196 classes, 8,144 train / 8,041 test) from the original .mat annotations."""

    def __init__(self, root: str, train: bool, transform):
        import scipy.io as sio
        split = "cars_train" if train else "cars_test"
        mat = _find(root, "cars_train_annos.mat" if train else "cars_test_annos_withlabels.mat")
        img_dir = os.path.join(root, split, split) if os.path.isdir(os.path.join(root, split, split)) \
            else os.path.join(root, split)
        self.items = [(os.path.join(img_dir, str(r["fname"][0])), int(r["class"][0, 0]) - 1)
                      for r in sio.loadmat(mat)["annotations"][0]]
        self.transform, self.num_classes = transform, len({c for _, c in self.items})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(load_img(p)), y


class Folder(Dataset):
    """Generic ImageFolder layout: <root>/<split>/<class>/<image>."""

    def __init__(self, root: str, train: bool, transform):
        from torchvision.datasets import ImageFolder
        self.ds = ImageFolder(os.path.join(root, "train" if train else "test"), loader=load_img)
        self.transform, self.num_classes = transform, len(self.ds.classes)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        p, y = self.ds.samples[i]
        return self.transform(load_img(p)), y


def build_dataset(cfg: Config, train: bool) -> Dataset:
    tf = train_transform() if train else eval_transform()
    if cfg.dataset == "cars":
        return StanfordCars(cfg.data_root, train, tf)
    if cfg.dataset == "folder":
        return Folder(cfg.data_root, train, tf)
    raise ValueError(f"unknown dataset '{cfg.dataset}'")


# --------------------------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------------------------
def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def param_groups(model: nn.Module, base_lr: float, head_lr: float, layer_decay: float, wd: float):
    """Layer-wise LR decay; head and scale embeddings at the head LR; no decay on norms/tokens."""
    L, groups = len(model.m.blocks), {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("m.blocks."):
            layer = int(n.split(".")[2]) + 1
        elif n.startswith(("m.cls_token", "m.pos_embed", "m.patch_embed", "m.reg_token")):
            layer = 0
        elif n.startswith(("m.head", "m.fc_norm", "scale_emb")):
            layer = L + 1
        else:
            layer = L
        lr = head_lr if layer == L + 1 else base_lr * layer_decay ** (L - layer)
        decay = 0.0 if (p.ndim == 1 or "token" in n or "pos_embed" in n or "scale_emb" in n) else wd
        groups.setdefault((lr, decay), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": d} for (lr, d), ps in groups.items()]


def amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def train(cfg: Config) -> str:
    """Train with L = CE(final) + lambda * CE(glance). Resumable; returns path to final weights."""
    assert torch.cuda.is_available(), "CUDA is required (trained on an NVIDIA A100)."
    dev, DT = torch.device("cuda"), amp_dtype()
    os.makedirs(cfg.out, exist_ok=True)
    set_seed(cfg.seed)

    ds = build_dataset(cfg, train=True)
    g = torch.Generator()
    dl = DataLoader(ds, cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True,
                    drop_last=True, generator=g)
    model = build_model(cfg, ds.num_classes).to(dev)
    opt = torch.optim.AdamW(param_groups(model, cfg.base_lr, cfg.head_lr, cfg.layer_decay, cfg.weight_decay))
    steps, warm = cfg.epochs * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=(DT == torch.float16))
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    ck_path, start_ep = os.path.join(cfg.out, "last.ckpt"), 0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        print(f"resuming at epoch {start_ep}")

    print(f"{cfg.model} | {model.gmacs():.2f} GMACs | {len(ds)} images | {ds.num_classes} classes "
          f"| GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    for ep in range(start_ep, cfg.epochs):
        model.train(); g.manual_seed(cfg.seed * 100003 + ep)
        t0, tot, n = time.time(), 0.0, 0
        for img, y in dl:
            img, y = img.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=DT):
                o = model(img)
                loss = ce(o["final"], y)
                if o["glance"] is not None and model.glance_weight > 0:
                    loss = loss + model.glance_weight * ce(o["glance"], y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"ep {ep:02d} | loss {tot / n:.4f} | {time.time() - t0:.0f}s | peak {peak:.2f} GiB")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep), ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    final = os.path.join(cfg.out, "final.pt")
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "num_classes": ds.num_classes}, final)
    os.remove(ck_path)
    print(f"saved {final}")
    return final


# --------------------------------------------------------------------------------------------
# Evaluation, gating, statistics
# --------------------------------------------------------------------------------------------
def load_checkpoint(path: str, device: str = "cuda") -> Tuple[nn.Module, Config]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    model = build_model(cfg, ck["num_classes"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def collect(model: nn.Module, loader: DataLoader, K: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Per-image glance/final predictions and glance confidences (reproducible random selector)."""
    model.eval(); dev, DT = next(model.parameters()).device, amp_dtype()
    out = {k: [] for k in ("final", "glance", "gconf", "y")}
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(0)
        for x, y in loader:
            with torch.autocast(dev.type, dtype=DT, enabled=dev.type == "cuda"):
                o = model(x.to(dev, non_blocking=True), K=K)
            lf = o["final"].float()
            lg = (o["glance"] if o["glance"] is not None else o["final"]).float()
            out["final"].append(lf.argmax(1).cpu()); out["glance"].append(lg.argmax(1).cpu())
            out["gconf"].append(lg.softmax(-1).amax(-1).cpu()); out["y"].append(y)
    return {k: torch.cat(v) for k, v in out.items()}


def rank_gate(glance: torch.Tensor, final: torch.Tensor, gconf: torch.Tensor, frac: float) -> torch.Tensor:
    """Budget-exact gate: refine exactly round(frac*n) least-confident images (needs the test pool)."""
    go = torch.zeros(len(gconf), dtype=torch.bool)
    go[torch.argsort(gconf)[:int(round(frac * len(gconf)))]] = True
    return torch.where(go, final, glance)


def calibrate_eta(gconf_heldout: torch.Tensor, frac: float) -> float:
    """Deployable gate threshold: the frac-quantile of HELD-OUT glance confidences.
    (Do not calibrate on training images: the glance is overconfident there.)"""
    return float(torch.quantile(gconf_heldout.double(), frac).item())


def acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred == y).float().mean().item() * 100


def mcnemar(correct_a: torch.Tensor, correct_b: torch.Tensor) -> Tuple[int, int, float]:
    """Exact McNemar test on paired per-image correctness."""
    from scipy.stats import binomtest
    w, l = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
    return w, l, (binomtest(w, w + l, 0.5).pvalue if w + l else 1.0)


def paired_bootstrap(correct_a: List[torch.Tensor], correct_b: List[torch.Tensor],
                     n_boot: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """Seed-averaged per-image difference, bootstrap over test images: (mean pp, 2.5%, 97.5%)."""
    d = torch.stack([a.float() - b.float() for a, b in zip(correct_a, correct_b)]).mean(0)
    g, n = torch.Generator().manual_seed(seed), len(d)
    ms = torch.cat([d[torch.randint(0, n, (min(250, n_boot - s), n), generator=g)].mean(1)
                    for s in range(0, n_boot, 250)]) * 100
    return d.mean().item() * 100, ms.quantile(0.025).item(), ms.quantile(0.975).item()


def evaluate(ckpt: str, data_root: str, ks: List[int], budgets: List[float], batch_size: int = 64,
             workers: int = 8) -> Dict:
    model, cfg = load_checkpoint(ckpt)
    cfg.data_root = data_root or cfg.data_root
    dl = DataLoader(build_dataset(cfg, train=False), batch_size, shuffle=False, num_workers=workers,
                    pin_memory=True)
    report = {"model": cfg.model, "K_sweep": {}, "gated": {}}
    if isinstance(model, UniformViT):
        r = collect(model, dl)
        report["acc"] = acc(r["final"], r["y"])
        print(f"uniform {cfg.res}px | {model.gmacs():.2f} GMACs | top-1 {report['acc']:.2f}")
        return report

    n_glance = model.seq_lens(0)[0]
    for K in ks:
        r = collect(model, dl, K=K)
        a = acc(r["final"], r["y"]); report["K_sweep"][K] = dict(gmacs=model.gmacs(K), acc=a)
        print(f"K={K:3d} | {model.gmacs(K):6.2f} GMACs | top-1 {a:.2f}")
        if K == model.K:
            n_fine = model.seq_lens(K)[1]
            for b in budgets:
                f = refine_fraction(b, n_glance, n_fine)
                ag = acc(rank_gate(r["glance"], r["final"], r["gconf"], f), r["y"])
                report["gated"][f"{b:.2f}"] = dict(refined_frac=f, acc=ag)
                print(f"  gated @ {b:5.2f} GMACs | refined {f * 100:4.1f}% | top-1 {ag:.2f}")
    return report


@torch.no_grad()
def bench(num_classes: int = 196, batch: int = 32, n: int = 20) -> None:
    """Batch-normalised forward time (ms/image), BF16, synthetic 448-px inputs on GPU."""
    dev, DT = torch.device("cuda"), amp_dtype()
    x = torch.randn(batch, 3, HI, HI, device=dev)
    models = {f"FoVT K={k}": (FoVT(num_classes, K=k, pretrained=False), k) for k in (32, 64)}
    models.update({f"Uniform {r}": (UniformViT(num_classes, r, pretrained=False), None) for r in (336, 392, 448)})
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for name, (m, K) in models.items():
        m = m.to(dev).eval()
        with torch.autocast("cuda", dtype=DT):
            for _ in range(5): m(x, K=K)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(n): m(x, K=K)
            torch.cuda.synchronize()
        print(f"{name:12s} | {m.gmacs(K):6.2f} GMACs | {(time.time() - t) / n / batch * 1000:.3f} ms/img")
        m.cpu(); del m; torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            p.add_argument(flag, type=_str2bool, default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type) if isinstance(f.type, str) else f.type
            p.add_argument(flag, type=typ, default=f.default)


def main() -> None:
    ap = argparse.ArgumentParser(description="FoVT: Simple Foveated ViTs for Fine-Grained Recognition")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="train FoVT or a uniform baseline"); _add_config_args(p_tr)

    p_ev = sub.add_parser("eval", help="K sweep + confidence-gated budgets")
    p_ev.add_argument("--ckpt", required=True); p_ev.add_argument("--data-root", default="")
    p_ev.add_argument("--ks", type=int, nargs="+", default=[0, 32, 64, 128, 256])
    p_ev.add_argument("--budgets", type=float, nargs="+", default=[14.92, 15.45])
    p_ev.add_argument("--batch-size", type=int, default=64); p_ev.add_argument("--workers", type=int, default=8)
    p_ev.add_argument("--save", default="")

    p_ca = sub.add_parser("calibrate", help="deployable gate threshold from a held-out split")
    p_ca.add_argument("--ckpt", required=True); p_ca.add_argument("--data-root", default="")
    p_ca.add_argument("--budget", type=float, default=15.45)
    p_ca.add_argument("--batch-size", type=int, default=64); p_ca.add_argument("--workers", type=int, default=8)

    p_be = sub.add_parser("bench", help="latency benchmark")
    p_be.add_argument("--num-classes", type=int, default=196)

    a = ap.parse_args()
    if a.cmd == "train":
        cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
        train(cfg)
    elif a.cmd == "eval":
        rep = evaluate(a.ckpt, a.data_root, a.ks, a.budgets, a.batch_size, a.workers)
        if a.save:
            json.dump(rep, open(a.save, "w"), indent=1)
    elif a.cmd == "calibrate":
        # NOTE: point --data-root at a HELD-OUT split (e.g. a 10% stratified split of the training set,
        # laid out as an ImageFolder 'test' directory); calibrating on training images overshoots the budget.
        model, cfg = load_checkpoint(a.ckpt)
        cfg.data_root = a.data_root or cfg.data_root
        dl = DataLoader(build_dataset(cfg, train=False), a.batch_size, num_workers=a.workers)
        r = collect(model, dl, K=0)
        f = refine_fraction(a.budget, model.seq_lens(0)[0], model.seq_lens()[1])
        eta = calibrate_eta(r["gconf"], f)
        print(f"budget {a.budget:.2f} GMACs -> refine {f * 100:.1f}% -> eta = {eta:.4f}")
    elif a.cmd == "bench":
        bench(a.num_classes)


if __name__ == "__main__":
    main()
