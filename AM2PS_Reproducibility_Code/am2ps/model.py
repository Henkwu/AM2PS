from __future__ import annotations

from typing import Sequence
import hashlib
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .prompts import build_prompts


class ScalableAdapter(nn.Module):
    """Two-layer bottleneck adapter with ReLU and residual use in AM2PS."""

    def __init__(self, dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossFusionLayer(nn.Module):
    """Bidirectional cross-fusion between backbone and residual-enhanced CLIP features."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.f_to_r = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.r_to_f = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_f1 = nn.LayerNorm(dim)
        self.norm_r1 = nn.LayerNorm(dim)
        self.norm_f2 = nn.LayerNorm(dim)
        self.norm_r2 = nn.LayerNorm(dim)
        hidden = 4 * dim
        self.ffn_f = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ffn_r = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, f: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Each global vector is represented as one token. Residual connections keep both streams active.
        fq, rq = f.unsqueeze(1), r.unsqueeze(1)
        f_cross, _ = self.f_to_r(fq, rq, rq, need_weights=False)
        r_cross, _ = self.r_to_f(rq, fq, fq, need_weights=False)
        f = self.norm_f1((fq + f_cross).squeeze(1))
        r = self.norm_r1((rq + r_cross).squeeze(1))
        f = self.norm_f2(f + self.ffn_f(f))
        r = self.norm_r2(r + self.ffn_r(r))
        return f, r


class MultiScaleFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        num_scales: int = 3,
        layers_per_scale: int = 3,
        num_heads: int = 8,
        alpha: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_scales = num_scales
        dims = [max(input_dim // (2**i), num_heads) for i in range(num_scales)]
        # Keep dimensions divisible by the number of heads.
        dims = [max(num_heads, (d // num_heads) * num_heads) for d in dims]
        self.scale_dims = dims
        self.f_proj = nn.ModuleList([nn.Linear(input_dim, d) for d in dims])
        self.r_proj = nn.ModuleList([nn.Linear(input_dim, d) for d in dims])
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([CrossFusionLayer(d, num_heads=num_heads) for _ in range(layers_per_scale)])
                for d in dims
            ]
        )
        self.up = nn.ModuleList([nn.Linear(d, input_dim) for d in dims])
        if alpha is None:
            alpha = [1.0 / num_scales] * num_scales
        if len(alpha) != num_scales:
            raise ValueError("alpha must have one weight per scale")
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha_tensor / alpha_tensor.sum())

    def forward(self, f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx in range(self.num_scales):
            fs = self.f_proj[idx](f)
            rs = self.r_proj[idx](r)
            for block in self.blocks[idx]:
                fs, rs = block(fs, rs)
            fused = 0.5 * (fs + rs)
            outputs.append(self.up[idx](fused))
        stacked = torch.stack(outputs, dim=1)  # B, L, D
        return (stacked * self.alpha.view(1, -1, 1)).sum(dim=1)


class DummyCLIP(nn.Module):
    """Offline smoke-test encoder; not intended for paper experiments."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.image_proj = nn.Linear(3, dim, bias=False)
        self.dim = dim
        for p in self.parameters():
            p.requires_grad = False

    def encode_image(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        pooled = x.mean(dim=(-2, -1))
        z = self.image_proj(pooled)
        return F.normalize(z, dim=-1) if normalize else z

    def encode_prompts(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        vectors = []
        for text in prompts:
            # deterministic pseudo text embedding for tests only
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vals = torch.tensor(list(digest), dtype=torch.float32, device=device)
            vals = vals.repeat((self.dim + len(vals) - 1) // len(vals))[: self.dim]
            vals = vals - vals.mean()
            vectors.append(F.normalize(vals, dim=0))
        return torch.stack(vectors, dim=0)


class AM2PS(nn.Module):
    def __init__(self, class_names: list[str], cfg: dict) -> None:
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        mcfg = cfg.get("model", {})
        ccfg = cfg.get("components", {})
        self.embed_dim = int(mcfg.get("embed_dim", 512))
        self.use_mp = bool(ccfg.get("multi_prompt", True))
        self.use_ms = bool(ccfg.get("multi_scale", True))
        self.use_sa = bool(ccfg.get("scalable_adapter", True))
        self.inter_modal_mode = str(mcfg.get("inter_modal_mode", "mean_class_probs"))
        self.normalize_fusion = bool(mcfg.get("normalize_probability_fusion", True))
        self.lambda1 = float(mcfg.get("lambda1", 1.0))
        self.lambda2 = float(mcfg.get("lambda2", 1.0))
        self.prompt_templates = list(mcfg.get("prompts", [
            "a photo of a {class}",
            "a clinical photo of a {class}",
            "a chest X-ray photo of a {class}",
        ]))

        # ResNet50 backbone from the manuscript, projected to the CLIP embedding dimension.
        weights = ResNet50_Weights.DEFAULT if bool(mcfg.get("resnet_pretrained", True)) else None
        net = resnet50(weights=weights)
        backbone_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.backbone_projector = nn.Linear(backbone_dim, self.embed_dim)
        self.baseline_classifier = nn.Linear(self.embed_dim, self.num_classes)

        self.clip_backend = str(mcfg.get("clip_backend", "open_clip"))
        if self.clip_backend == "open_clip":
            try:
                import open_clip
            except ImportError as e:
                raise ImportError("Install open_clip_torch to use the real CLIP encoder") from e
            model_name = str(mcfg.get("clip_model", "ViT-B-32"))
            pretrained = str(mcfg.get("clip_pretrained", "openai"))
            self.clip_model = open_clip.create_model(model_name, pretrained=pretrained)
            self.clip_tokenizer = open_clip.get_tokenizer(model_name)
            clip_dim = int(getattr(self.clip_model.visual, "output_dim", self.embed_dim))
        elif self.clip_backend == "dummy":
            self.clip_model = DummyCLIP(self.embed_dim)
            self.clip_tokenizer = None
            clip_dim = self.embed_dim
        else:
            raise ValueError(f"Unsupported clip_backend: {self.clip_backend}")

        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()
        self.clip_to_embed = nn.Identity() if clip_dim == self.embed_dim else nn.Linear(clip_dim, self.embed_dim)

        hidden = int(mcfg.get("adapter_hidden_dim", self.embed_dim // 2))
        self.adapter = ScalableAdapter(self.embed_dim, hidden)
        self.multi_scale = MultiScaleFusion(
            input_dim=self.embed_dim,
            num_scales=int(mcfg.get("num_scales", 3)),
            layers_per_scale=int(mcfg.get("transformer_layers", 3)),
            num_heads=int(mcfg.get("num_heads", 8)),
            alpha=mcfg.get("alpha"),
        )
        self.intra_classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.register_buffer("text_features", torch.empty(0), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        # CLIP is frozen in the manuscript.
        self.clip_model.eval()
        return self

    @torch.no_grad()
    def prepare_text_features(self, device: torch.device) -> None:
        if self.text_features.numel() > 0 and self.text_features.device == device:
            return
        grouped = build_prompts(self.class_names, self.prompt_templates)
        flat = [p for group in grouped for p in group]
        if self.clip_backend == "open_clip":
            tokens = self.clip_tokenizer(flat).to(device)
            z = self.clip_model.encode_text(tokens, normalize=True)
        else:
            z = self.clip_model.encode_prompts(flat, device)
        z = z.view(self.num_classes, len(self.prompt_templates), -1)
        if z.shape[-1] != self.embed_dim:
            z = self.clip_to_embed(z)
        self.text_features = F.normalize(z.float(), dim=-1)

    def _inter_modal_probs(self, f: torch.Tensor) -> torch.Tensor:
        self.prepare_text_features(f.device)
        f_norm = F.normalize(f, dim=-1)
        sim = torch.einsum("bd,ckd->bck", f_norm, self.text_features)
        temperature = 100.0

        if self.inter_modal_mode == "mean_class_probs":
            # Practical interpretation: class-normalize each prompt prediction, then ensemble prompts.
            per_prompt_probs = torch.softmax(temperature * sim, dim=1)
            return per_prompt_probs.mean(dim=2)
        if self.inter_modal_mode == "class_prototype":
            class_text = F.normalize(self.text_features.mean(dim=1), dim=-1)
            logits = temperature * (f_norm @ class_text.T)
            return torch.softmax(logits, dim=1)
        if self.inter_modal_mode == "paper_literal":
            warnings.warn(
                "paper_literal normalizes over prompts and then averages them. This yields 1/K per class "
                "mathematically and is included only to expose the equation-level ambiguity.",
                RuntimeWarning,
            )
            return torch.softmax(sim, dim=2).mean(dim=2)
        raise ValueError(f"Unknown inter_modal_mode={self.inter_modal_mode}")

    def _clip_visual(self, clip_x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            v = self.clip_model.encode_image(clip_x, normalize=True)
        v = self.clip_to_embed(v.float())
        return v

    def forward(self, backbone_x: torch.Tensor, clip_x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.backbone_projector(self.backbone(backbone_x))
        p_base = torch.softmax(self.baseline_classifier(f), dim=1)

        p_inter = self._inter_modal_probs(f) if self.use_mp else None
        p_intra = None
        v = None
        r = None
        fused_feature = f
        if self.use_ms or self.use_sa:
            v = self._clip_visual(clip_x)
            r = v + self.adapter(v) if self.use_sa else v
        if self.use_ms:
            fused_feature = self.multi_scale(f, r)
            p_intra = torch.softmax(self.intra_classifier(fused_feature), dim=1)
        elif self.use_sa:
            # Adapter-only fallback: combine task-specific and adapted CLIP representations.
            fused_feature = 0.5 * (f + r)
            p_intra = torch.softmax(self.intra_classifier(fused_feature), dim=1)

        if p_inter is not None and p_intra is not None:
            final = self.lambda1 * p_inter + self.lambda2 * p_intra
            if self.normalize_fusion:
                final = final / max(self.lambda1 + self.lambda2, 1e-12)
        elif p_inter is not None:
            # BL+DA+MP ablation: retain the baseline visual probability as the second stream.
            final = self.lambda1 * p_inter + self.lambda2 * p_base
            if self.normalize_fusion:
                final = final / max(self.lambda1 + self.lambda2, 1e-12)
        elif p_intra is not None:
            final = p_intra
        else:
            final = p_base

        final = final.clamp_min(1e-8)
        final = final / final.sum(dim=1, keepdim=True)
        return {
            "probs": final,
            "p_base": p_base,
            "p_inter": p_inter if p_inter is not None else torch.empty(0, device=final.device),
            "p_intra": p_intra if p_intra is not None else torch.empty(0, device=final.device),
            "backbone_feature": f,
            "clip_feature": v if v is not None else torch.empty(0, device=final.device),
            "fused_feature": fused_feature,
        }
