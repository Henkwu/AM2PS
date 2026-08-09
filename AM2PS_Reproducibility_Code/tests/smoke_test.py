from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from am2ps.model import AM2PS


def main():
    cfg = {
        "model": {
            "embed_dim": 64,
            "adapter_hidden_dim": 32,
            "clip_backend": "dummy",
            "resnet_pretrained": False,
            "num_scales": 3,
            "transformer_layers": 2,
            "num_heads": 8,
            "alpha": [1/3, 1/3, 1/3],
            "lambda1": 1.0,
            "lambda2": 1.0,
            "inter_modal_mode": "mean_class_probs",
        },
        "components": {
            "data_augmentation": True,
            "multi_prompt": True,
            "multi_scale": True,
            "scalable_adapter": True,
        },
    }
    model = AM2PS(["NORMAL", "PNEUMONIA"], cfg)
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    out = model(x1, x2)
    assert out["probs"].shape == (2, 2)
    assert torch.allclose(out["probs"].sum(dim=1), torch.ones(2), atol=1e-5)
    loss = -out["probs"][:, 0].clamp_min(1e-8).log().mean()
    loss.backward()
    print("SMOKE TEST PASSED")
    print("probabilities:", out["probs"].detach())


if __name__ == "__main__":
    main()
