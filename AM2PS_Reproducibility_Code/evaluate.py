from __future__ import annotations

import argparse
import torch

from am2ps.config import load_config
from am2ps.data import make_dataloaders
from am2ps.engine import evaluate
from am2ps.model import AM2PS
from am2ps.utils import resolve_device, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    data = make_dataloaders(cfg, seed=int(cfg.get("seed", 42)))
    model = AM2PS(data.class_names, cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    metrics, _ = evaluate(model, data.test, device)
    print(metrics)
    if args.save:
        save_json(metrics, args.save)


if __name__ == "__main__":
    main()
