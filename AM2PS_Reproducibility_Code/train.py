from __future__ import annotations

import argparse
from pathlib import Path
import torch

from am2ps.config import load_config
from am2ps.data import make_dataloaders
from am2ps.engine import fit, evaluate
from am2ps.model import AM2PS
from am2ps.utils import seed_everything, resolve_device, save_json, count_parameters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    data = make_dataloaders(cfg, seed=seed)
    model = AM2PS(data.class_names, cfg).to(device)

    output = Path(args.output or cfg.get("output_dir", "outputs/run"))
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")
    save_json({"class_names": data.class_names, **count_parameters(model)}, output / "model_info.json")

    summary = fit(model, data, cfg, device, output)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    test_metrics, arrays = evaluate(model, data.test, device)
    save_json(summary, output / "training_summary.json")
    save_json(test_metrics, output / "test_metrics.json")
    print("test:", test_metrics)


if __name__ == "__main__":
    main()
