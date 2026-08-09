from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path
import yaml

PAIRS = [(1.0, x) for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]] + [(x, 1.0) for x in [0.8, 0.6, 0.4, 0.2, 0.0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", default="outputs/lambda_sweep")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen = Path(args.output_root) / "generated_configs"
    gen.mkdir(parents=True, exist_ok=True)
    for l1, l2 in PAIRS:
        cfg = copy.deepcopy(base)
        cfg.setdefault("model", {})["lambda1"] = l1
        cfg["model"]["lambda2"] = l2
        tag = f"lambda_{l1:.1f}_{l2:.1f}"
        cfg_path = gen / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        subprocess.run([
            "python", "train.py", "--config", str(cfg_path),
            "--output", str(Path(args.output_root) / tag), "--device", args.device
        ], check=True)


if __name__ == "__main__":
    main()
