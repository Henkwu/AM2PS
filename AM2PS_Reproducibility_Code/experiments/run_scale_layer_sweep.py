from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", default="outputs/scale_layer_sweep")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen = Path(args.output_root) / "generated_configs"
    gen.mkdir(parents=True, exist_ok=True)
    for scales in [1, 2, 3]:
        for layers in [1, 2, 3, 4]:
            cfg = copy.deepcopy(base)
            cfg.setdefault("model", {})["num_scales"] = scales
            cfg["model"]["transformer_layers"] = layers
            cfg["model"]["alpha"] = [1.0 / scales] * scales
            tag = f"s{scales}_l{layers}"
            cfg_path = gen / f"{tag}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            subprocess.run([
                "python", "train.py", "--config", str(cfg_path),
                "--output", str(Path(args.output_root) / tag), "--device", args.device
            ], check=True)


if __name__ == "__main__":
    main()
