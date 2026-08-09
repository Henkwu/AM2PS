from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path
import yaml

VARIANTS = {
    "BL": dict(data_augmentation=False, multi_prompt=False, multi_scale=False, scalable_adapter=False),
    "BL_DA": dict(data_augmentation=True, multi_prompt=False, multi_scale=False, scalable_adapter=False),
    "BL_DA_MP": dict(data_augmentation=True, multi_prompt=True, multi_scale=False, scalable_adapter=False),
    "BL_DA_MS": dict(data_augmentation=True, multi_prompt=False, multi_scale=True, scalable_adapter=False),
    "BL_DA_MS_SA": dict(data_augmentation=True, multi_prompt=False, multi_scale=True, scalable_adapter=True),
    "FULL": dict(data_augmentation=True, multi_prompt=True, multi_scale=True, scalable_adapter=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", default="outputs/ablation")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    gen = Path(args.output_root) / "generated_configs"
    gen.mkdir(parents=True, exist_ok=True)
    for name, components in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg["components"] = components
        cfg_path = gen / f"{name}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        out = Path(args.output_root) / name
        cmd = ["python", "train.py", "--config", str(cfg_path), "--output", str(out), "--device", args.device]
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
