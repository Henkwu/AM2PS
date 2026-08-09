from __future__ import annotations

import argparse
import copy
import itertools
import subprocess
from pathlib import Path
import yaml

PROMPTS = [
    "a photo of a {class}",
    "a clinical photo of a {class}",
    "a chest X-ray photo of a {class}",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", default="outputs/prompt_sweep")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen = Path(args.output_root) / "generated_configs"
    gen.mkdir(parents=True, exist_ok=True)
    idx = 0
    for r in range(1, len(PROMPTS) + 1):
        for subset in itertools.combinations(PROMPTS, r):
            idx += 1
            cfg = copy.deepcopy(base)
            cfg.setdefault("model", {})["prompts"] = list(subset)
            cfg.setdefault("components", {})["multi_prompt"] = True
            cfg_path = gen / f"prompt_{idx}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            subprocess.run([
                "python", "train.py", "--config", str(cfg_path),
                "--output", str(Path(args.output_root) / f"prompt_{idx}"), "--device", args.device
            ], check=True)


if __name__ == "__main__":
    main()
