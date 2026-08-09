from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_split(path: Path):
    rows = []
    for class_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        count = sum(1 for f in class_dir.rglob("*") if f.suffix.lower() in VALID_EXT)
        rows.append((class_dir.name, count))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    for split in ["train", "val", "test"]:
        p = root / split
        if not p.exists():
            print(f"{split}: MISSING")
            continue
        rows = count_split(p)
        print(f"{split}:")
        for cls, n in rows:
            print(f"  {cls:20s} {n:6d}")


if __name__ == "__main__":
    main()
