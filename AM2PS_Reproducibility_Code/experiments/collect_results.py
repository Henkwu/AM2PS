from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--output", default="results.csv")
    args = ap.parse_args()
    rows = []
    for p in Path(args.root).rglob("test_metrics.json"):
        metrics = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"run": str(p.parent), **metrics})
    df = pd.DataFrame(rows).sort_values("run") if rows else pd.DataFrame()
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False) if len(df) else "No results found")


if __name__ == "__main__":
    main()
