#!/usr/bin/env python3
"""
Aggregate per-dataset summary metrics from JSON files in a folder.

For each input JSON file that has the structure:
{
  "dataset_name": {
    "exact_match": ...,
    "f1": ...,
    "latency": ...,
    "num_samples": ...,
    "avg_original_tokens": ...,
    "avg_compressed_tokens": ...,
    "token_compression_ratio": ...,
    "char_compression_ratio": ...
  },
  ...
}

This script computes, per dataset:
- files_seen (count of JSON files containing that dataset)
- mean_* (simple mean across files)
- total_num_samples (sum across files)

Output: JSON (default aggregate_summary.json) and CSV.
Usage:
    python aggregate_summaries.py --input-dir /path/to/folder --output-file aggregate_summary.json
"""
from pathlib import Path
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any

def aggregate_folder(input_dir: Path) -> Dict[str, Any]:
    """
    Collect per-file dataset summaries. For each JSON file under input_dir, read its top-level
    mapping and record dataset summaries (only the NUMERIC_KEYS).

    Return structure: { "relative/path/file.json": { "dataset_name": { key: value, ... }, ... }, ... }
    """
    out: Dict[str, Any] = {}
    for p in sorted(input_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # skip unreadable files
            continue
        if not isinstance(data, dict):
            continue
        dataset = str(p.relative_to(input_dir)).split("_")[0]
        metrics = data.get("metrics")
        if metrics:
            out[dataset] = {}
            out[dataset]["metrics"] = metrics
    return out


def write_outputs(agg: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=Path, default=Path("."), help="Folder containing JSON summary files")
    parser.add_argument("--output-file", "-o", type=Path, default=Path("aggregate_summary.json"))
    args = parser.parse_args()

    agg = aggregate_folder(args.input_dir)
    write_outputs(agg, args.output_file)
    print(f"Wrote aggregate summary to {args.output_file}")


if __name__ == "__main__":
    main()