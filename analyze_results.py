#!/usr/bin/env python3
"""
Analyze JSON result files under baseline_results/ and compute percentages of three error types per file:

1. non_retrieval: compressed_context empty or whitespace
2. fail_retrieval: compressed_context not empty but none of ground-truth strings appear in compressed_context
3. generation_error: at least one ground-truth appears in compressed_context but none appear in the prediction

Outputs:
- analysis_summary.csv
- analysis_summary.json
- error_percentages_per_file.png
- error_percentages_by_folder.png

Usage:
    python analyze_results.py [--base-dir .]

Run from the `baseline_results/` folder or provide --base-dir to point to it.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set(style='whitegrid')


def first_existing(d: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for k in candidates:
        if k in d:
            return d[k]
    return None


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i is not None]
    if isinstance(x, dict):
        # attempt to extract common list-valued keys
        for candidate in ['answers', 'ground_truths', 'references', 'targets']:
            if candidate in x and isinstance(x[candidate], list):
                return [str(i) for i in x[candidate] if i is not None]
        return [json.dumps(x, ensure_ascii=False)]
    return [str(x)]


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ''
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def contains_any_text(haystack: str, needles: List[str]) -> bool:
    if not haystack:
        return False
    hay = normalize_text(haystack)
    for n in needles:
        if not n:
            continue
        if normalize_text(n) in hay:
            return True
    return False


def extract_fields(entry: Dict[str, Any]) -> Tuple[str, List[str], str, Optional[Any]]:
    cc = first_existing(entry, ['compressed_context', 'compressed_contexts', 'context', 'compressed'])
    pred = first_existing(entry, ['prediction', 'pred', 'answer', 'generated', 'generated_text', 'output'])
    gts = first_existing(entry, ['ground_truths', 'ground_truth', 'answers', 'gold', 'targets', 'reference'])

    # Try to extract an exact-match (EM) score or flag if present
    em = first_existing(entry, ['em', 'exact_match', 'exact_match_score', 'em_score', 'metrics'])
    # If metrics is a dict, try to find em inside it
    if isinstance(em, dict):
        for k in ['em', 'exact_match', 'exact_match_score', 'em_score']:
            if k in em:
                em = em[k]
                break

    if isinstance(gts, dict) and 'answers' in gts:
        gts = gts['answers']

    return (cc if cc is not None else '', as_list(gts), pred if pred is not None else '', em)


def classify_entry(entry: Dict[str, Any]) -> str:
    cc_raw, gts_raw, pred_raw, em_raw = extract_fields(entry)
    cc = normalize_text(cc_raw)
    gts = [normalize_text(g) for g in gts_raw if g is not None and str(g).strip() != '']
    pred = normalize_text(pred_raw)

    # If EM is present and equals 1.0 (or >=1.0), treat as correct regardless of retrieved context
    em_val = None
    try:
        if em_raw is not None:
            em_val = float(em_raw)
    except Exception:
        # leave em_val as None
        em_val = None

    if em_val is not None and em_val >= 1.0:
        return 'correct'

    if not cc:
        return 'non_retrieval'
    if len(gts) == 0:
        return 'bad_format'
    if not any(gt in cc for gt in gts):
        return 'fail_retrieval'
    else:
        return 'generation_error'
    # print(em_val)
    # print(entry)
    # return 'correct'


def discover_json_files(base_dir: Path) -> List[Path]:
    # exclude files whose filename contains 'summary' (case-insensitive)
    files = sorted([p for p in base_dir.rglob('*.json') if p.is_file() and 'summary' not in p.name.lower() and 'musique' not in p.name.lower()])
    return files


def process_file(path: Path) -> Optional[Dict[str, Any]]:
    counts = {'total': 0, 'non_retrieval': 0, 'fail_retrieval': 0, 'generation_error': 0, 'correct': 0, 'bad_format': 0}
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning('Failed to parse %s: %s', path, e)
        return None

    # Only analyze the 'results' part of each JSON file. If missing or not a list, skip.
    entries = None
    if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
        entries = data['results']
    else:
        logger.warning("File %s does not contain a 'results' list; skipping", path)
        return None

    for e in entries:
        if not isinstance(e, dict):
            continue
        counts['total'] += 1
        cls = classify_entry(e)
        if cls not in counts:
            counts.setdefault(cls, 0)
        counts[cls] += 1

    return counts


def aggregate_results(json_files: List[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    detailed = {}
    for p in tqdm(json_files, desc='Processing files'):
        counts = process_file(p)
        if counts is None:
            continue
        folder = p.parent.name
        fname = p.name
        row = {'file_path': str(p), 'folder': folder, 'file_name': fname, **counts}
        rows.append(row)
        detailed[str(p)] = counts

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['file_path', 'folder', 'file_name', 'total'])
    df = df.fillna(0)
    for col in ['non_retrieval', 'fail_retrieval', 'generation_error', 'correct', 'bad_format']:
        df[col + '_pct'] = (df[col] / df['total']).fillna(0) * 100
    return df, detailed


def plot_results(df: pd.DataFrame, base_dir: Path) -> None:
    if df.empty:
        logger.info('No data to plot')
        return
    plot_df = df.copy()
    plot_df = plot_df.sort_values('total', ascending=False)
    labels = plot_df['file_name'] + '\n' + plot_df['folder']
    x = np.arange(len(plot_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.5)))
    ax.bar(x - 1.5 * width, plot_df['non_retrieval_pct'], width, label='non_retrieval')
    ax.bar(x - 0.5 * width, plot_df['fail_retrieval_pct'], width, label='fail_retrieval')
    ax.bar(x + 0.5 * width, plot_df['generation_error_pct'], width, label='generation_error')
    ax.bar(x + 1.5 * width, plot_df['correct_pct'], width, label='correct')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Error percentages per file')
    ax.legend()
    plt.tight_layout()
    fig_file = base_dir / 'error_percentages_per_file.png'
    fig.savefig(fig_file, dpi=200)
    logger.info('Saved plot to %s', fig_file)

    avg_df = df.groupby('folder')[[ 'non_retrieval_pct','fail_retrieval_pct','generation_error_pct','correct_pct']].mean().reset_index()
    avg_df_m = avg_df.melt(id_vars='folder', var_name='metric', value_name='pct')
    plt.figure(figsize=(8,4))
    sns.barplot(data=avg_df_m, x='folder', y='pct', hue='metric')
    plt.title('Average error percentages by models')
    plt.xlabel('Model')
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    avg_fig_file = base_dir / 'error_percentages_by_folder.png'
    plt.savefig(avg_fig_file, dpi=200)
    logger.info('Saved plot to %s', avg_fig_file)


def save_summary(df: pd.DataFrame, base_dir: Path) -> None:
    if df.empty:
        logger.info('No summary to save')
        return
    out_csv = base_dir / 'analysis_summary.csv'
    out_json = base_dir / 'analysis_summary.json'
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient='records', indent=2)
    logger.info('Wrote %s and %s', out_csv, out_json)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Analyze baseline_results JSON files for retrieval/generation errors')
    parser.add_argument('--base-dir', type=str, default='./outputs/baseline_results', help='Base directory to scan for JSON files (path to baseline_results)')
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error('Base directory %s not found', base_dir)
        return 2

    json_files = discover_json_files(base_dir)
    logger.info('Found %d JSON files under %s', len(json_files), base_dir)
    if len(json_files) == 0:
        return 0

    df, detailed = aggregate_results(json_files)
    if df.empty:
        logger.info('No results processed')
        return 0

    # round percentage columns
    pct_cols = [c for c in df.columns if c.endswith('_pct')]
    df[pct_cols] = df[pct_cols].round(2)

    save_summary(df, base_dir)
    plot_results(df, base_dir)

    # print concise textual summary
    for _, row in df.sort_values(['folder','file_name']).iterrows():
        logger.info("%s/%s: total=%d, non_retrieval=%d (%.2f%%), fail_retrieval=%d (%.2f%%), generation_error=%d (%.2f%%), correct=%d (%.2f%%), bad_format=%d",
                    row['folder'], row['file_name'], int(row['total']), int(row['non_retrieval']), float(row['non_retrieval_pct']),
                    int(row['fail_retrieval']), float(row['fail_retrieval_pct']), int(row['generation_error']), float(row['generation_error_pct']),
                    int(row['correct']), float(row['correct_pct']), int(row.get('bad_format', 0)))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
