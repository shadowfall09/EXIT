#!/usr/bin/env python3
"""
Merge results from parallel test runs
"""

import json
import os
import sys

def merge_results(output_dir: str):
    """Merge individual task results into a single summary."""
    
    summary = {}
    
    # List of expected tasks
    tasks = ["HotpotQA", "2wikimultihop", "musique", "NQ", "TQA"]
    
    print(f"Merging results from: {output_dir}")
    print(f"Looking for tasks: {tasks}")
    print("")
    
    # Check for existing task results
    for task in tasks:
        result_file = os.path.join(output_dir, f"{task}_results.json")
        if os.path.exists(result_file):
            print(f"✓ Found: {task}_results.json")
            with open(result_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
                if 'metrics' in task_data:
                    summary[task] = task_data['metrics']
        else:
            print(f"✗ Missing: {task}_results.json")
    
    # Save merged summary
    if summary:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Summary saved to: {summary_path}")
        print(f"\nMerged {len(summary)}/{len(tasks)} tasks")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for task, metrics in summary.items():
            print(f"\n{task}:")
            print(f"  EM: {metrics.get('exact_match', 0):.4f}")
            print(f"  F1: {metrics.get('f1', 0):.4f}")
            print(f"  Latency: {metrics.get('latency', 0):.4f}s")
            if 'token_compression_ratio' in metrics:
                print(f"  Token Compression: {metrics['token_compression_ratio']:.4f}")
        
        return True
    else:
        print("\n✗ No results found to merge")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_results.py <output_directory>")
        print("\nExample:")
        print("  python merge_results.py ./outputs/baseline_results/reproduction")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory does not exist: {output_dir}")
        sys.exit(1)
    
    success = merge_results(output_dir)
    sys.exit(0 if success else 1)
