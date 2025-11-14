#!/usr/bin/env python3
"""
Baseline Testing Script for EXIT RAG Pipeline
Tests the EXIT RAG system on multiple datasets and calculates EM, F1, and latency metrics.
"""

import json
import time
import argparse
import os
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from exit_rag import ExitRAG, Document

def normalize_answer(s: str) -> str:
    """Normalize answer text for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Calculate exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    """Calculate maximum metric over all ground truths."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def load_dataset(data_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    print(f"Loading dataset from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def prepare_documents(item: Dict, top_k: int = None) -> List[Document]:
    """Convert dataset item contexts to Document objects."""
    documents = []
    ctxs = item.get('ctxs', [])
    
    # Limit to top_k documents if specified
    if top_k is not None and top_k > 0:
        ctxs = ctxs[:top_k]
    
    for ctx in ctxs:
        doc = Document(
            title=ctx.get('title', ''),
            text=ctx.get('text', ''),
            score=float(ctx.get('score', 1.0))
        )
        documents.append(doc)
    return documents

def evaluate_sample(
    rag: ExitRAG,
    question: str,
    documents: List[Document],
    ground_truths: List[str],
    compression_threshold: float = 0.5,
    tokenizer = None
) -> Dict[str, Any]:
    """Evaluate a single sample."""
    start_time = time.time()
    
    # Calculate original context
    original_context = ' '.join([d.text for d in documents])
    
    # Run RAG pipeline
    result = rag.run_rag(
        query=question,
        documents=documents,
        compression_threshold=compression_threshold
    )
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Calculate metrics
    prediction = result['answer']
    em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    
    # Calculate token counts and compression ratio
    compressed_context = result.get('compressed_context', '')
    
    if tokenizer is not None:
        original_tokens = len(tokenizer.encode(original_context, add_special_tokens=False))
        compressed_tokens = len(tokenizer.encode(compressed_context, add_special_tokens=False))
        token_compression_ratio = compressed_tokens / max(original_tokens, 1)
    else:
        original_tokens = 0
        compressed_tokens = 0
        token_compression_ratio = 0.0
    
    char_compression_ratio = len(compressed_context) / max(len(original_context), 1)
    
    return {
        'question': question,
        'prediction': prediction,
        'ground_truths': ground_truths,
        'em': float(em),
        'f1': float(f1),
        'latency': latency,
        'compressed_context': compressed_context,
        'original_tokens': original_tokens,
        'compressed_tokens': compressed_tokens,
        'token_compression_ratio': token_compression_ratio,
        'char_compression_ratio': char_compression_ratio
    }

def run_evaluation(
    rag: ExitRAG,
    dataset: List[Dict],
    max_samples: int = None,
    compression_threshold: float = 0.5,
    tokenizer = None,
    top_k: int = None
) -> Dict[str, Any]:
    """Run evaluation on entire dataset."""
    results = []
    total_em = 0
    total_f1 = 0
    total_latency = 0
    total_original_tokens = 0
    total_compressed_tokens = 0
    total_token_compression_ratio = 0
    total_char_compression_ratio = 0
    
    samples_to_process = dataset[:max_samples] if max_samples else dataset
    
    print(f"Processing {len(samples_to_process)} samples...")
    if top_k is not None:
        print(f"Using top {top_k} documents per sample")
    
    for item in tqdm(samples_to_process):
        # Extract question and answers
        question = item.get('question', '')
        
        # Handle different answer formats
        if 'answers' in item:
            answers = item['answers'] if isinstance(item['answers'], list) else [item['answers']]
        elif 'answer' in item:
            answers = [item['answer']] if isinstance(item['answer'], str) else item['answer']
        else:
            print(f"Warning: No answer found for question: {question}")
            continue
        
        # Prepare documents
        documents = prepare_documents(item, top_k=top_k)
        
        if not documents:
            print(f"Warning: No documents found for question: {question}")
            continue
        
        try:
            # Evaluate sample
            # print("---- DEBUG ----")
            # print(f"Question: {question}")
            # print(f"Ground Truths: {answers}")
            # print(f"Documents: {documents}")
            
            sample_result = evaluate_sample(
                rag=rag,
                question=question,
                documents=documents,
                ground_truths=answers,
                compression_threshold=compression_threshold,
                tokenizer=tokenizer
            )
            
            results.append(sample_result)
            total_em += sample_result['em']
            total_f1 += sample_result['f1']
            total_latency += sample_result['latency']
            total_original_tokens += sample_result['original_tokens']
            total_compressed_tokens += sample_result['compressed_tokens']
            total_token_compression_ratio += sample_result['token_compression_ratio']
            total_char_compression_ratio += sample_result['char_compression_ratio']
            
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")
            continue
    
    # Calculate average metrics
    num_samples = len(results)
    avg_em = total_em / num_samples if num_samples > 0 else 0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0
    avg_latency = total_latency / num_samples if num_samples > 0 else 0
    avg_original_tokens = total_original_tokens / num_samples if num_samples > 0 else 0
    avg_compressed_tokens = total_compressed_tokens / num_samples if num_samples > 0 else 0
    avg_token_compression_ratio = total_token_compression_ratio / num_samples if num_samples > 0 else 0
    avg_char_compression_ratio = total_char_compression_ratio / num_samples if num_samples > 0 else 0
    
    return {
        'results': results,
        'metrics': {
            'exact_match': avg_em,
            'f1': avg_f1,
            'latency': avg_latency,
            'num_samples': num_samples,
            'avg_original_tokens': avg_original_tokens,
            'avg_compressed_tokens': avg_compressed_tokens,
            'token_compression_ratio': avg_token_compression_ratio,
            'char_compression_ratio': avg_char_compression_ratio
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Test EXIT RAG on multiple datasets")
    
    # Model arguments
    parser.add_argument(
        "--compression_model",
        type=str,
        default="/mnt/data2/yichengtao/EXIT/outputs/exit_model/final_model",
        help="Path to compression model"
    )
    parser.add_argument(
        "--reader_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to reader model"
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="google/gemma-2b-it",
        help="Path to retriever model"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/data2/yichengtao/data/retrieval",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=["HotpotQA", "2wikimultihop", "musique", "NQ", "TQA"],
        help="Tasks to evaluate"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="contriever-msmarco",
        help="Retriever method"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per task"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top documents to use per sample (default: use all documents)"
    )
    
    # Compression arguments
    parser.add_argument(
        "--compression_threshold",
        type=float,
        default=0.5,
        help="Compression threshold"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/baseline_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer for token counting
    print("Loading tokenizer for token counting...")
    tokenizer = AutoTokenizer.from_pretrained(args.reader_model)
    
    # Initialize RAG pipeline
    print("Initializing EXIT RAG pipeline...")
    rag = ExitRAG(
        retriever_model=args.retriever_model,
        compression_model=args.compression_model,
        reader_model=args.reader_model
    )
    
    # Store all results
    all_results = {}
    summary = {}
    
    # Evaluate each task
    for task in args.tasks:
        print(f"\n{'='*50}")
        print(f"Evaluating task: {task}")
        print(f"{'='*50}\n")
        
        # Determine split
        split = "test" if task == "TQA" else "dev"
        
        # Construct data path
        data_path = os.path.join(
            args.data_root,
            f"{args.retriever}_{task}",
            f"{split}.json"
        )
        
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found: {data_path}")
            continue
        
        # Load dataset
        dataset = load_dataset(data_path)
        
        # Run evaluation
        task_results = run_evaluation(
            rag=rag,
            dataset=dataset,
            max_samples=args.max_samples,
            compression_threshold=args.compression_threshold,
            tokenizer=tokenizer,
            top_k=args.top_k
        )
        
        # Store results
        all_results[task] = task_results
        summary[task] = task_results['metrics']
        
        # Print task summary
        print(f"\nTask: {task}")
        print(f"  Exact Match: {task_results['metrics']['exact_match']:.4f}")
        print(f"  F1 Score: {task_results['metrics']['f1']:.4f}")
        print(f"  Latency (s): {task_results['metrics']['latency']:.4f}")
        print(f"  Num Samples: {task_results['metrics']['num_samples']}")
        print(f"  Avg Original Tokens: {task_results['metrics']['avg_original_tokens']:.2f}")
        print(f"  Avg Compressed Tokens: {task_results['metrics']['avg_compressed_tokens']:.2f}")
        print(f"  Token Compression Ratio: {task_results['metrics']['token_compression_ratio']:.4f}")
        print(f"  Char Compression Ratio: {task_results['metrics']['char_compression_ratio']:.4f}")
        
        # Save task-specific results
        task_output_path = os.path.join(args.output_dir, f"{task}_results.json")
        with open(task_output_path, 'w', encoding='utf-8') as f:
            json.dump(task_results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to: {task_output_path}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print overall summary
    print(f"\n{'='*50}")
    print("OVERALL SUMMARY")
    print(f"{'='*50}\n")
    
    for task, metrics in summary.items():
        print(f"{task}:")
        print(f"  EM: {metrics['exact_match']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Latency: {metrics['latency']:.4f}s")
        print(f"  Token Compression: {metrics['token_compression_ratio']:.4f} ({metrics['avg_compressed_tokens']:.0f}/{metrics['avg_original_tokens']:.0f})")
        print()
    
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
