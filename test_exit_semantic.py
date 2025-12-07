#!/usr/bin/env python3
"""
Test script for EXIT-Semantic RAG Pipeline
Tests the EXIT-Semantic hybrid compression method on multiple datasets.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressors import EXITSemanticCompressor, SearchResult


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


def prepare_documents(item: Dict, top_k: int = None) -> List[SearchResult]:
    """Convert dataset item contexts to SearchResult objects."""
    documents = []
    ctxs = item.get('ctxs', [])
    
    # Limit to top_k documents if specified
    if top_k is not None and top_k > 0:
        ctxs = ctxs[:top_k]
    
    for idx, ctx in enumerate(ctxs):
        doc = SearchResult(
            evi_id=idx,
            docid=idx,
            title=ctx.get('title', ''),
            text=ctx.get('text', ''),
            score=float(ctx.get('score', 1.0))
        )
        documents.append(doc)
    return documents


def generate_answer(
    reader_model,
    reader_tokenizer,
    query: str,
    context: str,
    device: str = "cuda"
) -> Tuple[str, float]:
    """Generate answer using compressed context."""
    start_time = time.time()
    
    # Get actual device from model (in case device_map="auto" placed it elsewhere)
    model_device = next(reader_model.parameters()).device
    
    # Format prompt
    chat = [{
        "role": "system",
        "content": f"Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query. Do not provide any explanation."
    },
    {
        "role": "user",
        "content": f"Query: {query}\nAnswer: "
    }]
    
    prompt = reader_tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate answer
    inputs = reader_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model_device)
    
    with torch.no_grad():
        outputs = reader_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            pad_token_id=reader_tokenizer.eos_token_id,
            temperature=0.01,
            top_p=1.0
        )
        
    answer = reader_tokenizer.decode(
        outputs[0][inputs.input_ids.size(1):],
        skip_special_tokens=True
    ).strip()
    
    generation_time = time.time() - start_time
    
    return answer, generation_time


def evaluate_sample(
    compressor: EXITSemanticCompressor,
    reader_model,
    reader_tokenizer,
    question: str,
    documents: List[SearchResult],
    ground_truths: List[str],
    tokenizer=None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Evaluate a single sample."""
    start_time = time.time()
    
    # Calculate original context
    original_context = ' '.join([d.text for d in documents])
    
    # Record timing stats before compression
    prev_count = compressor.timing_stats['count'] if hasattr(compressor, 'timing_stats') else 0
    prev_stats = {}
    if hasattr(compressor, 'timing_stats'):
        prev_stats = {k: v for k, v in compressor.timing_stats.items()}
    
    # Compress documents
    compressed_docs = compressor.compress(question, documents)
    compressed_context = compressed_docs[0].text if compressed_docs else ""
    
    # Generate answer
    answer, generation_time = generate_answer(
        reader_model,
        reader_tokenizer,
        question,
        compressed_context,
        device
    )
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Calculate metrics
    prediction = answer
    em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    
    # Calculate token counts and compression ratio
    if tokenizer is not None:
        original_tokens = len(tokenizer.encode(original_context, add_special_tokens=False))
        compressed_tokens = len(tokenizer.encode(compressed_context, add_special_tokens=False))
        token_compression_ratio = compressed_tokens / max(original_tokens, 1)
    else:
        original_tokens = 0
        compressed_tokens = 0
        token_compression_ratio = 0.0
    
    char_compression_ratio = len(compressed_context) / max(len(original_context), 1)
    
    # Calculate timing for this sample only (delta from previous)
    timing_info = {
        'generation_time': generation_time
    }
    if hasattr(compressor, 'timing_stats') and compressor.timing_stats['count'] > prev_count:
        timing_info.update({
            'sentence_split': compressor.timing_stats['sentence_split'] - prev_stats.get('sentence_split', 0),
            'hyde_generation': compressor.timing_stats['hyde_generation'] - prev_stats.get('hyde_generation', 0),
            'query_encoding': compressor.timing_stats['query_encoding'] - prev_stats.get('query_encoding', 0),
            'doc_encoding': compressor.timing_stats['doc_encoding'] - prev_stats.get('doc_encoding', 0),
            'similarity_compute': compressor.timing_stats['similarity_compute'] - prev_stats.get('similarity_compute', 0),
            'filtering': compressor.timing_stats['filtering'] - prev_stats.get('filtering', 0),
            'exit_inference': compressor.timing_stats['exit_inference'] - prev_stats.get('exit_inference', 0),
            'total_compress': compressor.timing_stats['total_compress'] - prev_stats.get('total_compress', 0)
        })
    
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
        'char_compression_ratio': char_compression_ratio,
        'timing': timing_info
    }


def run_evaluation(
    compressor: EXITSemanticCompressor,
    reader_model,
    reader_tokenizer,
    dataset: List[Dict],
    max_samples: int = None,
    tokenizer=None,
    top_k: int = None,
    device: str = "cuda"
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
    total_timing = {
        'sentence_split': 0.0,
        'hyde_generation': 0.0,
        'query_encoding': 0.0,
        'doc_encoding': 0.0,
        'similarity_compute': 0.0,
        'filtering': 0.0,
        'exit_inference': 0.0,
        'total_compress': 0.0,
        'generation_time': 0.0
    }
    
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
            sample_result = evaluate_sample(
                compressor=compressor,
                reader_model=reader_model,
                reader_tokenizer=reader_tokenizer,
                question=question,
                documents=documents,
                ground_truths=answers,
                tokenizer=tokenizer,
                device=device
            )
            
            results.append(sample_result)
            total_em += sample_result['em']
            total_f1 += sample_result['f1']
            total_latency += sample_result['latency']
            total_original_tokens += sample_result['original_tokens']
            total_compressed_tokens += sample_result['compressed_tokens']
            total_token_compression_ratio += sample_result['token_compression_ratio']
            total_char_compression_ratio += sample_result['char_compression_ratio']
            
            # Accumulate timing info
            if 'timing' in sample_result:
                for key in total_timing.keys():
                    total_timing[key] += sample_result['timing'].get(key, 0.0)
            
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
    
    # Calculate average timing
    avg_timing = {}
    if num_samples > 0:
        for key, value in total_timing.items():
            avg_timing[key] = value / num_samples
    
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
            'char_compression_ratio': avg_char_compression_ratio,
            'avg_timing': avg_timing,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Test EXIT-Semantic RAG")
    
    # Model arguments
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model for semantic filtering"
    )
    parser.add_argument(
        "--compression_model",
        type=str,
        default=None,
        help="Path to trained EXIT checkpoint for compression"
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="google/gemma-2b-it",
        help="Base model for EXIT (retriever model)"
    )
    parser.add_argument(
        "--reader_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to reader model"
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
        help="Number of top documents to use per sample"
    )
    
    # EXIT-Semantic arguments
    parser.add_argument(
        "--semantic_filter_ratio_context",
        type=float,
        default=0.5,
        help="Ratio of sentences to keep for context generation (default: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--semantic_filter_ratio_relevance",
        type=float,
        default=0.5,
        help="Ratio of sentences to keep as relevance testing candidates (default: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--compression_threshold",
        type=float,
        default=0.5,
        help="EXIT confidence threshold for selection (default: 0.5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for EXIT processing"
    )
    parser.add_argument(
        "--num_hypothetical_documents",
        type=int,
        default=0,
        help="Number of hypothetical documents to generate for HyDE (default: 0 = disabled)"
    )
    parser.add_argument(
        "--hypothetical_document_model",
        type=str,
        default=None,
        help="Model for generating hypothetical documents (default: None = use base model)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/exit_semantic_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for models"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize tokenizer for token counting
    print("Loading tokenizer for token counting...")
    tokenizer = AutoTokenizer.from_pretrained(args.reader_model)
    
    # Initialize EXIT-Semantic compressor
    print("Initializing EXIT-Semantic compressor...")
    compressor = EXITSemanticCompressor(
        base_model=args.retriever_model,
        checkpoint=args.compression_model,
        embedding_model=args.embedding_model,
        device=device,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        threshold=args.compression_threshold,
        semantic_filter_ratio_context=args.semantic_filter_ratio_context,
        semantic_filter_ratio_relevance=args.semantic_filter_ratio_relevance,
        num_hypothetical_documents=args.num_hypothetical_documents,
        hypothetical_document_model=args.hypothetical_document_model
    )
    
    # Initialize reader model
    print("Loading reader model...")
    reader_model = AutoModelForCausalLM.from_pretrained(
        args.reader_model,
        device_map="auto",
        torch_dtype=torch.float16
    )
    reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model)
    
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
            compressor=compressor,
            reader_model=reader_model,
            reader_tokenizer=reader_tokenizer,
            dataset=dataset,
            max_samples=args.max_samples,
            tokenizer=tokenizer,
            top_k=args.top_k,
            device=device
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
        
        # Print timing breakdown if available
        if 'avg_timing' in task_results['metrics'] and task_results['metrics']['avg_timing']:
            print(f"  Timing Breakdown (avg per sample):")
            timing = task_results['metrics']['avg_timing']
            print(f"    Sentence Split: {timing.get('sentence_split', 0):.4f}s")
            print(f"    HyDE Generation: {timing.get('hyde_generation', 0):.4f}s")
            print(f"    Query Encoding: {timing.get('query_encoding', 0):.4f}s")
            print(f"    Doc Encoding: {timing.get('doc_encoding', 0):.4f}s")
            print(f"    Similarity Compute: {timing.get('similarity_compute', 0):.4f}s")
            print(f"    Filtering: {timing.get('filtering', 0):.4f}s")
            print(f"    EXIT Inference: {timing.get('exit_inference', 0):.4f}s")
            print(f"    Total Compress: {timing.get('total_compress', 0):.4f}s")
            print(f"    Generation Time: {timing.get('generation_time', 0):.4f}s")
        
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
