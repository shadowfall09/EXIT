#!/bin/bash
#SBATCH --job-name=exit_semantic_parallel
#SBATCH --output=/home/yichengtao/EXIT/outputs/logs/test_exit_semantic_%A_%a.out
#SBATCH --error=/home/yichengtao/EXIT/outputs/logs/test_exit_semantic_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=aries
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yt2@andrew.cmu.edu

# Test configuration:
# EXIT-Semantic with semantic_filter_ratio_context=0.7 semantic_filter_ratio_relevance=0.5
# With compression_threshold=0.1

# Print job information
echo "=========================================="
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Set HuggingFace cache BEFORE activating environment
export HF_HOME=/mnt/data6/yichengtao/hf_cache
export HF_DATASETS_CACHE=/mnt/data6/yichengtao/hf_cache/datasets

conda activate exit

# Set CUDA visible devices for this task (distribute across 4 GPUs)
export CUDA_VISIBLE_DEVICES=1

# Change to project directory
cd /home/yichengtao/EXIT

# Ensure cache directories exist
mkdir -p /mnt/data6/yichengtao/hf_cache/datasets

# Print environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
echo ""
nvidia-smi
echo ""

# Define models and tasks
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"
RETRIEVER_MODEL="google/gemma-2b-it"
COMPRESSION_MODEL="Yugong09/exit_reproduction"
READER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS="TQA"
DATA_ROOT="/mnt/data6/yichengtao/data/retrieval"
RETRIEVER="contriever-msmarco"

# Test configuration
FILTER_RATIO_CONTEXT=0.7
FILTER_RATIO_RELEVANCE=0.5
OUTPUT_DIR="/home/yichengtao/EXIT/outputs/exit_semantic_results/tiered_filter${FILTER_RATIO_CONTEXT}_${FILTER_RATIO_RELEVANCE}"
COMPRESSION_THRESHOLD=0.1

# Run test for this configuration
echo "=========================================="
echo "TEST: EXIT-Semantic (filter_context=${FILTER_RATIO_CONTEXT}, filter_relevance=${FILTER_RATIO_RELEVANCE}, compression_threshold=${COMPRESSION_THRESHOLD})"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Compression Model: $COMPRESSION_MODEL"
echo "Reader Model: $READER_MODEL"
echo "Retriever Model: $RETRIEVER_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Tasks: $TASKS"
echo "Semantic Filter Ratio Context: $FILTER_RATIO_CONTEXT"
echo "Semantic Filter Ratio Relevance: $FILTER_RATIO_RELEVANCE"
echo "Compression Threshold: $COMPRESSION_THRESHOLD"
echo ""
# Force HuggingFace cache location with all possible environment variables
export TRANSFORMERS_CACHE=/mnt/data6/yichengtao/hf_cache
export HF_DATASETS_CACHE=/mnt/data6/yichengtao/hf_cache/datasets
export HF_HOME=/mnt/data6/yichengtao/hf_cache
export TOKENIZERS_PARALLELISM=false

    # --max_samples 10 \

python test_exit_semantic.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --compression_model "$COMPRESSION_MODEL" \
    --reader_model "$READER_MODEL" \
    --retriever_model "$RETRIEVER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASKS \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --semantic_filter_ratio_context "$FILTER_RATIO_CONTEXT" \
    --semantic_filter_ratio_relevance "$FILTER_RATIO_RELEVANCE" \
    --compression_threshold "$COMPRESSION_THRESHOLD" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Task (filter=${FILTER_PERCENT}%) completed successfully at $(date)"
    echo "=========================================="
else
    echo ""
    echo "Task (filter=${FILTER_PERCENT}%) FAILED at $(date)"
    exit 1
fi
