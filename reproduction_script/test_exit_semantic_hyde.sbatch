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
#SBATCH --array=0-2%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yt2@andrew.cmu.edu

# Job array indices:
# 0: EXIT-Semantic with semantic_filter_ratio=0.5 (50%)
# 1: EXIT-Semantic with semantic_filter_ratio=0.3 (30%)
# 2: EXIT-Semantic with semantic_filter_ratio=0.7 (70%)
# All with compression_threshold=0.1
# Will run on 4 GPUs with SLURM scheduling (max 4 parallel tasks)

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
conda activate exit

# Set CUDA visible devices for this task (distribute across 4 GPUs)
# export CUDA_VISIBLE_DEVICES=$((SLURM_ARRAY_TASK_ID % 4))

# Change to project directory
cd /home/yichengtao/EXIT

# Set HuggingFace cache
export HF_HOME=/mnt/data6/yichengtao/hf_cache

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
HYPOTHETICAL_DOCUMENT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
READER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS="HotpotQA TQA"
DATA_ROOT="/mnt/data6/yichengtao/data/retrieval"
RETRIEVER="contriever-msmarco"

# Define test configurations array
# Array index: 0 = filter50_th0.1, 1 = filter30_th0.1, 2 = filter70_th0.1
FILTER_RATIOS=(0.5 0.3 0.7)
OUTPUT_DIRS=(
    "/home/yichengtao/EXIT/outputs/exit_semantic_hyde_results/filter50_th0.1"
    "/home/yichengtao/EXIT/outputs/exit_semantic_hyde_results/filter30_th0.1"
    "/home/yichengtao/EXIT/outputs/exit_semantic_hyde_results/filter70_th0.1"
)

# Get configuration for this task
FILTER_RATIO=${FILTER_RATIOS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR=${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}
NUM_HYPOTHETICAL_DOCUMENTS=5
COMPRESSION_THRESHOLD=0.1

# Determine filter percentage for display
if [ "$FILTER_RATIO" = "0.5" ]; then
    FILTER_PERCENT=50
elif [ "$FILTER_RATIO" = "0.3" ]; then
    FILTER_PERCENT=30
elif [ "$FILTER_RATIO" = "0.7" ]; then
    FILTER_PERCENT=70
fi

# Run test for this configuration
echo "=========================================="
echo "TEST: EXIT-Semantic (filter=${FILTER_PERCENT}%, compression_threshold=${COMPRESSION_THRESHOLD})"
echo "Task ID: $SLURM_ARRAY_TASK_ID / GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Compression Model: $COMPRESSION_MODEL"
echo "Reader Model: $READER_MODEL"
echo "Retriever Model: $RETRIEVER_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo "Tasks: $TASKS"
echo "Semantic Filter Ratio: $FILTER_RATIO (${FILTER_PERCENT}%)"
echo "Compression Threshold: $COMPRESSION_THRESHOLD"
echo ""

python test_exit_semantic.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --compression_model "$COMPRESSION_MODEL" \
    --reader_model "$READER_MODEL" \
    --retriever_model "$RETRIEVER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASKS \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --semantic_filter_ratio "$FILTER_RATIO" \
    --compression_threshold "$COMPRESSION_THRESHOLD" \
    --num_hypothetical_documents "$NUM_HYPOTHETICAL_DOCUMENTS" \
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
