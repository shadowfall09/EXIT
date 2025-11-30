#!/bin/bash
#SBATCH --job-name=exit_semantic_test
#SBATCH --output=/home/yichengtao/EXIT/outputs/logs/test_exit_semantic_%j.out
#SBATCH --error=/home/yichengtao/EXIT/outputs/logs/test_exit_semantic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=taurus
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yt2@andrew.cmu.edu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exit

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd /home/yichengtao/EXIT

# Set HuggingFace cache
export HF_HOME=/mnt/data2/yichengtao/huggingface_cache

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
COMPRESSION_MODEL="/home/yichengtao/EXIT/outputs/checkpoints/exit_lora"  # Update with your checkpoint path
READER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS="HotpotQA 2wikimultihop musique NQ TQA"
DATA_ROOT="/mnt/data2/yichengtao/data/retrieval"
RETRIEVER="contriever-msmarco"

# Test EXIT-Semantic with default parameters (semantic_filter_ratio=0.5, compression_threshold=0.1)
echo "=========================================="
echo "TEST: EXIT-Semantic (filter=50%, compression_threshold=0.1)"
echo "=========================================="
OUTPUT_DIR_1="/home/yichengtao/EXIT/outputs/exit_semantic_results/filter50_th0.1"

echo "Embedding Model: $EMBEDDING_MODEL"
echo "Compression Model: $COMPRESSION_MODEL"
echo "Reader Model: $READER_MODEL"
echo "Retriever Model: $RETRIEVER_MODEL"
echo "Output Directory: $OUTPUT_DIR_1"
echo "Tasks: $TASKS"
echo "Semantic Filter Ratio: 0.5 (50%)"
echo "Compression Threshold: 0.1"
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
    --semantic_filter_ratio 0.5 \
    --compression_threshold 0.1 \
    --output_dir "$OUTPUT_DIR_1"

if [ $? -eq 0 ]; then
    echo ""
    echo "TEST completed successfully at $(date)"
else
    echo ""
    echo "TEST failed at $(date)"
    exit 1
fi

# Test with different filter ratio (filter=30%, compression_threshold=0.1)
echo ""
echo "=========================================="
echo "TEST: EXIT-Semantic (filter=30%, compression_threshold=0.1)"
echo "=========================================="
OUTPUT_DIR_2="/home/yichengtao/EXIT/outputs/exit_semantic_results/filter30_th0.1"

echo "Semantic Filter Ratio: 0.3 (30%)"
echo "Compression Threshold: 0.1"
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
    --semantic_filter_ratio 0.3 \
    --compression_threshold 0.1 \
    --output_dir "$OUTPUT_DIR_2"

if [ $? -eq 0 ]; then
    echo ""
    echo "TEST completed successfully at $(date)"
else
    echo ""
    echo "TEST failed at $(date)"
    exit 1
fi

# Test with different filter ratio (filter=70%, compression_threshold=0.1)
echo ""
echo "=========================================="
echo "TEST: EXIT-Semantic (filter=70%, compression_threshold=0.1)"
echo "=========================================="
OUTPUT_DIR_3="/home/yichengtao/EXIT/outputs/exit_semantic_results/filter70_th0.1"

echo "Semantic Filter Ratio: 0.7 (70%)"
echo "Compression Threshold: 0.1"
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
    --semantic_filter_ratio 0.7 \
    --compression_threshold 0.1 \
    --output_dir "$OUTPUT_DIR_3"

if [ $? -eq 0 ]; then
    echo ""
    echo "All tests completed successfully at $(date)"
else
    echo ""
    echo "TEST failed at $(date)"
    exit 1
fi

echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "End time: $(date)"
echo "=========================================="
