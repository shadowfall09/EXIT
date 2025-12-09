#!/bin/bash
#SBATCH --job-name=exit_semantic_single
#SBATCH --output=/home/yichengtao/EXIT/outputs/logs/single_test_%j.out
#SBATCH --error=/home/yichengtao/EXIT/outputs/logs/single_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=aries
#SBATCH --time=12:00:00

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables
export HF_HOME=/mnt/data6/yichengtao/hf_cache
export HF_DATASETS_CACHE=/mnt/data6/yichengtao/hf_cache/datasets
export TRANSFORMERS_CACHE=/mnt/data6/yichengtao/hf_cache
export TOKENIZERS_PARALLELISM=false

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exit

# Change to project directory
cd /home/yichengtao/EXIT

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
HYPOTHETICAL_DOCUMENT_MODEL="google/gemma-2b-it"
READER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASK="HotpotQA"
DATA_ROOT="/mnt/data6/yichengtao/data/retrieval"
RETRIEVER="contriever-msmarco"
FILTER_RATIO=0.3
COMPRESSION_THRESHOLD=0.1
NUM_HYPOTHETICAL_DOCUMENTS=5
MAX_SAMPLES=50  # Test with 50 samples to see timing
OUTPUT_DIR="/home/yichengtao/EXIT/outputs/single_test_hyde_filter30"

echo "=========================================="
echo "Running EXIT-Semantic with HyDE Test"
echo "=========================================="
echo "Task: $TASK"
echo "Max Samples: $MAX_SAMPLES"
echo "Filter Ratio: $FILTER_RATIO (30%)"
echo "Compression Threshold: $COMPRESSION_THRESHOLD"
echo "HyDE Documents: $NUM_HYPOTHETICAL_DOCUMENTS"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run test
python test_exit_semantic.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --compression_model "$COMPRESSION_MODEL" \
    --reader_model "$READER_MODEL" \
    --retriever_model "$RETRIEVER_MODEL" \
    --hypothetical_document_model "$HYPOTHETICAL_DOCUMENT_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASK \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --semantic_filter_ratio_context "$FILTER_RATIO" \
    --semantic_filter_ratio_relevance "$FILTER_RATIO" \
    --compression_threshold "$COMPRESSION_THRESHOLD" \
    --num_hypothetical_documents "$NUM_HYPOTHETICAL_DOCUMENTS" \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Test completed successfully at $(date)"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "Test FAILED at $(date)"
    exit 1
fi
