#!/bin/bash
#SBATCH --job-name=semantic_sentence_cover_test
#SBATCH --output=/home/yichengtao/EXIT/outputs/logs/test_ssc_%j.out
#SBATCH --error=/home/yichengtao/EXIT/outputs/logs/test_ssc_%j.err
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
EMBEDDING_MODEL="Alibaba-NLP/gte-Qwen2-7B-instruct"
READER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS="HotpotQA 2wikimultihop musique NQ TQA"
DATA_ROOT="/mnt/data2/yichengtao/data/retrieval"
RETRIEVER="contriever-msmarco"

# Test Semantic Sentence Cover with default parameters (p=15%, query expansion 10%)
echo "=========================================="
echo "TEST: Semantic Sentence Cover (p=15%)"
echo "=========================================="
OUTPUT_DIR_1="/home/yichengtao/EXIT/outputs/semantic_sentence_cover_results/p15"

echo "Embedding Model: $EMBEDDING_MODEL"
echo "Reader Model: $READER_MODEL"
echo "Output Directory: $OUTPUT_DIR_1"
echo "Tasks: $TASKS"
echo "Document Sampling Ratio: 0.15 (15%)"
echo "Query Expansion Ratio: 0.1 (10%)"
echo ""

python test_semantic_sentence_cover.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --reader_model "$READER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASKS \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --query_expansion_min 10 \
    --query_expansion_ratio 0.1 \
    --document_sampling_ratio 0.15 \
    --output_dir "$OUTPUT_DIR_1"

if [ $? -eq 0 ]; then
    echo ""
    echo "TEST completed successfully at $(date)"
else
    echo ""
    echo "TEST failed at $(date)"
    exit 1
fi

# Test with different sampling ratio (p=10%)
echo ""
echo "=========================================="
echo "TEST: Semantic Sentence Cover (p=10%)"
echo "=========================================="
OUTPUT_DIR_2="/home/yichengtao/EXIT/outputs/semantic_sentence_cover_results/p10"

echo "Document Sampling Ratio: 0.10 (10%)"
echo ""

python test_semantic_sentence_cover.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --reader_model "$READER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASKS \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --query_expansion_min 10 \
    --query_expansion_ratio 0.1 \
    --document_sampling_ratio 0.10 \
    --output_dir "$OUTPUT_DIR_2"

if [ $? -eq 0 ]; then
    echo ""
    echo "TEST completed successfully at $(date)"
else
    echo ""
    echo "TEST failed at $(date)"
    exit 1
fi

# Test with different sampling ratio (p=20%)
echo ""
echo "=========================================="
echo "TEST: Semantic Sentence Cover (p=20%)"
echo "=========================================="
OUTPUT_DIR_3="/home/yichengtao/EXIT/outputs/semantic_sentence_cover_results/p20"

echo "Document Sampling Ratio: 0.20 (20%)"
echo ""

python test_semantic_sentence_cover.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --reader_model "$READER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASKS \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --query_expansion_min 10 \
    --query_expansion_ratio 0.1 \
    --document_sampling_ratio 0.20 \
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
