#!/bin/bash
#SBATCH --job-name=exit_baseline_test
#SBATCH --output=/home/yichengtao/EXIT/outputs/logs/test_%j.out
#SBATCH --error=/home/yichengtao/EXIT/outputs/logs/test_%j.err
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
READER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
RETRIEVER_MODEL="google/gemma-2b-it"
TASKS="HotpotQA 2wikimultihop musique NQ TQA"
DATA_ROOT="/mnt/data2/yichengtao/data/retrieval"
RETRIEVER="contriever-msmarco"

# Test 1: Fine-tuned model
echo "=========================================="
echo "TEST 1: Fine-tuned Model"
echo "=========================================="
COMPRESSION_MODEL_1="/mnt/data2/yichengtao/EXIT/outputs/exit_model/final_model"
OUTPUT_DIR_1="/home/yichengtao/EXIT/outputs/baseline_results/reproduction"

echo "Compression Model: $COMPRESSION_MODEL_1"
echo "Output Directory: $OUTPUT_DIR_1"
echo "Tasks: $TASKS"
echo ""

python test_baseline.py \
    --compression_model "$COMPRESSION_MODEL_1" \
    --reader_model "$READER_MODEL" \
    --retriever_model "$RETRIEVER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASKS \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --compression_threshold 0.5 \
    --output_dir "$OUTPUT_DIR_1"

if [ $? -eq 0 ]; then
    echo ""
    echo "TEST 1 completed successfully at $(date)"
else
    echo ""
    echo "TEST 1 failed at $(date)"
    exit 1
fi

echo ""
echo "=========================================="
echo "TEST 2: Pretrained Model (HuggingFace)"
echo "=========================================="
COMPRESSION_MODEL_2="doubleyyh/exit-gemma-2b"
OUTPUT_DIR_2="/home/yichengtao/EXIT/outputs/baseline_results/official"

echo "Compression Model: $COMPRESSION_MODEL_2"
echo "Output Directory: $OUTPUT_DIR_2"
echo "Tasks: $TASKS"
echo ""

python test_baseline.py \
    --compression_model "$COMPRESSION_MODEL_2" \
    --reader_model "$READER_MODEL" \
    --retriever_model "$RETRIEVER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks $TASKS \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --compression_threshold 0.5 \
    --output_dir "$OUTPUT_DIR_2"

if [ $? -eq 0 ]; then
    echo ""
    echo "TEST 2 completed successfully at $(date)"
else
    echo ""
    echo "TEST 2 failed at $(date)"
    exit 1
fi

# Compare results
echo ""
echo "=========================================="
echo "COMPARISON SUMMARY"
echo "=========================================="
echo ""
echo "Fine-tuned Model Results:"
cat "$OUTPUT_DIR_1/summary.json" | python -m json.tool
echo ""
echo "Pretrained Model Results:"
cat "$OUTPUT_DIR_2/summary.json" | python -m json.tool
echo ""

echo "=========================================="
echo "All tests completed successfully!"
echo "End time: $(date)"
echo "=========================================="
