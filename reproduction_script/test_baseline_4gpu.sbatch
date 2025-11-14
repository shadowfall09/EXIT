#!/bin/bash
#SBATCH --job-name=exit_4gpu_test
#SBATCH --output=/home/yichengtao/EXIT/outputs/logs/test_4gpu_%A_%a.out
#SBATCH --error=/home/yichengtao/EXIT/outputs/logs/test_4gpu_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=taurus
#SBATCH --time=24:00:00
#SBATCH --array=0-9%5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yt2@andrew.cmu.edu

# Job array indices:
# 0-4: Fine-tuned model on 4 tasks (HotpotQA, 2wikimultihop, NQ, TQA)
# 5-9: Pretrained model on 4 tasks (HotpotQA, 2wikimultihop, NQ, TQA)
# Total: 10 tasks, will run on 4 GPUs with SLURM scheduling

# Print job information
echo "=========================================="
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exit

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

# Define models and common settings
READER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
RETRIEVER_MODEL="google/gemma-2b-it"
DATA_ROOT="/mnt/data2/yichengtao/data/retrieval"
RETRIEVER="contriever-msmarco"

# Define tasks array
TASKS_ARRAY=("HotpotQA" "2wikimultihop" "NQ" "TQA")

# Determine which model and task to run based on array task ID
if [ $SLURM_ARRAY_TASK_ID -lt 5 ]; then
    # Fine-tuned model (tasks 0-4)
    MODEL_TYPE="finetuned"
    COMPRESSION_MODEL="/mnt/data2/yichengtao/EXIT/outputs/exit_model/final_model"
    OUTPUT_BASE="/home/yichengtao/EXIT/outputs/baseline_results/reproduction_0.1"
    TASK_INDEX=$SLURM_ARRAY_TASK_ID
else
    # Pretrained model (tasks 5-9)
    MODEL_TYPE="pretrained"
    COMPRESSION_MODEL="doubleyyh/exit-gemma-2b"
    OUTPUT_BASE="/home/yichengtao/EXIT/outputs/baseline_results/official_0.1"
    TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 5))
fi

TASK=${TASKS_ARRAY[$TASK_INDEX]}
OUTPUT_DIR="$OUTPUT_BASE"

echo "=========================================="
echo "Configuration:"
echo "  Model Type: $MODEL_TYPE"
echo "  Compression Model: $COMPRESSION_MODEL"
echo "  Task: $TASK"
echo "  Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run evaluation for this specific task
python test_baseline.py \
    --compression_model "$COMPRESSION_MODEL" \
    --reader_model "$READER_MODEL" \
    --retriever_model "$RETRIEVER_MODEL" \
    --data_root "$DATA_ROOT" \
    --tasks "$TASK" \
    --top_k 5 \
    --retriever "$RETRIEVER" \
    --compression_threshold 0.1 \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Task $TASK ($MODEL_TYPE) completed successfully at $(date)"
    echo "=========================================="
else
    echo ""
    echo "Task $TASK ($MODEL_TYPE) FAILED at $(date)"
    exit 1
fi
