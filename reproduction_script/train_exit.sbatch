#!/bin/bash
#SBATCH --job-name=exit_train
#SBATCH --output=/home/yichengtao/EXIT/outputs/logs/train_%j.out
#SBATCH --error=/home/yichengtao/EXIT/outputs/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:2
#SBATCH --partition=taurus
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yt2@andrew.cmu.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exit

# Set CUDA visible devices to 6 and 7
export CUDA_VISIBLE_DEVICES=6,7

# Change to project directory
cd /home/yichengtao/EXIT

# Set HuggingFace cache
export HF_HOME=/mnt/data2/yichengtao/huggingface_cache

# Print environment info
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
nvidia-smi

# Run training
python train/train.py \
    --model_id "google/gemma-2b-it" \
    --train_dataset "data/processed/train_dataset" \
    --test_dataset "data/processed/test_dataset" \
    --output_dir "/mnt/data2/yichengtao/EXIT/outputs/exit_model" 
    
echo "End time: $(date)"