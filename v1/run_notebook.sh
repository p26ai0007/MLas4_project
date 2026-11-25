#!/bin/bash
#SBATCH -J medmnist_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -p dgx
#SBATCH -t 20-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -o medmnist_%j.out
#SBATCH -e medmnist_%j.err

# 1. Change directory to where this specific batch job was submitted from
cd "$SLURM_SUBMIT_DIR"

# Clean up leftover slurm logs in current folder safely (suppress error if none exist)
rm -f slurm-*.out

# 2. Activate Python virtual environment from parent directory
source /scratch/data/p26ai0007/iitj/ml_project/.venv/bin/activate

# 3. Execute the Jupyter notebook
papermill medmnist-adversarial-training.ipynb executed_medmnist_output.ipynb -k python3