#!/bin/bash
#SBATCH --export=ALL
#SBATCH -A gpu
#SBATCH --partition=scholar-j
#SBATCH --nodelist=scholar-j001       # pick one specific node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=laplace
#SBATCH --output=slurm_logs/%j

echo "Running on node:" $(hostname)
echo "CUDA device in use:"  
nvidia-smi --query-gpu=name,memory.total --format=csv

srun "$@"