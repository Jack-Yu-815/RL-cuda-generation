#!/bin/bash
#SBATCH --job-name=kernelbench_dpo
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amittur@cs.cmu.edu
#SBATCH --time=2-00:00:00
#SBATCH --output=kernelbench_dpo_%j.out


eval "$(conda shell.bash hook)"
conda activate kernel-bench

wandb login <TOKEN>

srun python3 train.py
