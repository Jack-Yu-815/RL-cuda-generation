#!/bin/bash
#SBATCH --job-name=kernelbench_dpo
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amittur@cs.cmu.edu
#SBATCH --time=2-00:00:00
#SBATCH --output=kernelbench_dpo_%j.out


eval "$(conda shell.bash hook)"
conda activate kernel-bench

wandb login 06c3c32dd2d81f1a7a6a6c184afa4927b4c3ad8c

srun python3 train.py
