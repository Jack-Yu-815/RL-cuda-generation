#!/bin/bash

#SBATCH --job-name=llama_31_8b_dapt_%j # Job name
#SBATCH --output=run_dapt_%j.out   # Standard output log (%j expands to job ID)
##SBATCH --error=run_dapt_%j.err    # Standard error log (%j expands to job ID)
#SBATCH --gres=gpu:A100_40GB:1      # Number of GPUs requested per node
##SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=64G                   # Memory per node
#SBATCH --time=2-00:00:00           # Wall time limit (hh:mm:ss)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amittur@cs.cmu.edu


echo "--------------------------------------------------"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "Starting Time: $(date)"
echo "--------------------------------------------------"


eval "$(conda shell.bash hook)"
conda activate research


# --- Define your torchtune command ---
# Replace 'your_recipe.py' and 'your_config.yaml' with your actual files
# Adjust '--nproc_per_node' based on the number of GPUs you want to use per node
# Add any other arguments required by your recipe/config
TUNE_CMD="tune run lora_finetune_single_device --config config.yaml"


echo "Running command:"
echo "$TUNE_CMD"
echo "--------------------------------------------------"

# --- Execute the command ---
eval $TUNE_CMD

# --- Job completion ---
echo "--------------------------------------------------"
echo "Ending Time: $(date)"
echo "Job finished with exit code $?"
echo "--------------------------------------------------"
