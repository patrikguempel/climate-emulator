#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=./output/slurm_output_evaluate_mb_mlp3.txt
#SBATCH --job-name=eval

module load TensorFlow
source myenv/bin/activate
python scoring/evaluate.py
