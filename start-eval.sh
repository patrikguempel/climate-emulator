#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=./output/slurm_output_eval_xgb_lr.txt
#SBATCH --job-name=evalxgb

module load TensorFlow
source myenv/bin/activate
python scoring/evaluate.py