#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --output=./output/slurm_output_xgb1.txt
#SBATCH --job-name=xgb1

module load TensorFlow
source myenv/bin/activate
python train/train_xgb.py
