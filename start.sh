#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=7:59:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_output_attempt2.txt
#SBATCH --job-name=attempt2

module load TensorFlow
source myenv/bin/activate
python main.py
