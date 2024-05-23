#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=7:59:00
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --output=slurm_output_attempt1.txt
#SBATCH --job-name=attempt1

module load TensorFlow
source myenv/bin/activate
python main.py remote attempt1
