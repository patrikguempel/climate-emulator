#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --output=slurm_output_attempt2.txt
#SBATCH --job-name=attempt2

module load TensorFlow
git pull origin master
source myenv/bin/activate
python main.py remote
