#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --output=slurm_output.txt

module load TensorFlow
git pull origin master
source myenv/bin/activate
python main.py remote
