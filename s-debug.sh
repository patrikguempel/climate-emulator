#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:05:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_output.txt

git pull origin master
module load TensorFlow
source myenv/bin/activate
python main.py remote