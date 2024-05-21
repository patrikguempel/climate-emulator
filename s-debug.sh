#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:05:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

module load Python
git pull origin master
python main.py remote