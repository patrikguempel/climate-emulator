#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=./output/slurm_output_gridsearch.txt
#SBATCH --job-name=grids

module load TensorFlow
source myenv/bin/activate
echo "ada-boost:"
python train/train_adaboost.py