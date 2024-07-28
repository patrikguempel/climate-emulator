#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=./output/slurm_output_cnn1.txt
#SBATCH --job-name=cnn1

module load TensorFlow
source myenv/bin/activate
python ./train/train_cnn.py
