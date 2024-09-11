#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=./output/slurm_output_train_mlp_ca.txt
#SBATCH --job-name=mlp_ca

module load TensorFlow
source myenv/bin/activate
python train/train_mlp_ca.py