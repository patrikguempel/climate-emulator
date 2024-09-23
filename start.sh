#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=./output/slurm_output_train_mlp_ca8-3.txt
#SBATCH --job-name=mlpca8+

module load TensorFlow
source myenv/bin/activate
python train/train_mlp_ca8-3.py