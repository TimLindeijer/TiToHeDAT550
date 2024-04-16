#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=03:15:00
#SBATCH --job-name=pip
#SBATCH --output=pip.out
 
# Activate environment
uenv verbose cuda-11.4.4 cudnn-11.4-8.2.4
uenv miniconda3-py311
# conda env remove --name torch550
# conda env create --file requirements.yaml
conda activate torch550
pip install wandb