#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=03:15:00
#SBATCH --job-name=setup
#SBATCH --output=outputs/setup.out
 
# Activate environment
uenv verbose cuda-11.4.4 cudnn-11.4-8.2.4
uenv miniconda3-py311
conda env create --file requirements.yaml

