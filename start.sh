#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --nodelist=gorina8 
#SBATCH --time=30:00:00
#SBATCH --job-name=g7Dat550
#SBATCH --output=main.out
 
# Activate environment
uenv verbose cuda-11.4.4 cudnn-11.4-8.2.4
uenv miniconda3-py311
conda activate torch550
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run the Python script that uses the GPU
python -u main.py