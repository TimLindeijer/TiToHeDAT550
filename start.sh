#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=20:00:00
#SBATCH --job-name=btrb
#SBATCH --output=outputs/roberta_only_bootstrap.out
 
# Activate environment
uenv verbose cuda-11.4.4 cudnn-11.4-8.2.4
uenv miniconda3-py311
conda activate torch550
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Login wandb
# wandb login --relogin 16232e63f53b8b502555cea8afc019f0dfc5b5ee

# Run the Python script that uses the GPU
# python -u roberta_train.py
# python -u clip_train.py
# python -u clip_train_kfold.py
# python -u bootstrap_roberta.py
python -u bootstrap_roberta_llava.py
# python -u bootstrap.py
