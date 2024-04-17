#!/bin/bash
#SBATCH --time=4-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-g4

# Your command to run the Python script
#pip install torch torchvision compressai
#source activate compressails
python3 testing_pix2pix_video.py
