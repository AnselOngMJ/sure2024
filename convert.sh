#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --export=NONE
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=example@example.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source ~/miniconda3/bin/activate
conda activate fractal-clouds-env

python convert.py
