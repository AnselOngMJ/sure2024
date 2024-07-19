#!/bin/bash 
#SBATCH --job-name=cloud-data
#SBATCH --partition=short-serial
#SBATCH --output=./%A/%A_%a.out
#SBATCH --error=./%A/%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --mem=2G
#SBATCH --array=0-1406

source /home/users/${USER}/miniconda3/bin/activate
conda activate fractal-clouds-env

python get_cloud_data.py -c ${SLURM_ARRAY_TASK_ID} -d ${SLURM_ARRAY_JOB_ID}
