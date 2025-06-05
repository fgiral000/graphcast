#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train-gencast
#SBATCH --mem-per-cpu=128G                  
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fa.giral@alumnos.upm.es
##------------------------ End job description ------------------------

module --force purge
module load apps/2021
module load Anaconda3
module load CUDA/12.1.1

source /media/beegfs/home/x249/x249087/.bashrc
conda activate graphcast_env

srun python -m training.train