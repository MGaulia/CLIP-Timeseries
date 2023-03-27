#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu
#SBATCH -t 300
. gpu_env/bin/activate
python3 train.py $1 $2