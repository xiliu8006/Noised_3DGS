#!/bin/bash
#SBATCH --job-name 3DGS
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 64gb
#SBATCH --time 72:00:00
#SBATCH --gpus a100:1

source /etc/profile.d/modules.sh
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate CFW

cd /scratch/xi9/code/Noised_3DGS 

python train.py -s $1 -m $2 --port $3 -r 8

Basename=$(basename "$1")
python render.py -s "/scratch/xi9/DATASET/DL3DV/$Basename/" -m $2 -i images_8 -r 1 --eval --skip_train
python metrics.py -m $2

# python train.py -s /scratch/xi9/DATASET/DL3DV-COLMAP-recolor/Ref-12-colmap/bd47fd2bd339b8b286470aa40673d829ab646fb92dfc6172e70a9ee966904135 -m /scratch/xi9/OUTPUTS/max_mean_conf_and_repeat_5-recolor/Ref-12/bd47fd2bd339b8b286470aa40673d829ab646fb92dfc6172e70a9ee966904135 --port 6001