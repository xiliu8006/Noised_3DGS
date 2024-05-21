#!/bin/bash

# 指定要遍历的目录路径
directory="/home/liuxi/code/DATASET/DL3DV-evaluation/Ref-9-colmap"

for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
    Basename=$(basename "$subdir")
    sbatch train.sh "$subdir" "/home/liuxi/code/Noised_3DGS/output/only_generated_images/Ref-9/$Basename"
done