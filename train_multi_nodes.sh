#!/bin/bash

# 指定要遍历的目录路径
list=("3")
l=8000
for element in "${list[@]}"; do
    directory="/scratch/xi9/DATASET/DL3DV-960P-Benchmark-SVD/camP_h100x8_2K/samples"
    for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
        Basename=$(basename "$subdir")
        sbatch train.sh "$subdir/$element" "/scratch/xi9/OUTPUTS/DL3DV-Dual/camP_h100x8-$element/$Basename" $l $Basename
        l=$((l + 1))
    done
done



