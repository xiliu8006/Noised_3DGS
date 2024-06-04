#!/bin/bash

# 指定要遍历的目录路径
list=("3" "6" "9" "12")
l=8000
# list=("3")
for element in "${list[@]}"; do
    directory="/scratch/xi9/DATASET/DL3DV-COLMAP-recolor/Ref-$element-colmap"
    for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
        Basename=$(basename "$subdir")
        sbatch render.sh "$subdir" "/scratch/xi9/OUTPUTS/for_rendering/Ref-$element/$Basename" $l
        l=$((l + 1))
    done
done