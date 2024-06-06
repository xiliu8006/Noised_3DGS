#!/bin/bash

# 指定要遍历的目录路径
list=("3" "6" "9")
l=8000
for element in "${list[@]}"; do
    directory="/scratch/xi9/DATASET/LLFF-COLMAP/Ref-$element-colmap"
    for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
        Basename=$(basename "$subdir")
        sbatch train.sh "$subdir" "/scratch/xi9/OUTPUTS/LLFF-ours/Ref-$element/$Basename" $l
        l=$((l + 1))
    done
done

# directory="/scratch/xi9/DATASET/DL3DV-COLMAP/Ref-6-colmap"
# for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
#     Basename=$(basename "$subdir")
#     sbatch train.sh "$subdir" "/scratch/xi9/OUTPUTS/real_image_norepeat/Ref-6/$Basename"
# done

# directory="/scratch/xi9/DATASET/DL3DV-COLMAP/Ref-9-colmap"
# for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
#     Basename=$(basename "$subdir")
#     sbatch train.sh "$subdir" "/scratch/xi9/OUTPUTS/real_image_norepeat/Ref-9/$Basename"
# done

# directory="/scratch/xi9/DATASET/DL3DV-COLMAP/Ref-12-colmap"
# for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
#     Basename=$(basename "$subdir")
#     sbatch train.sh "$subdir" "/scratch/xi9/OUTPUTS/real_image_norepeat/Ref-12/$Basename"
# done

