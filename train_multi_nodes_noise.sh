list=("3")
l=8000
for element in "${list[@]}"; do
    directory="/scratch/xi9/DATASET/DL3DV-960P-Benchmark-Noised/motion_blur"
    for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
        Basename=$(basename "$subdir")
        sbatch train.sh "$subdir" "/scratch/xi9/OUTPUTS/DL3DV-Dual/motion-Noised-usedual/$Basename" $l $Basename
        l=$((l + 1))
    done
done
