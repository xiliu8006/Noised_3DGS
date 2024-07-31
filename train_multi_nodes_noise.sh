list=("3")
l=8000
task_count=0
max_tasks=3

for element in "${list[@]}"; do
    # directory="/scratch/xi9/DATASET/DL3DV-960P-Benchmark-Noised/Nonoised-45"
    directory="/scratch/xi9/DATASET/deblur_dataset/synthetic_camera_motion_blur"
    for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
        if [ $task_count -ge $max_tasks ]; then
            break 2
        fi
        Basename=$(basename "$subdir")
        sbatch train.sh "$subdir" "/scratch/xi9/OUTPUTS/DL3DV-Dual/Nonoised-gaussian-mode-finnal/$Basename" $l $Basename
        l=$((l + 1))
        task_count=$((task_count + 1))
    done
done
