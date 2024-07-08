# python render.py -m /home/xi/code/gaussian-splatting/output/base
# python render.py -m /home/xi/code/gaussian-splatting/output/base_no_reweight
# python render.py -m /home/xi/code/gaussian-splatting/output/on_weight_l1
# python render.py -m /home/xi/code/gaussian-splatting/output/on_weight_ssim_l1


#!/bin/bash

# 指定要遍历的目录路径
directory="/home/liuxi/code/DATASET/DL3DV-evaluation/Ref-3-colmap"

# 使用find命令获取目录列表，然后使用for循环逐个处理
for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
    Basename=$(basename "$subdir")
    CUDA_VISIBLE_DEVICES=1, python train.py -s "$subdir" -m "/home/liuxi/code/Noised_3DGS/output/Realimage_with_generated_images/Ref-3/$Basename" --port 6019 -r 8
    # 在这里可以添加你想要执行的命令，比如列出该目录下的文件等
done

# directory="/home/liuxi/code/DATASET/DL3DV-evaluation/Ref-3-colmap"
# # 使用find命令获取目录列表，然后使用for循环逐个处理
# for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
#     Basename=$(basename "$subdir")
#     python train.py -s "$subdir" -m "/home/liuxi/code/Noised_3DGS/output/only_generated_images/Ref-3/$Basename"
#     # 在这里可以添加你想要执行的命令，比如列出该目录下的文件等
# done