#!/bin/bash

# 指定要遍历的目录路径
directory=/home/liuxi/code/Noised_3DGS/output/only_generated_images/Ref-9

# 使用find命令获取目录列表，然后使用for循环逐个处理
for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
    Basename=$(basename "$subdir")
    python render.py -s "/data/chaoyi/dataset/DL3DV-10K/1K_pairset_9/$Basename/train/input" -m "$subdir" -i images_8 --eval --skip_train
    # python metrics.py -m "$subdir"
    # 在这里可以添加你想要执行的命令，比如列出该目录下的文件等
done

directory=/home/liuxi/code/Noised_3DGS/output/only_generated_images/Ref-3
# 使用find命令获取目录列表，然后使用for循环逐个处理
for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
    Basename=$(basename "$subdir")
    python render.py -s "/data/chaoyi/dataset/DL3DV-10K/1K_pairset_3/$Basename/train/input" -m "$subdir" -i images_8 --eval --skip_train
    # python metrics.py -m "$subdir"
    # 在这里可以添加你想要执行的命令，比如列出该目录下的文件等
done

# # 指定要遍历的目录路径
# directory=/home/liuxi/code/Noised_3DGS/output/only_generated_images/Ref-9

# # 使用find命令获取目录列表，然后使用for循环逐个处理
# for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
#     Basename=$(basename "$subdir")
#     python render.py -s "/home/liuxi/code/DATASET/DL3DV-evaluation/Ref-9-colmap/$Basename" -m "$subdir" --eval --skip_train
#     # 在这里可以添加你想要执行的命令，比如列出该目录下的文件等
# done

# directory=/home/liuxi/code/Noised_3DGS/output/only_generated_images/Ref-3
# # 使用find命令获取目录列表，然后使用for循环逐个处理
# for subdir in $(find "$directory" -mindepth 1 -maxdepth 1 -type d); do
#     Basename=$(basename "$subdir")
#     python render.py -s "/home/liuxi/code/DATASET/DL3DV-evaluation/Ref-3-colmap/$Basename" -m "$subdir" --eval --skip_train
#     # 在这里可以添加你想要执行的命令，比如列出该目录下的文件等
# done