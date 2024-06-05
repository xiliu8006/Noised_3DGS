import os
import shutil

# def copy_and_move_images(source_dir, destination_dir, video_dir):
#     # 复制文件和目录到目标文件夹
#     shutil.copytree(source_dir, destination_dir)

#     # 遍历目标文件夹下的子目录
#     for subdir in os.listdir(destination_dir):
#         # print(dirs)
#         # for subdir in dirs:
#             subdir_path = os.path.join(destination_dir, subdir)

#             # 如果是子目录
#             if os.path.isdir(subdir_path):
#                 images_dir = os.path.join(subdir_path, "images")
#                 # 创建名为 "images" 的目录
#                 os.makedirs(images_dir, exist_ok=True)
#                 # 移动子目录下的所有文件到 "images" 目录下
#                 for file in os.listdir(subdir_path):
#                     file_path = os.path.join(subdir_path, file)
#                     if os.path.isfile(file_path):
#                         shutil.move(file_path, images_dir)
                
#                 # 在子目录下创建名为 "sparse/0" 的目录
#                 sparse_dir = os.path.join(subdir_path, "sparse", "0_svd")
#                 os.makedirs(sparse_dir, exist_ok=True)

#                 video_cameras_dir = os.path.join(video_dir, subdir, "video", "cameras")
#                 for file in os.listdir(video_cameras_dir):
#                     file_path = os.path.join(video_cameras_dir, file)
#                     if os.path.isfile(file_path):
#                         if file == "uniform_images.txt":
#                             shutil.copy(file_path, os.path.join(sparse_dir, "images.txt"))
#                         else:
#                             shutil.copy(file_path, sparse_dir)
#                 file_path = os.path.join(video_dir, subdir, sparse/0, "points3D.txt")
#                 shutil.copy(file_path, sparse_dir)

def copy_and_move_images(source_dir, destination_dir, video_dir):
    # 复制文件和目录到目标文件夹
    shutil.copytree(source_dir, destination_dir)

    # 遍历目标文件夹下的子目录
    for subdir in os.listdir(destination_dir):
        # print(dirs)
        # for subdir in dirs:
            subdir_path = os.path.join(destination_dir, subdir)

            # 如果是子目录
            if os.path.isdir(subdir_path):
                images_dir = os.path.join(subdir_path, "images")
                # 创建名为 "images" 的目录
                os.makedirs(images_dir, exist_ok=True)
                # 移动子目录下的所有文件到 "images" 目录下
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        shutil.move(file_path, images_dir)
                
                # 在子目录下创建名为 "sparse/0" 的目录
                sparse_dir = os.path.join(subdir_path, "sparse", "0")
                os.makedirs(sparse_dir, exist_ok=True)

                video_cameras_dir = os.path.join(video_dir, subdir, "sparse", "0")
                for file in os.listdir(video_cameras_dir):
                    file_path = os.path.join(video_cameras_dir, file)
                    if os.path.isfile(file_path):
                        if file == "uniform_images.txt":
                            shutil.copy(file_path, os.path.join(sparse_dir, "images.txt"))
                        else:
                            shutil.copy(file_path, sparse_dir)
                # file_path = os.path.join(video_dir, subdir, "sparse/0", "points3D.txt")
                # shutil.copy(file_path, sparse_dir)

case_list = [3, 6, 9, 12]
for case in case_list:
    source_dir = f"/scratch/xi9/code/3DGS_SD_SR/outputs/Ref-{case}-reweight-all-consistency-recolor/samples"
    destination_dir = f"/scratch/xi9/DATASET/DL3DV-COLMAP-recolor/Ref-{case}-colmap-reweight"
    video_dir = f"/scratch/xi9/DATASET/DL3DV-COLMAP/Ref-{case}-colmap"
    copy_and_move_images(source_dir, destination_dir, video_dir)

print("复制和移动完成！")
