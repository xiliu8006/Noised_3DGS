import copy
import os
import shutil

def merge_images_and_cameras(file1_path, cameras1_path, cameras2_path, output_images_path, output_cameras_path, pc_path):
    def read_camera_data(file_path, camera_id = 1):
        camera_info = None
        comments = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                comments.append(line)
            elif not camera_info and line.strip():
                parts = line.split()
                if camera_id == 2:
                    parts[2] = str(4946)
                parts[0] = str(camera_id)
                print(camera_info)
                new_line = ' '.join(parts) + '\n'
                camera_info = new_line

        return camera_info, comments

    def read_image_data(file_path, camera_id_offset=0, preserve_comments=False, repeat_num=0):
        images = []
        comments = []
        max_image_id = 0
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                if preserve_comments:
                    comments.append(line.strip())
            elif line.strip():
                parts = line.split()
                image_id = int(parts[0])
                # print(parts[9])
                if len(parts[9]) > 9:
                    print("Adjust camera ID to be either 1 or 2")
                    camera_id = 2  # Adjust camera ID to be either 1 or 2
                else:
                    camera_id = 1
                parts[-2] = str(camera_id)
                max_image_id = max(max_image_id, image_id)
                images.append(' '.join(parts))
        
        images_repeat = copy.deepcopy(images)
        for i in range(repeat_num):
            images.extend(images_repeat)
        
        image_index = 0  # Start indexing images from 0
        resort_images = []
        for line in images:
            if line.strip() and not line.startswith('#'):  # Process only non-empty, non-comment lines
                parts = line.split()
                parts[0] = str(image_index)  # Replace the first element (image ID) with the new index
                new_line = ' '.join(parts) + '\n'
                resort_images.append(new_line)
                image_index += 1
            else:
                resort_images.append(line)  # Preserve comments and empty lines as they are

        return resort_images, comments, max_image_id + (max_image_id * repeat_num)

    # Read camera data from both camera files
    camera1, comments1 = read_camera_data(cameras1_path)
    camera2, comments2 = read_camera_data(cameras1_path, 2)

    # Decide if a new camera ID is needed
    camera_id_offset = 0 if camera1 == camera2 else 1

    # Read and parse both images.txt files
    images1, image_comments1, max_id1 = read_image_data(file1_path, preserve_comments=True, camera_id_offset= 1, repeat_num=0)

    # Write the output images.txt
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_cameras_path, exist_ok=True)
    output_dir = output_images_path
    output_cameras_path = os.path.join(output_cameras_path, "cameras.txt")
    output_images_path = os.path.join(output_images_path, "images.txt")
    output_point_cloud_path = os.path.join(pc_path, "points3D.bin")
    with open(output_images_path, 'w') as file:
        for comment in image_comments1:
            file.write(comment + '\n')
        for image in images1:
            file.write(image + '\n')
            file.write('\n')  # Write an empty line after each image line

    # Write the output cameras.txt
    with open(output_cameras_path, 'w') as file:
        file.writelines([c + '\n' for c in comments1])
        file.write(camera1 + '\n')
        if camera_id_offset == 1:
            file.write(camera2 + '\n')
    
    shutil.copy(output_point_cloud_path, output_dir)

# Usage example
case_list = [3, 6, 9]
for case in case_list:
    for scene_dir in os.listdir(f"/scratch/xi9/DATASET/MipNeRF-COLMAP/Ref-{case}-colmap"):
        merge_images_and_cameras(
            f'/scratch/xi9/DATASET/MipNeRF-COLMAP/Ref-{case}-colmap/{scene_dir}/sparse/0_svd/images.txt',
            f'/scratch/xi9/DATASET/MipNeRF-COLMAP/Ref-{case}-colmap/{scene_dir}/sparse/0_svd/cameras.txt',
            f'/scratch/xi9/DATASET/MipNeRF-COLMAP/Ref-{case}-colmap/{scene_dir}/sparse/0_svd/cameras.txt',
            f'/scratch/xi9/DATASET/MipNeRF-COLMAP/Ref-{case}-colmap/{scene_dir}/sparse/0',
            f'/scratch/xi9/DATASET/MipNeRF-COLMAP/Ref-{case}-colmap/{scene_dir}/sparse/0',
            f'/scratch/xi9/DATASET/MipNeRF-COLMAP/Ref-{case}-colmap/{scene_dir}/sparse/0_svd'
        )


# Usage example
# file1_path = '/home/xi/code/DATASET/Mipnerf360_pairset/bicycle/partial_0.2/input/sparse/0/images.txt'  # Adjust this path
# file2_path = '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0_svd/images.txt'  # Adjust this path
# output_path = '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0/images.txt'  # Adjust this path
# merge_images_txt(file1_path, file2_path, output_path)


# # Usage example:
# input_file_path = '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0/images_none.txt'  # Adjust this path
# output_file_path = '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0/images.txt'  # Adjust this path
# modify_images_txt(input_file_path, output_file_path)