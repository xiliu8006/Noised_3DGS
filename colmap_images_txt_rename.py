import copy
def modify_images_txt(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # Initialize a counter for the filename
    image_counter = 0

    # Prepare to collect modified lines
    modified_lines = []

    for i in range(len(lines)):
        if i % 2 == 0:
            # Even lines contain the image metadata
            parts = lines[i].split()
            print(parts)
            if parts[-1] == "None":
                # Replace 'none' with a formatted number
                parts[-1] = f"{image_counter:05d}.png"
                image_counter += 1
            modified_lines.append(" ".join(parts) + "\n")
        else:
            # Odd lines contain just feature information, copy as is
            modified_lines.append(lines[i])

    # Write the modified lines to the output file
    with open(output_path, 'w') as file:
        file.writelines(modified_lines)

def merge_images_txt(file1_path, file2_path, output_path, repeat_num=1):
    def read_image_data(file_path, preserve_comments=False):
        images = []
        comments = []
        max_image_id = 0
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith('#'):
                if preserve_comments:
                    comments.append(line.strip())  # Store comment lines if preserving
            else:
                # Parse image information
                parts = line.strip().split()
                if parts:
                    image_id = int(parts[0])
                    max_image_id = max(max_image_id, image_id)
                    images.append(line.strip())  # Store the full line
        
        images_repeat = images.deepcopy()
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


        return resort_images, comments, max_image_id + max_image_id * (repeat_num - 1)

    # Read and parse both images.txt files
    # Preserve comments only from the first file
    images1, comments1, max_id1 = read_image_data(file1_path, preserve_comments=True, repeat_num=30)
    images2, _, _ = read_image_data(file2_path, preserve_comments=False)

    # Offset for the second file's image IDs
    image_id_offset = max_id1 + 1

    # Adjust IDs in the second set of images
    adjusted_images2 = []
    for image_info in images2:
        parts = image_info.split()
        new_image_id = int(parts[0]) + image_id_offset
        parts[0] = str(new_image_id)
        adjusted_images2.append(' '.join(parts))

    # Write the output file by combining the original and adjusted second file
    with open(output_path, 'w') as file:
        # Write comments from the first file
        for comment in comments1:
            file.write(comment + '\n')
        # Write adjusted images
        for image_info in images1 + adjusted_images2:
            file.write(image_info + '\n')
            file.write('\n')  # Write an empty line after each image info line

def merge_images_and_cameras(file1_path, file2_path, cameras1_path, cameras2_path, output_images_path, output_cameras_path):
    def read_camera_data(file_path):
        camera_info = None
        comments = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                comments.append(line.strip())
            elif not camera_info and line.strip():
                camera_info = line.strip()

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
                camera_id = 1 + camera_id_offset  # Adjust camera ID to be either 1 or 2
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
    camera2, comments2 = read_camera_data(cameras2_path)

    # Decide if a new camera ID is needed
    camera_id_offset = 0 if camera1 == camera2 else 1

    # Read and parse both images.txt files
    images1, image_comments1, max_id1 = read_image_data(file1_path, preserve_comments=True, repeat_num=30)
    images2, _, _ = read_image_data(file2_path, camera_id_offset)

    # Offset for the second file's image IDs
    image_id_offset = max_id1 + 1

    # Adjust IDs in the second set of images
    adjusted_images2 = []
    for image_info in images2:
        parts = image_info.split()
        new_image_id = int(parts[0]) + image_id_offset
        parts[0] = str(new_image_id)
        adjusted_images2.append(' '.join(parts))

    # Write the output images.txt
    with open(output_images_path, 'w') as file:
        for comment in image_comments1:
            file.write(comment + '\n')
        for image in images1 + adjusted_images2:
            file.write(image + '\n')
            file.write('\n')  # Write an empty line after each image line

    # Write the output cameras.txt
    with open(output_cameras_path, 'w') as file:
        file.writelines([c + '\n' for c in comments1])
        file.write(camera1 + '\n')
        if camera_id_offset == 1:
            file.write(camera2 + '\n')

# Usage example
merge_images_and_cameras(
    '/home/xi/code/DATASET/Mipnerf360_pairset/bicycle/partial_0.2/input/sparse/0/images.txt',
    '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0_svd/images.txt',
    '/home/xi/code/DATASET/Mipnerf360_pairset/bicycle/partial_0.2/input/sparse/0/cameras.txt',
    '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0_svd/cameras.txt',
    '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0/images.txt',
    '/home/xi/code/DATASET/SVD_inferences/bicycle/sparse/0/cameras.txt'
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