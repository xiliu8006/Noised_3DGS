import cv2
import os
import math
import shutil
import glob
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
import random
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
from degradations import random_mixed_kernels
from colmap_utils import *
from torch.utils.data import Dataset

class DatasetProcessing(Dataset):
    def __init__(self, blur_mode='Mix'):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.target_folder = f'/scratch/xi9/DATASET/DL3DV-960P-Benchmark-Noised/{blur_mode}-45'
        self.ref_folder = '/scratch/xi9/Large-DATASET/DL3DV-10K/1K'
        self.scenes = set(os.listdir(self.ref_folder))
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_kernel_size = 21
        self.blur_sigma = [1, 1.5]
        self.blur_mode = blur_mode
        self.samples = {}
        for scene in self.scenes:
            scene_path = os.path.join(self.ref_folder, scene, 'images_4')
            frames = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path)])
            self.samples[scene] = frames

    def __len__(self):
        return len(self.samples)
    
    def copy_colmap_files(self):
        for scene in self.scenes:
            print("current dealing with scene: ", scene)
            images_path = os.path.join(self.ref_folder, scene, 'colmap/sparse/0/images.bin')
            camera_path = os.path.join(self.ref_folder, scene, 'colmap/sparse/0/cameras.bin')
            cameras, images = read_model(path=os.path.join(self.ref_folder, scene, "colmap/sparse/0"), ext=".bin")
            train_cam_infos = [c for idx, c in images.items()]
            train_cam_infos_sorted = sorted(train_cam_infos.copy(), key=lambda x: x.name)
            
            pc_path = os.path.join(self.ref_folder, scene, 'colmap/sparse/0/points3D.ply')
            colmap_dir = os.path.join(self.target_folder, scene, 'sparse/0')
            if not os.path.exists(colmap_dir):
                os.makedirs(colmap_dir)
            new_images_path = os.path.join(colmap_dir, 'images.bin')
            new_camera_path = os.path.join(colmap_dir, 'cameras.bin')
            new_pc_path = os.path.join(colmap_dir, 'points3D.ply')
            # shutil.copy2(images_path, new_images_path)
            # shutil.copy2(camera_path, new_camera_path)
            shutil.copy2(pc_path, new_pc_path)

            # # 定义图片目录路径
            images_dir = os.path.join(self.target_folder, scene, 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            new_images = []
            for i, img in enumerate(train_cam_infos_sorted):
                if i % 50 == 0:
                    base, ext = os.path.splitext(img.name)
                    img_name = f'{base}_ref{ext}'
                    new_image = BaseImage(
                        id=img.id,
                        qvec=img.qvec,
                        tvec=img.tvec,
                        camera_id=img.camera_id,
                        name=img_name,
                        xys=np.array([]),
                        point3D_ids=np.array([])
                    )
                    png_file = os.path.join(self.ref_folder, scene, 'images_4', f'{base}.png')
                    new_png_file = os.path.join(images_dir, f'{img_name}')
                    print("ref image: ", png_file, new_png_file)
                    shutil.copy2(png_file, new_png_file)
                else:
                    base, ext = os.path.splitext(img.name)
                    new_image = BaseImage(
                        id=img.id,
                        qvec=img.qvec,
                        tvec=img.tvec,
                        camera_id=img.camera_id,
                        name=img.name,
                        xys=np.array([]),
                        point3D_ids=np.array([])
                    )
                    png_file = os.path.join(self.ref_folder, scene, 'images_4', f'{base}.png')
                    basename = os.path.basename(png_file)
                    save_file = os.path.join(images_dir, basename)
                    img = cv2.imread(png_file)
                    if self.blur_mode == 'Mix':
                        kernel = random_mixed_kernels(
                            self.kernel_list,
                            self.kernel_prob,
                            self.blur_kernel_size,
                            self.blur_sigma,
                            self.blur_sigma, [-math.pi, math.pi],
                            noise_range=None)
                        img = cv2.filter2D(img, -1, kernel)

                    elif self.blur_mode == 'motion_blur':
                        kernel_size = self.blur_kernel_size
                        angle = 45
                        k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                        k[(kernel_size-1)//2, :] = np.ones(kernel_size, dtype=np.float32)
                        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((kernel_size/2-0.5, kernel_size/2-0.5), angle, 1.0), (kernel_size, kernel_size))
                        k = k * (1.0/np.sum(k))
                        img = cv2.filter2D(img, -1, k)
                    cv2.imwrite(img=img, filename=save_file)
                new_images.append(new_image)
            images_res = {image.id: image for image in new_images}
            write_model(cameras, images_res, colmap_dir, ext='.txt')
                

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        scene = self.scenes[idx]

        return {'image list': self.samples[scene]}
    
    
    def open_image(self, file_name):
        name, _ = os.path.splitext(file_name)
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 添加你认为可能的图片格式
        for ext in possible_extensions:
            full_path = name + ext
            full_path_upper = name + ext.upper()
            if os.path.isfile(full_path):
                return Image.open(full_path)
            elif os.path.isfile(full_path_upper):
                return Image.open(full_path_upper)
        raise FileNotFoundError("No image file found for {}".format(file_name))
    
if __name__ == '__main__':
    train_dataset = DatasetProcessing(blur_mode='motion_blur')
    train_dataset.copy_colmap_files()