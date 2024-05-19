from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from math import exp
import torch
import os
import lpips
import torch.nn.functional as F
from dreamsim import dreamsim

def load_image_as_tensor(image_path):
    """Load an image and convert it to a PyTorch tensor with values in [0, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    return transform(image)

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_psnr_for_common_images(dir1, dir2, lpips_model, dreamsim_model, preprocess):
    """Calculate the PSNR for images with the same name in two directories."""
    # Get the set of common filenames in both directories
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))
    common_filenames = files_dir1.intersection(files_dir2)
    # lpips_model = lpips.LPIPS(net='alex')

    if not common_filenames:
        print("No common image filenames found between the two directories.")
        return

    psnrs = []
    lpips_ = []
    ssims = []
    dreamsims = []
    for filename in common_filenames:
        if '_DSC' in filename:
            continue
        img_path1 = os.path.join(dir1, filename)
        img_path2 = os.path.join(dir2, filename)

        img1 = preprocess(Image.open(img_path1)).to("cuda")
        img2 = preprocess(Image.open(img_path2)).to("cuda")
        distance = dreamsim_model(img1, img2)
        dreamsims.append(distance)

        image_tensor1 = load_image_as_tensor(img_path1)
        image_tensor2 = load_image_as_tensor(img_path2)

        current_psnr = psnr(image_tensor1, image_tensor2)
        current_ssim = ssim(image_tensor1, image_tensor2)
        current_lpips = lpips_model(image_tensor1.unsqueeze(0), image_tensor2.unsqueeze(0))
        psnrs.append(current_psnr.mean())
        lpips_.append(current_lpips)
        ssims.append(current_ssim)
        # print(f"PSNR for {filename}: {current_psnr.mean()}")

        # total_psnr += current_psnr
    # c = torch.tensor(psnrs)
    # print(len(ssims))
    print(f"Average PSNR for common images: {torch.tensor(psnrs).mean().item()} {torch.tensor(lpips_).mean().item()} {torch.tensor(ssims).mean().item()} {torch.tensor(dreamsims).mean().item()}")


dir1 = '/home/xi/code/DATASET/SVD_inferences/bicycle/HR_images'
dir2 = '/home/xi/code/DATASET/SVD_inferences/bicycle/lr_images'
dir3 = '/home/xi/code/DATASET/SVD_inferences/bicycle/images'
dir4 = '/home/xi/code/gaussian-splatting/output/on_weight_ssim_l1/train/ours_30000/renders'
# dir_list = [dir1, dir2, dir3, dir4]
dir_list = [dir1, dir4]
# output_list = ["render", "samples", "diffusion", "ours"]
output_list = ["ours"]
lpips_model = lpips.LPIPS(net='alex')
dreamsim_model, preprocess = dreamsim(pretrained=True)

# calculate_psnr_for_common_images(dir2, dir4, lpips_model,dreamsim_model, preprocess)

for dir, name in zip(dir_list[1:], output_list):
    print(">>>>>>>>>>>>>>>>>>>current print: ", name)
    calculate_psnr_for_common_images(dir1, dir, lpips_model,dreamsim_model, preprocess)

# base psnr 20.67
# base_no_weight psnr 20.63
# base on weight l1 20.33
# base on weight ssim 20.433