#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import numpy as np

import torch

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]

def create_covariance_matrix_torch(cov3D):
    """
    Convert flattened covariance components to a 3x3 covariance matrix.
    """
    Sigma = torch.zeros((cov3D.shape[0], 3, 3), device=cov3D.device, dtype=cov3D.dtype)
    Sigma[:, 0, 0] = cov3D[:, 0]
    Sigma[:, 0, 1] = cov3D[:, 1]
    Sigma[:, 0, 2] = cov3D[:, 2]
    Sigma[:, 1, 1] = cov3D[:, 3]
    Sigma[:, 1, 2] = cov3D[:, 4]
    Sigma[:, 2, 2] = cov3D[:, 5]
    Sigma[:, 1, 0] = Sigma[:, 0, 1]
    Sigma[:, 2, 0] = Sigma[:, 0, 2]
    Sigma[:, 2, 1] = Sigma[:, 1, 2]
    return Sigma


def project_3d_gaussian_to_2d(mean3d, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    cov3d = create_covariance_matrix_torch(cov3D)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    projected_means = t

    return projected_means, cov2d[:, :2, :2] + filter[None]


def project_3d_gaussian_to_2d_backup(means, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    """
    Vectorized version to project a batch of 3D Gaussian means to 2D.
    """
    device = means.device
    ones = torch.ones(means.shape[0], 1, device=device)
    means_hom = torch.cat((means, ones), dim=1)  # Nx4
    t = (viewmatrix @ means_hom.T)[:3, :] / viewmatrix[3, 3]  # Perspective division
    limx = 1.3 * tan_fovx * t[2, :]
    limy = 1.3 * tan_fovy * t[2, :]
    t[0, :] = torch.clamp(t[0, :], -limx, limx)
    t[1, :] = torch.clamp(t[1, :], -limy, limy)

    focal = torch.tensor([focal_x, focal_y], device=device).view(2, 1)
    J = torch.zeros(means.shape[0], 2, 3, device=device)
    J[:, 0, 0] = focal[0] / t[2, :]
    J[:, 1, 1] = focal[1] / t[2, :]
    J[:, 0, 2] = -focal[0] * t[0, :] / (t[2, :]**2)
    J[:, 1, 2] = -focal[1] * t[1, :] / (t[2, :]**2)

    Sigma_3D = create_covariance_matrix_torch(cov3D)
    if not torch.allclose(Sigma_3D, Sigma_3D.transpose(-2, -1)):
        raise ValueError("3D Covariance matrix is not symmetric.")
    W = viewmatrix[:3, :3].T
    # print("viewmatrix shape: ", viewmatrix.shape, J.shape)
    Sigma_3D = create_covariance_matrix_torch(cov3D)
    Sigma_2D = J @ W @ Sigma_3D @ W.T @ J.permute(0, 2, 1)
    Sigma_2D = (Sigma_2D + Sigma_2D.transpose(-2, -1)) / 2

    Sigma_2D += 0.3 * torch.eye(2, device=device)[None]
    projected_means = t[:2, :].T

    # 计算特征值确认所有特征值非负
    # eigenvalues, _ = torch.linalg.eigh(Sigma_2D)
    # negative_eigen_mask = eigenvalues < 0
    # indices_with_negatives = torch.any(negative_eigen_mask, dim=1)
    # if torch.any(indices_with_negatives):
    #     print("Detected negative eigenvalues in the following matrices:")
    #     for idx in torch.where(indices_with_negatives)[0]:
    #         print(f"Index: {idx}")
    #         print(f"Eigenvalues: {eigenvalues[idx]}")
    #         print(f"Covariance Matrix:\n{Sigma_2D[idx]}")
    #         print("------")  # 分隔符以提高可读性
    # else:
    #     print("All covariance matrices are positive semi-definite.")
    return projected_means, Sigma_2D

def calculate_ellipse_axes_lengths(sigma_2d):
    """
    Calculate the lengths of the major and minor axes of ellipses described by a batch
    of 2D Gaussian covariance matrices using PyTorch.

    Args:
    sigma_2d (torch.Tensor): An Nx2x2 batch of covariance matrix tensors.

    Returns:
    torch.Tensor: A tensor of shape Nx2 containing the lengths of the major and minor axes for each ellipse.
    """
    # Ensure input is on the correct device
    device = sigma_2d.device

    # Calculate eigenvalues of each covariance matrix
    eigenvalues, _ = torch.linalg.eigh(sigma_2d)
    axes_lengths = 2 * torch.sqrt(eigenvalues)
    axes_lengths, _ = torch.sort(axes_lengths, descending=True, dim=1)

    return axes_lengths

def calculate_ellipse_areas(sigma_2d):
    """
    Calculate the areas of ellipses described by a batch of 2D Gaussian covariance matrices using PyTorch.
    The input tensor should have shape (N, 2, 2), where each 2x2 matrix represents the covariance matrix
    of a 2D Gaussian distribution.

    Args:
    sigma_2d (torch.Tensor): An Nx2x2 batch of covariance matrix tensors.

    Returns:
    torch.Tensor: A tensor containing the areas of each ellipse.
    """
    axes_lengths = calculate_ellipse_axes_lengths(sigma_2d)
    areas = torch.pi * axes_lengths[:, 0] * axes_lengths[:, 1]

    return areas

def adjust_for_distance(Sigma_2D, z_values):
    """
    Adjust the 2D covariance matrices to remove the effect of distance from the camera.

    Args:
    Sigma_2D (torch.Tensor): A batch of 2x2 covariance matrices of shape (N, 2, 2).
    z_values (torch.Tensor): The z-coordinates (distances) of each matrix, shape (N).

    Returns:
    torch.Tensor: Adjusted 2x2 covariance matrices of shape (N, 2, 2).
    """
    # Compute eigenvalues and eigenvectors for each covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_2D)
    
    # Normalize eigenvalues to eliminate the effect of distance
    # We need to ensure z_values are reshaped to broadcast correctly
    adjusted_eigenvalues = eigenvalues / (z_values.unsqueeze(-1) ** 2)

    # Reconstruct the covariance matrices
    adjusted_Sigma_2D = eigenvectors @ torch.diag_embed(adjusted_eigenvalues) @ eigenvectors.transpose(-2, -1)

    return adjusted_Sigma_2D

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, render_mode = 'normal'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python:
        if render_mode == 'volume':
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = pc.get_scaling
        elif render_mode == 'area':
            focal_x = viewpoint_camera.image_width / (2.0 * tanfovx)
            focal_y = viewpoint_camera.image_height / (2.0 * tanfovy)
            viewmatrix = viewpoint_camera.world_view_transform
            scales = pc.get_scaling
            # print("scale shape: ", scales.max(), scales.min())
            cov3D = pc.get_covariance(scaling_modifier)
            mu_2d, cov2d = project_3d_gaussian_to_2d(pc.get_xyz, focal_x, focal_y, tanfovx, tanfovy, cov3D, viewmatrix)
            colors_precomp = pc.get_scaling.clone()
            colors_precomp = colors_precomp * 0
            colors_precomp[:, 0] = calculate_ellipse_areas(cov2d) * (abs(mu_2d[:, 2]))**2
        elif render_mode == 'fake':
            shs = pc.get_features
            shs = pc.get_fake_features
            shs = shs[:, 1:, :]
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
