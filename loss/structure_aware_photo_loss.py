import torch
import torch.nn.functional as F
import torch.nn as nn

pixel_coords = None


def build_pixel_coordinate_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h, device=depth.device, dtype=depth.dtype).view(1, h, 1).expand(
        1, h, w)
    j_range = torch.arange(0, w, device=depth.device, dtype=depth.dtype).view(1, 1, w).expand(
        1, h, w)
    ones = torch.ones(1, h, w, device=depth.device, dtype=depth.dtype)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)

def backproject_depth_to_camera(depth, intrinsics_inv):
    global pixel_coords
    b, h, w = depth.size()
    if (pixel_coords is None or pixel_coords.size(2) < h or pixel_coords.size(3) < w
            or pixel_coords.device != depth.device or pixel_coords.dtype != depth.dtype):
        build_pixel_coordinate_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def validate_tensor_shape(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def project_camera_to_pixel_grid(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, boundary):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_temp = (X / Z).long()
    Y_temp = (Y / Z).long()
    X_mask1 = ((X_temp >= w) + (X_temp < 0)).detach()
    X_temp[X_mask1] = 0
    Y_mask1 = ((Y_temp >= h) + (Y_temp < 0)).detach()
    Y_temp[Y_mask1] = 0

    boundary_temp = boundary[0][0]
    mask_1 = boundary_temp[Y_temp, X_temp]
    X_norm = 2 * (X / Z) / (w - 1) - 1
    Y_norm = 2 * (Y / Z) / (h - 1) - 1
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w), mask_1.reshape(b, h, w)

def warp_source_image_to_target(depth, img, pose, intrinsics, boundary, padding_mode):
    validate_tensor_shape(img, 'img', 'B3HW')
    validate_tensor_shape(depth, 'depth', 'B1HW')
    validate_tensor_shape(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = backproject_depth_to_camera(depth.squeeze(1), intrinsics.inverse())

    pose_mat = pose

    proj_cam_to_src_pixel = intrinsics @ pose_mat

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]

    src_pixel_coords, computed_depth, mask_1 = project_camera_to_pixel_grid(
        cam_coords, rot, tr, padding_mode, boundary)

    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    valid_mask = valid_mask * mask_1.unsqueeze(1)
    projection_penalty = torch.relu(src_pixel_coords.abs() - 1).mean()

    return projected_img, valid_mask, projection_penalty


class StructuralSimilarity(nn.Module):

    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


structural_similarity_loss = StructuralSimilarity()

def masked_average(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / (mask.sum()+1e-8)
    return mean_value

def compute_triplet_photometric_loss(
    im1,
    im2,
    im3,
    depth2,
    pose21,
    pose23,
    intrinsic,
    boundary,
    padding_mode,
    projection_penalty_weight=0.0,
):
    im12, valid_mask12, projection_penalty12 = warp_source_image_to_target(
        depth2, im1, pose21, intrinsic, boundary, padding_mode
    )
    im32, valid_mask32, projection_penalty32 = warp_source_image_to_target(
        depth2, im3, pose23, intrinsic, boundary, padding_mode
    )

    valid_mask = valid_mask12 * valid_mask32 * boundary

    diff_img1 = (im2 - im12).abs().clamp(0, 1)
    diff_img2 = (im2 - im32).abs().clamp(0, 1)

    ssim_map1 = structural_similarity_loss(im2, im12)
    ssim_map2 = structural_similarity_loss(im2, im32)

    diff_img1 = (0.15 * diff_img1 + 0.85 * ssim_map1)
    diff_img2 = (0.15 * diff_img2 + 0.85 * ssim_map2)

    diff_img = torch.min(diff_img1 ,diff_img2)

    loss = masked_average(diff_img, valid_mask)

    projection_penalty = (projection_penalty12 + projection_penalty32) / 2.0
    return loss + projection_penalty_weight * projection_penalty
