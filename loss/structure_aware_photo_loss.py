
#Part of the code is from
# @article{ozyoruk2021endoslam,
#   title={EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos},
#   author={Ozyoruk, Kutsev Bengisu and Gokceler, Guliz Irem and Bobrow, Taylor L and Coskun, Gulfize and Incetan, Kagan and Almalioglu, Yasin and Mahmood, Faisal and Curto, Eva and Perdigoto, Luis and Oliveira, Marina and others},
#   journal={Medical image analysis},
#   volume={71},
#   pages={102058},
#   year={2021},
#   publisher={Elsevier}
# }

import torch
import torch.nn.functional as F
import torch.nn as nn

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, boundary):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
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

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w), mask_1.reshape(b, h, w)

def inverse_warp_newphoto(depth, img, pose, intrinsics, boundary, padding_mode):
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    # pose_mat = pose_vec2mat(pose)  # [B,3,4]
    pose_mat = pose

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]

    src_pixel_coords, computed_depth, mask_1 = cam2pixel2(cam_coords, rot, tr, padding_mode, boundary)  # [B,H,W,2]

    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    valid_mask = valid_mask * mask_1.unsqueeze(1)

    return projected_img, valid_mask


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSIM(nn.Module):

    def __init__(self):
        super(SSIM, self).__init__()
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


compute_ssim_loss = SSIM().to(device)

def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / (mask.sum()+1e-8)
    return mean_value

def compute_3photo_loss(im1 ,im2 ,im3 ,depth2 ,pose21 ,pose23 ,intrinsic ,boundary ,padding_mode):
    im12 ,valid_mask12 = inverse_warp_newphoto(depth2 ,im1 ,pose21 ,intrinsic ,boundary ,padding_mode)
    im32 ,valid_mask32 = inverse_warp_newphoto(depth2 ,im3 ,pose23 ,intrinsic ,boundary ,padding_mode)

    valid_mask = valid_mask12 * valid_mask32 * boundary

    diff_img1 = (im2 - im12).abs().clamp(0, 1)
    diff_img2 = (im2 - im32).abs().clamp(0, 1)

    ssim_map1 = compute_ssim_loss(im2, im12)
    ssim_map2 = compute_ssim_loss(im2, im32)

    diff_img1 = (0.15 * diff_img1 + 0.85 * ssim_map1)
    diff_img2 = (0.15 * diff_img2 + 0.85 * ssim_map2)

    diff_img = torch.min(diff_img1 ,diff_img2)

    loss = mean_on_mask(diff_img, valid_mask)

    return loss