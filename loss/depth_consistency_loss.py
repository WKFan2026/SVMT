#Part of the code is from
# @article{liu2019dense,
#   title={Dense depth estimation in monocular endoscopy with self-supervised learning methods},
#   author={Liu, Xingtong and Sinha, Ayushi and Ishii, Masaru and Hager, Gregory D and Reiter, Austin and Taylor, Russell H and Unberath, Mathias},
#   journal={IEEE transactions on medical imaging},
#   volume={39},
#   number={5},
#   pages={1438--1447},
#   year={2019},
#   publisher={IEEE}
# }

import torch
import torch.nn as nn
def _bilinear_interpolate(im, x, y, padding_mode="zeros"):
    num_batch, height, width, channels = im.shape
    # Range [-1, 1]
    grid = torch.cat([torch.tensor(2.0).float().cuda() *
                      (x.view(num_batch, height, width, 1) / torch.tensor(width).float().cuda())
                      - torch.tensor(1.0).float().cuda(), torch.tensor(2.0).float().cuda() * (
                              y.view(num_batch, height, width, 1) / torch.tensor(height).float().cuda()) - torch.tensor(
        1.0).float().cuda()], dim=-1)

    return torch.nn.functional.grid_sample(input=im.permute(0, 3, 1, 2), grid=grid, mode='bilinear',
                                           padding_mode=padding_mode,align_corners=True).permute(0, 2, 3, 1)

class DepthWarpingLayer(torch.nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(DepthWarpingLayer, self).__init__()
        self.zero = torch.tensor(0.0).float().cuda()
        self.epsilon = torch.tensor(epsilon).float().cuda()

    def forward(self, x):
        depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        warped_depth_maps, intersect_masks = _depth_warping(depth_maps_1, depth_maps_2, img_masks,
                                                            translation_vectors,
                                                            rotation_matrices, intrinsic_matrices, self.epsilon)
        return warped_depth_maps, intersect_masks


# Warping depth map in coordinate system 2 to coordinate system 1
def _depth_warping(depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices,
                   intrinsic_matrices, epsilon):
    # Generate a meshgrid for each depth map to calculate value

    depth_maps_1 = torch.mul(depth_maps_1, img_masks)
    depth_maps_2 = torch.mul(depth_maps_2, img_masks)

    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    depth_maps_2 = depth_maps_2.permute(0, 2, 3, 1)

    img_masks = img_masks.permute(0, 2, 3, 1)

    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda()])

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32).cuda()

    eye = torch.eye(3).float().cuda().view(1, 3, 3).expand(intrinsic_matrices.shape[0], -1, -1)
    intrinsic_matrices_inverse = torch.inverse(intrinsic_matrices)
    rotation_matrices_inverse = rotation_matrices.transpose(1, 2)

    temp_mat = torch.bmm(intrinsic_matrices, rotation_matrices_inverse)
    W = torch.bmm(temp_mat, -translation_vectors)
    M = torch.bmm(temp_mat, intrinsic_matrices_inverse)

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).view(height, width, 3, 1)
    intermediate_result = torch.matmul(M.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3)

    depth_maps_2_calculate = W.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).view(-1, height,
                                                                   width, 1))
    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.where(img_masks > 0.5, depth_maps_2_calculate, epsilon)
    depth_maps_2_calculate = torch.where(depth_maps_2_calculate > 0.0, depth_maps_2_calculate, epsilon)

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.view(-1, 3).narrow(dim=-1, start=0, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=0,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / (
              depth_maps_2_calculate)

    v_2 = (W.view(-1, 3).narrow(dim=-1, start=1, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=1,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / (
              depth_maps_2_calculate)

    W_2 = torch.bmm(intrinsic_matrices, translation_vectors)
    M_2 = torch.bmm(torch.bmm(intrinsic_matrices, rotation_matrices), intrinsic_matrices_inverse)

    temp = torch.matmul(M_2.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3).narrow(dim=-1, start=2,
                                                                                               length=1).view(-1,
                                                                                                              height,
                                                                                                              width, 1)
    depth_maps_1_calculate = W_2.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_2, temp)
    depth_maps_1_calculate = torch.mul(img_masks, depth_maps_1_calculate)

    u_2_flat = u_2.view(-1)
    v_2_flat = v_2.view(-1)

    warped_depth_maps_2 = _bilinear_interpolate(depth_maps_1_calculate, u_2_flat, v_2_flat).view(num_batch, 1, height,
                                                                                                 width)

    # binarize
    intersect_masks = torch.where(_bilinear_interpolate(img_masks, u_2_flat, v_2_flat) * img_masks >= 0.9,
                                  torch.tensor(1.0).float().cuda(),
                                  torch.tensor(0.0).float().cuda()).view(num_batch, 1, height, width)

    return [warped_depth_maps_2, intersect_masks]


class NormalizedDistanceLoss(nn.Module):
    def __init__(self, height, width, eps=1.0e-5):
        super(NormalizedDistanceLoss, self).__init__()
        self.eps = eps
        self.y_grid, self.x_grid = torch.meshgrid(
            [torch.arange(start=0, end=height, dtype=torch.float32).cuda(),
             torch.arange(start=0, end=width, dtype=torch.float32).cuda()])
        self.y_grid = self.y_grid.view(1, 1, height, width)
        self.x_grid = self.x_grid.view(1, 1, height, width)

    def forward(self, x):
        depth_maps, warped_depth_maps, intersect_masks, intrinsics = x
        fx = intrinsics[:, 0, 0].view(-1, 1, 1, 1)
        fy = intrinsics[:, 1, 1].view(-1, 1, 1, 1)
        cx = intrinsics[:, 0, 2].view(-1, 1, 1, 1)
        cy = intrinsics[:, 1, 2].view(-1, 1, 1, 1)

        with torch.no_grad():
            mean_value = torch.sum(intersect_masks * depth_maps, dim=(1, 2, 3), keepdim=False) / (
                    self.eps + torch.sum(intersect_masks, dim=(1, 2, 3),
                                         keepdim=False))

        location_3d_maps = torch.cat(
            [(self.x_grid - cx) / fx * depth_maps, (self.y_grid - cy) / fy * depth_maps, depth_maps], dim=1)

        warped_location_3d_maps = torch.cat(
            [(self.x_grid - cx) / fx * warped_depth_maps, (self.y_grid - cy) / fy * warped_depth_maps,
             warped_depth_maps], dim=1)

        loss = 2.0 * torch.sum(intersect_masks * torch.abs(location_3d_maps - warped_location_3d_maps), dim=(1, 2, 3),
                               keepdim=False) / \
               (1.0e-8 + torch.sum(
                   intersect_masks * (depth_maps + torch.abs(warped_depth_maps)), dim=(1, 2, 3),
                   keepdim=False))
        return torch.mean(loss)