import torch
import torch.nn as nn


def sample_bilinear_map(im, x, y, padding_mode="zeros"):
    num_batch, height, width, channels = im.shape
    two = torch.tensor(2.0, dtype=torch.float32, device=x.device)
    one = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    width_tensor = torch.tensor(width, dtype=torch.float32, device=x.device)
    height_tensor = torch.tensor(height, dtype=torch.float32, device=x.device)
    grid = torch.cat([
        two * (x.view(num_batch, height, width, 1) / width_tensor) - one,
        two * (y.view(num_batch, height, width, 1) / height_tensor) - one,
    ], dim=-1)

    return torch.nn.functional.grid_sample(input=im.permute(0, 3, 1, 2), grid=grid, mode='bilinear',
                                           padding_mode=padding_mode,align_corners=True).permute(0, 2, 3, 1)

class DepthReprojectionLayer(torch.nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super().__init__()
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=torch.float32), persistent=False)

    def forward(self, x):
        depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        epsilon = self.epsilon.to(device=depth_maps_1.device)
        warped_depth_maps, intersect_masks = reproject_depth_maps(depth_maps_1, depth_maps_2, img_masks,
                                                                   translation_vectors,
                                                                   rotation_matrices, intrinsic_matrices, epsilon)
        return warped_depth_maps, intersect_masks


def reproject_depth_maps(depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices,
                         intrinsic_matrices, epsilon):
    depth_maps_1 = torch.mul(depth_maps_1, img_masks)
    depth_maps_2 = torch.mul(depth_maps_2, img_masks)

    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    depth_maps_2 = depth_maps_2.permute(0, 2, 3, 1)

    img_masks = img_masks.permute(0, 2, 3, 1)

    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32, device=depth_maps_1.device),
         torch.arange(start=0, end=width, dtype=torch.float32, device=depth_maps_1.device)],
        indexing='ij')

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32, device=depth_maps_1.device)

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
    depth_maps_2_calculate = torch.where(img_masks > 0.5, depth_maps_2_calculate, epsilon)
    depth_maps_2_calculate = torch.where(depth_maps_2_calculate > 0.0, depth_maps_2_calculate, epsilon)

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

    warped_depth_maps_2 = sample_bilinear_map(depth_maps_1_calculate, u_2_flat, v_2_flat).view(
        num_batch, 1, height, width)

    intersect_masks = torch.where(sample_bilinear_map(img_masks, u_2_flat, v_2_flat) * img_masks >= 0.9,
                                  torch.ones((), dtype=torch.float32, device=depth_maps_1.device),
                                  torch.zeros((), dtype=torch.float32, device=depth_maps_1.device)).view(
                                      num_batch, 1, height, width)

    return [warped_depth_maps_2, intersect_masks]


class NormalizedPointDistanceLoss(nn.Module):
    def __init__(self, height, width, eps=1.0e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('y_grid', torch.empty(0), persistent=False)
        self.register_buffer('x_grid', torch.empty(0), persistent=False)
        self._set_coordinate_grids(height, width, torch.device('cpu'))

    def _set_coordinate_grids(self, height, width, device):
        y_grid, x_grid = torch.meshgrid(
            [torch.arange(start=0, end=height, dtype=torch.float32, device=device),
             torch.arange(start=0, end=width, dtype=torch.float32, device=device)],
            indexing='ij')
        self.y_grid = y_grid.view(1, 1, height, width)
        self.x_grid = x_grid.view(1, 1, height, width)

    def forward(self, x):
        depth_maps, warped_depth_maps, intersect_masks, intrinsics = x
        height, width = depth_maps.shape[-2:]
        if (self.y_grid.shape[-2:] != (height, width)) or self.y_grid.device != depth_maps.device:
            self._set_coordinate_grids(height, width, depth_maps.device)

        fx = intrinsics[:, 0, 0].view(-1, 1, 1, 1)
        fy = intrinsics[:, 1, 1].view(-1, 1, 1, 1)
        cx = intrinsics[:, 0, 2].view(-1, 1, 1, 1)
        cy = intrinsics[:, 1, 2].view(-1, 1, 1, 1)

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
