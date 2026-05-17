import datetime
from pathlib import Path
import tqdm
from huggingface_hub import save_torch_model
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import dataload
import torch
import numpy as np
import random
import torch.nn.functional as F
from model import SVMT
import utils
from loss import structure_aware_photo_loss
from loss import depth_consistency_loss as dcl

set_data_path = "train_data/"
set_model_save_path = "trained_model_path/" 
set_model_load_flag = True # True：resume training from the pretrained checkpoint
set_model_load_path = "trained_model_path/model-2026-05-16-01-51-35/checkpoint_depth_model.pt"
set_automask_threshold = 20
set_downsample_factor = 1
set_max_length = 3
set_batchsize = 1
set_epochs = 500
set_iterations = 500
set_learning_rate = 1e-6
set_stage1_epochs = 30 
set_depth_consistency_weight = 2
set_photo_weight = 20
set_height = 256
set_width = 320
set_intrinsic_matrix = np.array([[586.385/4, 0.0, (695.637-44)/4],
                                  [0.0, 582.765/4, (543.701-28)/4],
                                 [0.0, 0.0, 1.0]],dtype=np.float32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

currentDT = datetime.datetime.now()
log_root = Path(set_model_save_path) / currentDT.strftime(
    "model-%Y-%m-%d-%H-%M-%S"
)

if not log_root.exists():
    log_root.mkdir()
print("Tensorboard visualization at {}".format(str(log_root)))

loss_record_root = log_root / "loss_record"
if not loss_record_root.exists():
    loss_record_root.mkdir()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])

train_dataset = dataload.TrainDatasetLoad(set_data_path,
                                           set_intrinsic_matrix,
                                           transform,
                                           set_downsample_factor,
                                           set_max_length,
                                           set_iterations,
                                           set_automask_threshold)

dataloader = DataLoader(train_dataset, batch_size=set_batchsize, shuffle=False)


depth_pose_estimation_model = SVMT.StructureValidMaskTransformer()
depth_pose_estimation_model = depth_pose_estimation_model.to(device)


#optimization settings
optim_params = [
    {'params': depth_pose_estimation_model.parameters(), 'lr': set_learning_rate},
]
optimizer = torch.optim.Adam(optim_params)


depth_warping_layer = dcl.DepthWarpingLayer(epsilon=1e-8)
depth_consistency_loss_function = dcl.NormalizedDistanceLoss(height=set_height, width=set_width)

# load model
if set_model_load_flag is False:
    epoch = 0
    step = 0
else:
    if Path(set_model_load_path).exists():
        print("Loading {:s} ...".format(set_model_load_path))
        state = torch.load(set_model_load_path)
        step = state['step']
        epoch = state['epoch']
        depth_pose_estimation_model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained model detected")
        raise OSError

save_pcl_flag = 1000
# training
for epoch in range(epoch, set_epochs + 1):

    print("current epoch:",epoch)

    torch.manual_seed(10086 + epoch)
    np.random.seed(10086 + epoch)
    random.seed(10086 + epoch)

    depth_pose_estimation_model.train()

    tq = tqdm.tqdm(total=len(dataloader) * set_batchsize, dynamic_ncols=True, ncols=40)

    for batch, (imgs, intrinsic, boundary, sv_masks) in enumerate(dataloader):

        imgs = imgs.to(device)
        boundary = boundary.to(device)
        intrinsic = intrinsic.to(device)
        sv_masks = sv_masks.to(device)

        photo_loss = torch.tensor(0)
        depth_consistency_loss = torch.tensor(0)

        sv_masks_model = F.interpolate(sv_masks,size = (1,set_height//2,set_width//2))
        depths,Rs,Ts = depth_pose_estimation_model(imgs,sv_masks_model)

        for i in range(set_max_length - 2):

            colors_1 = imgs[:, i, :, :, :]
            colors_2 = imgs[:, i + 1, :, :, :]
            colors_3 = imgs[:, i + 2, :, :, :]

            predicted_depth_maps_1 = depths[:, i, :, :].unsqueeze(0)
            predicted_depth_maps_2 = depths[:, i + 1, :, :].unsqueeze(0)
            predicted_depth_maps_3 = depths[:, i + 2, :, :].unsqueeze(0)

            #mask
            sv_mask_1 = sv_masks[:,i,:,:,:]
            sv_mask_2 = sv_masks[:,i + 1,:,:,:]
            sv_mask_for_loss = torch.min(sv_mask_1,sv_mask_2)

            if epoch<set_stage1_epochs:
                boundary_pcl = boundary
            else:
                boundary_pcl = boundary * sv_mask_for_loss

            #R T
            rotations_2_wrt_1 = Rs[:, i, :, :]
            translations_2_wrt_1 = Ts[:, i, :, :]
            rotations_3_wrt_2 = Rs[:, i + 1, :, :]
            translations_3_wrt_2 = Ts[:, i + 1, :, :]

            rotations_1_wrt_2 = torch.inverse(rotations_2_wrt_1)
            translations_1_wrt_2 = -torch.matmul(torch.inverse(rotations_2_wrt_1), translations_2_wrt_1)
            rotations_2_wrt_3 = torch.inverse(rotations_3_wrt_2)
            translations_2_wrt_3 = -torch.matmul(torch.inverse(rotations_3_wrt_2), translations_3_wrt_2)

            #R T for PCL
            pose12 = torch.cat([rotations_2_wrt_1, translations_2_wrt_1], dim=2)
            pose21 = torch.cat([rotations_1_wrt_2, translations_1_wrt_2], dim=2)
            pose23 = torch.cat([rotations_3_wrt_2, translations_3_wrt_2], dim=2)
            pose32 = torch.cat([rotations_2_wrt_3, translations_2_wrt_3], dim=2)

            photo_loss = photo_loss + structure_aware_photo_loss.compute_3photo_loss(colors_1, colors_2, colors_3, predicted_depth_maps_2,
                                                             pose21, pose23, intrinsic, boundary_pcl, "zeros")

            # dcl
            warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer([predicted_depth_maps_1,
                                                                               predicted_depth_maps_2,
                                                                               boundary,
                                                                               translations_1_wrt_2,
                                                                               rotations_1_wrt_2,
                                                                               intrinsic])

            warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer([predicted_depth_maps_2,
                                                                               predicted_depth_maps_1,
                                                                               boundary,
                                                                               translations_2_wrt_1,
                                                                               rotations_2_wrt_1,
                                                                               intrinsic])

            warped_depth_maps_3_to_2, intersect_masks_3 = depth_warping_layer([predicted_depth_maps_2,
                                                                               predicted_depth_maps_3,
                                                                               boundary,
                                                                               translations_2_wrt_3,
                                                                               rotations_2_wrt_3,
                                                                               intrinsic])

            warped_depth_maps_2_to_3, intersect_masks_4 = depth_warping_layer([predicted_depth_maps_3,
                                                                               predicted_depth_maps_2,
                                                                               boundary,
                                                                               translations_3_wrt_2,
                                                                               rotations_3_wrt_2,
                                                                               intrinsic])

            depth_consistency_loss = depth_consistency_loss + (depth_consistency_loss_function(
                [predicted_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1, intrinsic]) +
                                                               depth_consistency_loss_function(
                                                                   [predicted_depth_maps_2, warped_depth_maps_1_to_2,
                                                                    intersect_masks_2, intrinsic]) +
                                                               depth_consistency_loss_function(
                                                                   [predicted_depth_maps_2, warped_depth_maps_3_to_2,
                                                                    intersect_masks_3, intrinsic]) +
                                                               depth_consistency_loss_function(
                                                                   [predicted_depth_maps_3, warped_depth_maps_2_to_3,
                                                                    intersect_masks_4, intrinsic])) / 4.0

        # total loss

        loss = (set_photo_weight * photo_loss
                + set_depth_consistency_weight * depth_consistency_loss)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #
        if batch == 0:
            mean_loss = loss.item()
            mean_depth_consistency_loss = depth_consistency_loss.item()
            mean_photo_loss = photo_loss.item()

        else:
            mean_loss = (mean_loss * batch + loss.item()) / (batch + 1.0)
            mean_depth_consistency_loss = (mean_depth_consistency_loss * batch +
                                           depth_consistency_loss.item()) / (batch + 1.0)
            mean_photo_loss = (mean_photo_loss * batch + photo_loss.item()) / (batch + 1.0)

        #
        step += 1
        tq.update(set_batchsize)

        tq.set_postfix(loss="avg: {:.5f} cur: {:.5f}".format(mean_loss, loss.item()),
                       loss_depth_consistency='avg: {:.5f} cur: {:.5f}'.format(
                           mean_depth_consistency_loss,
                           depth_consistency_loss.item()),
                       loss_photo='avg: {:.5f} cur: {:.5f}'.format(
                           mean_photo_loss,
                           photo_loss.item()),
                       )

    #save
    if mean_photo_loss < save_pcl_flag:
        save_pcl_flag = mean_photo_loss

        model_path_epoch = log_root / 'checkpoint_depth_model.pt'
        utils.save_model(model=depth_pose_estimation_model, optimizer=optimizer,
                         epoch=epoch, step=step, model_path=model_path_epoch,
                         validation_loss=mean_loss)

    tq.close()
    torch.cuda.empty_cache()

print("over")
