import datetime
import json
from pathlib import Path
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import dataload
import torch
import numpy as np
import random
import torch.nn.functional as F
from model import SVMT
from loss import structure_aware_photo_loss
from loss import depth_consistency_loss as dcl


def save_model(model, optimizer, epoch, step, model_path, validation_loss, best_photo_loss,
               photo_loss_epochs_without_improvement):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'validation': validation_loss,
        'best_photo_loss': best_photo_loss,
        'photo_loss_epochs_without_improvement': photo_loss_epochs_without_improvement,
    }, str(model_path))


def reproject_depth_pair(target_depth, source_depth, translation, rotation, boundary, intrinsic,
                         depth_warping_layer):
    return depth_warping_layer([
        target_depth,
        source_depth,
        boundary,
        translation,
        rotation,
        intrinsic,
    ])


def compose_pose(rotation, translation):
    return torch.cat([rotation, translation], dim=2)


def invert_pose(rotation, translation):
    inverse_rotation = torch.inverse(rotation)
    inverse_translation = -torch.matmul(torch.inverse(rotation), translation)
    return compose_pose(inverse_rotation, inverse_translation)


project_root = Path(__file__).resolve().parent
with (project_root / "parameters_json" / "train_parameters.json").open(encoding="utf-8") as file:
    train_parameters = json.load(file)
with (project_root / "parameters_json" / "svmt_parameters.json").open(encoding="utf-8") as file:
    svmt_parameters = json.load(file)


def get_learning_rate(epoch):
    if not 0 < train_parameters["learning_rate_decay_rate"] <= 1:
        raise ValueError("learning_rate_decay_rate must be in the range (0, 1].")
    if train_parameters["initial_learning_rate"] < train_parameters["min_learning_rate"]:
        raise ValueError("initial_learning_rate must be greater than or equal to min_learning_rate.")
    return max(
        train_parameters["min_learning_rate"],
        train_parameters["initial_learning_rate"] * (train_parameters["learning_rate_decay_rate"] ** epoch),
    )


def get_svmask_threshold(epoch):
    if train_parameters["target_svmask_threshold"] < train_parameters["initial_svmask_threshold"]:
        raise ValueError("target_svmask_threshold must be at least the initial value.")
    if train_parameters["svmask_threshold_ramp_epochs"] <= 0:
        raise ValueError("svmask_threshold_ramp_epochs must be positive.")

    ramp_progress = min(
        1.0,
        epoch / train_parameters["svmask_threshold_ramp_epochs"],
    )
    return train_parameters["initial_svmask_threshold"] + ramp_progress * (
        train_parameters["target_svmask_threshold"] - train_parameters["initial_svmask_threshold"]
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

currentDT = datetime.datetime.now()
log_root = Path(train_parameters["model_save_path"]) / currentDT.strftime(
    "model-%Y-%m-%d-%H-%M-%S"
)

if not log_root.exists():
    log_root.mkdir()
print("Tensorboard visualization at {}".format(str(log_root)))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_parameters["normalization_mean"], train_parameters["normalization_std"]),
])
sequence_augmentation = dataload.SequencePhotometricAugmentation(
    enabled=train_parameters["enable_sequence_augmentation"],
    brightness_range=train_parameters["brightness_range"],
    contrast_range=train_parameters["contrast_range"],
    gamma_range=train_parameters["gamma_range"],
    saturation_range=train_parameters["saturation_range"],
    noise_std_range=train_parameters["noise_std_range"],
)

train_dataset = dataload.TrainDatasetLoad(
    train_parameters["data_path"],
    np.asarray(train_parameters["intrinsic_matrix"], dtype=np.float32),
    transform,
    train_parameters["downsample_factor"],
    train_parameters["max_length"],
    train_parameters["iterations"],
    train_parameters["initial_svmask_threshold"],
    sequence_augmentation,
)

dataloader = DataLoader(
    train_dataset,
    batch_size=train_parameters["batch_size"],
    shuffle=train_parameters["data_loader_shuffle"],
)


depth_pose_estimation_model = SVMT.StructureValidMaskTransformer(**svmt_parameters)
depth_pose_estimation_model = depth_pose_estimation_model.to(device)


optimizer = torch.optim.Adam(
    depth_pose_estimation_model.parameters(),
    lr=train_parameters["initial_learning_rate"],
)


depth_warping_layer = dcl.DepthReprojectionLayer(epsilon=train_parameters["depth_warping_epsilon"])
depth_consistency_loss_function = dcl.NormalizedPointDistanceLoss(
    height=train_parameters["height"],
    width=train_parameters["width"],
)

best_photo_loss = float("inf")
photo_loss_epochs_without_improvement = 0

if not train_parameters["model_load_path"]:
    epoch = 0
    step = 0
else:
    if Path(train_parameters["model_load_path"]).exists():
        print("Loading {:s} ...".format(train_parameters["model_load_path"]))
        state = torch.load(train_parameters["model_load_path"], map_location=device)
        step = state['step']
        epoch = state['epoch'] + 1
        depth_pose_estimation_model.load_state_dict(state['model'])
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        best_photo_loss = state.get('best_photo_loss', float("inf"))
        photo_loss_epochs_without_improvement = state.get('photo_loss_epochs_without_improvement', 0)
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained model detected")
        raise OSError

if train_parameters["photo_loss_early_stopping_patience"] <= 0:
    raise ValueError("photo_loss_early_stopping_patience must be positive.")
if train_parameters["gradient_clip_norm"] is not None and train_parameters["gradient_clip_norm"] <= 0:
    raise ValueError("gradient_clip_norm must be positive or null.")

for epoch in range(epoch, train_parameters["epochs"] + 1):

    print("current epoch:",epoch)

    current_learning_rate = get_learning_rate(epoch)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = current_learning_rate

    train_dataset.svmask_threshold = get_svmask_threshold(epoch)

    torch.manual_seed(train_parameters["random_seed"] + epoch)
    np.random.seed(train_parameters["random_seed"] + epoch)
    random.seed(train_parameters["random_seed"] + epoch)

    depth_pose_estimation_model.train()

    tq = tqdm.tqdm(total=len(dataloader) * train_parameters["batch_size"], dynamic_ncols=True, ncols=40)

    for batch, (imgs, intrinsic, boundary, sv_masks) in enumerate(dataloader):

        imgs = imgs.to(device)
        boundary = boundary.to(device)
        intrinsic = intrinsic.to(device)
        sv_masks = sv_masks.to(device)

        photo_loss = imgs.new_zeros(())
        depth_consistency_loss = imgs.new_zeros(())

        sv_masks_model = F.interpolate(
            sv_masks,
            size=(1, train_parameters["height"] // 2, train_parameters["width"] // 2),
        )
        depths,Rs,Ts = depth_pose_estimation_model(imgs,sv_masks_model)

        for i in range(train_parameters["max_length"] - 2):

            colors_1 = imgs[:, i, :, :, :]
            colors_2 = imgs[:, i + 1, :, :, :]
            colors_3 = imgs[:, i + 2, :, :, :]

            predicted_depth_maps_1 = depths[:, i, :, :].unsqueeze(0)
            predicted_depth_maps_2 = depths[:, i + 1, :, :].unsqueeze(0)
            predicted_depth_maps_3 = depths[:, i + 2, :, :].unsqueeze(0)

            sv_mask_1 = sv_masks[:,i,:,:,:]
            sv_mask_2 = sv_masks[:,i + 1,:,:,:]
            sv_mask_for_loss = torch.min(sv_mask_1,sv_mask_2)

            boundary_pcl = boundary * sv_mask_for_loss

            rotations_2_wrt_1 = Rs[:, i, :, :]
            translations_2_wrt_1 = Ts[:, i, :, :]
            rotations_3_wrt_2 = Rs[:, i + 1, :, :]
            translations_3_wrt_2 = Ts[:, i + 1, :, :]

            pose21 = invert_pose(rotations_2_wrt_1, translations_2_wrt_1)
            pose23 = compose_pose(rotations_3_wrt_2, translations_3_wrt_2)
            pose32 = invert_pose(rotations_3_wrt_2, translations_3_wrt_2)

            photo_loss = photo_loss + structure_aware_photo_loss.compute_triplet_photometric_loss(colors_1, colors_2, colors_3, predicted_depth_maps_2,
                                                             pose21, pose23, intrinsic, boundary_pcl,
                                                             train_parameters["photometric_padding_mode"],
                                                             train_parameters["projection_penalty_weight"])

            warped_depth_maps_2_to_1, intersect_masks_1 = reproject_depth_pair(
                predicted_depth_maps_1, predicted_depth_maps_2, pose21[:, :, -1:], pose21[:, :, :3],
                boundary, intrinsic, depth_warping_layer
            )
            warped_depth_maps_1_to_2, intersect_masks_2 = reproject_depth_pair(
                predicted_depth_maps_2, predicted_depth_maps_1, translations_2_wrt_1, rotations_2_wrt_1,
                boundary, intrinsic, depth_warping_layer
            )
            warped_depth_maps_3_to_2, intersect_masks_3 = reproject_depth_pair(
                predicted_depth_maps_2, predicted_depth_maps_3, pose32[:, :, -1:], pose32[:, :, :3],
                boundary, intrinsic, depth_warping_layer
            )
            warped_depth_maps_2_to_3, intersect_masks_4 = reproject_depth_pair(
                predicted_depth_maps_3, predicted_depth_maps_2, translations_3_wrt_2, rotations_3_wrt_2,
                boundary, intrinsic, depth_warping_layer
            )

            depth_consistency_loss = depth_consistency_loss + (
                depth_consistency_loss_function([predicted_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1, intrinsic]) +
                depth_consistency_loss_function([predicted_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2, intrinsic]) +
                depth_consistency_loss_function([predicted_depth_maps_2, warped_depth_maps_3_to_2, intersect_masks_3, intrinsic]) +
                depth_consistency_loss_function([predicted_depth_maps_3, warped_depth_maps_2_to_3, intersect_masks_4, intrinsic])
            ) / 4.0

        loss = (train_parameters["photo_weight"] * photo_loss
                + train_parameters["depth_consistency_weight"] * depth_consistency_loss)

        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite training loss.")

        optimizer.zero_grad()
        loss.backward()
        if train_parameters["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                depth_pose_estimation_model.parameters(),
                train_parameters["gradient_clip_norm"],
            )
        optimizer.step()

        if batch == 0:
            mean_loss = loss.item()
            mean_depth_consistency_loss = depth_consistency_loss.item()
            mean_photo_loss = photo_loss.item()

        else:
            mean_loss = (mean_loss * batch + loss.item()) / (batch + 1.0)
            mean_depth_consistency_loss = (mean_depth_consistency_loss * batch +
                                           depth_consistency_loss.item()) / (batch + 1.0)
            mean_photo_loss = (mean_photo_loss * batch + photo_loss.item()) / (batch + 1.0)

        step += 1
        tq.update(train_parameters["batch_size"])

        tq.set_postfix(loss="avg: {:.5f} cur: {:.5f}".format(mean_loss, loss.item()),
                       loss_depth_consistency='avg: {:.5f} cur: {:.5f}'.format(
                           mean_depth_consistency_loss,
                           depth_consistency_loss.item()),
                       loss_photo='avg: {:.5f} cur: {:.5f}'.format(
                           mean_photo_loss,
                           photo_loss.item()),
                       )

    should_stop_early = False
    if mean_photo_loss < best_photo_loss:
        best_photo_loss = mean_photo_loss
        photo_loss_epochs_without_improvement = 0

        best_model_path = log_root / 'checkpoint_best_photo_model.pt'
        save_model(model=depth_pose_estimation_model, optimizer=optimizer,
                   epoch=epoch, step=step, model_path=best_model_path,
                   validation_loss=mean_loss,
                   best_photo_loss=best_photo_loss,
                   photo_loss_epochs_without_improvement=photo_loss_epochs_without_improvement)
    else:
        photo_loss_epochs_without_improvement += 1
        should_stop_early = (
            photo_loss_epochs_without_improvement >= train_parameters["photo_loss_early_stopping_patience"]
        )

    latest_model_path = log_root / 'checkpoint_latest_model.pt'
    save_model(model=depth_pose_estimation_model, optimizer=optimizer,
               epoch=epoch, step=step, model_path=latest_model_path,
               validation_loss=mean_loss,
               best_photo_loss=best_photo_loss,
               photo_loss_epochs_without_improvement=photo_loss_epochs_without_improvement)

    tq.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if should_stop_early:
        print(
            "Early stopping: mean_photo_loss did not improve for {} consecutive epochs.".format(
                train_parameters["photo_loss_early_stopping_patience"]
            )
        )
        break

print("over")
