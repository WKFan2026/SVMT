import datetime
import utils
from pathlib import Path
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import random
import cv2
import os
import dataload
from model import SVMT
import torch.nn.functional as F

test_data_root = "test_data/"
trained_depth_model_path = "trained_model_path/model-2026-05-16-01-51-35/checkpoint_depth_model.pt"
save_index = "model-2026-05-16-01-51-35"

set_batch_size = 1
set_downsample_factor = 1.0
set_max_length = 3
set_height = 256
set_width = 320
set_automask_threshold = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_data = os.listdir(test_data_root)

for data_index in all_data:

    Data_Path = test_data_root + data_index
    result_path = Data_Path + "/" + save_index + "/depth_map/"
    result_pose_path = Data_Path + "/" + save_index + "/pose/"
    if not os.path.exists(Data_Path + "/" + save_index ):
        os.mkdir(Data_Path + "/" + save_index)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(result_pose_path):
        os.mkdir(result_pose_path)

    # dataload
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])
    my_dataset = dataload.TestDatasetLoad(Data_Path, transform, set_downsample_factor, set_max_length, set_automask_threshold)
    dataloader = DataLoader(my_dataset, batch_size=set_batch_size, shuffle=False)

    # model
    depth_pose_estimation_model = SVMT.StructureValidMaskTransformer()
    depth_pose_estimation_model = depth_pose_estimation_model.to(device)

    if Path(trained_depth_model_path).exists():
        print("Loading {:s} ...".format(trained_depth_model_path))
        state = torch.load(trained_depth_model_path)
        step = state['step']
        epoch = state['epoch']
        depth_pose_estimation_model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained model detected")
        raise OSError


    with torch.no_grad():

        depth_pose_estimation_model.eval()

        tq = tqdm.tqdm(total=len(dataloader) * set_batch_size)

        flag = []
        scale_value = 1
        for index, (img_paths, imgs, sv_masks, boundary) in enumerate(dataloader):

            result_path_index = result_path + str(index) + "/"
            result_pose_path_index = result_pose_path + str(index) + "/"

            if not os.path.exists(result_path_index):
                os.mkdir(result_path_index)
            if not os.path.exists(result_pose_path_index):
                os.mkdir(result_pose_path_index)

            imgs = imgs.to(device)
            boundary = boundary.to(device)
            sv_masks = sv_masks.to(device)

            sv_masks_model = F.interpolate(sv_masks, size=(1, set_height // 2, set_width // 2))
            depths, Rs123, Ts123 = depth_pose_estimation_model(imgs, sv_masks_model)

            Rs = Rs123.data.cpu().numpy()
            Ts = Ts123.data.cpu().numpy()
            Rs = Rs[0]
            Ts = Ts[0]
            np.save(result_pose_path_index + "rotations.npy", Rs)
            np.save(result_pose_path_index + "motions.npy", Ts)


            for i in range(len(img_paths)):
                colors_1 = imgs[:, i, :, :, :]


                img_path = img_paths[i]
                img_index = img_path[0].split("/")[-1]

                predicted_depth_maps_1 = depths[:, i, :, :]
                predicted_depth_maps_1 = predicted_depth_maps_1.unsqueeze(0)


                color_display = np.uint8(
                    255 * (0.225 * colors_1[0].permute(1, 2, 0).data.cpu().numpy() + 0.45).reshape((set_height, set_width, 3)))

                boundary1 = boundary[0].data.cpu().numpy().reshape((set_height, set_width))
                color_display = np.uint8(boundary1.reshape((set_height, set_width, 1)) * color_display)


                depth_map = predicted_depth_maps_1[0].data.cpu().numpy().reshape((set_height, set_width))
                depth_map = depth_map * boundary1

                depth_display = cv2.applyColorMap(np.uint8(255 * depth_map / np.max(depth_map)), cv2.COLORMAP_JET)

                np.save(result_path_index + img_index.split(".")[0] + ".npy", depth_map)
                cv2.imwrite(result_path_index + img_index, cv2.hconcat([color_display, depth_display]))

            tq.update(set_batch_size)
    tq.close()
