import json
from pathlib import Path
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
import os
import dataload
from model import SVMT
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent
with (project_root / "parameters_json" / "test_parameters.json").open(encoding="utf-8") as file:
    test_parameters = json.load(file)
with (project_root / "parameters_json" / "svmt_parameters.json").open(encoding="utf-8") as file:
    svmt_parameters = json.load(file)

normalization_mean = np.asarray(test_parameters["normalization_mean"], dtype=np.float32).reshape(1, 1, 3)
normalization_std = np.asarray(test_parameters["normalization_std"], dtype=np.float32).reshape(1, 1, 3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_data = [
    data_index for data_index in os.listdir(test_parameters["test_data_root"])
    if os.path.isdir(os.path.join(test_parameters["test_data_root"], data_index))
]

for data_index in all_data:

    Data_Path = test_parameters["test_data_root"] + data_index
    result_path = Data_Path + "/" + test_parameters["save_index"] + "/depth_map/"
    result_pose_path = Data_Path + "/" + test_parameters["save_index"] + "/pose/"
    if not os.path.exists(Data_Path + "/" + test_parameters["save_index"]):
        os.mkdir(Data_Path + "/" + test_parameters["save_index"])
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(result_pose_path):
        os.mkdir(result_pose_path)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(test_parameters["normalization_mean"],
                                                     test_parameters["normalization_std"])])
    my_dataset = dataload.TestDatasetLoad(
        Data_Path,
        transform,
        test_parameters["downsample_factor"],
        test_parameters["max_length"],
        test_parameters["svmask_threshold"],
    )
    dataloader = DataLoader(
        my_dataset,
        batch_size=test_parameters["batch_size"],
        shuffle=test_parameters["data_loader_shuffle"],
    )

    depth_pose_estimation_model = SVMT.StructureValidMaskTransformer(**svmt_parameters)
    depth_pose_estimation_model = depth_pose_estimation_model.to(device)

    if Path(test_parameters["trained_depth_model_path"]).exists():
        print("Loading {:s} ...".format(test_parameters["trained_depth_model_path"]))
        state = torch.load(test_parameters["trained_depth_model_path"])
        step = state['step']
        epoch = state['epoch']
        depth_pose_estimation_model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained model detected")
        raise OSError


    with torch.no_grad():

        depth_pose_estimation_model.eval()

        tq = tqdm.tqdm(total=len(dataloader) * test_parameters["batch_size"])

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

            sv_masks_model = F.interpolate(
                sv_masks,
                size=(1, test_parameters["height"] // 2, test_parameters["width"] // 2),
            )
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


                color_display = np.uint8(255 * (
                    colors_1[0].permute(1, 2, 0).data.cpu().numpy() * normalization_std
                    + normalization_mean
                ).reshape((test_parameters["height"], test_parameters["width"], 3)))

                boundary1 = boundary[0].data.cpu().numpy().reshape(
                    (test_parameters["height"], test_parameters["width"])
                )
                color_display = np.uint8(boundary1.reshape(
                    (test_parameters["height"], test_parameters["width"], 1)
                ) * color_display)


                depth_map = predicted_depth_maps_1[0].data.cpu().numpy().reshape(
                    (test_parameters["height"], test_parameters["width"])
                )
                depth_map = depth_map * boundary1

                depth_display = cv2.applyColorMap(np.uint8(255 * depth_map / np.max(depth_map)), cv2.COLORMAP_JET)

                np.save(result_path_index + img_index.split(".")[0] + ".npy", depth_map)
                cv2.imwrite(result_path_index + img_index, cv2.hconcat([color_display, depth_display]))

            tq.update(test_parameters["batch_size"])
    tq.close()
