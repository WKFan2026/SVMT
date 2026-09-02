from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import cv2
import os
import random
import re


def sort_key(s):
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)


def strsort(alist):
    alist.sort(key=sort_key)
    return alist


class SequencePhotometricAugmentation:
    """Apply shared non-spatial photometric augmentation to a frame sequence."""

    def __init__(
        self,
        enabled=True,
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        gamma_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        noise_std_range=(0.0, 0.01),
    ):
        self.enabled = enabled
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.saturation_range = saturation_range
        self.noise_std_range = noise_std_range

    @staticmethod
    def sample_factor(value_range, name):
        minimum, maximum = value_range
        if minimum > maximum:
            raise ValueError(f"{name} range must be ordered as (minimum, maximum).")
        return np.float32(np.random.uniform(minimum, maximum))

    def __call__(self, imgs):
        if not self.enabled:
            return imgs

        augmented = np.stack(imgs, axis=0)
        if augmented.ndim != 4 or augmented.shape[-1] != 3:
            raise ValueError("Sequence augmentation expects images with shape [N, H, W, 3].")

        brightness = self.sample_factor(self.brightness_range, "brightness")
        augmented = augmented * brightness

        contrast = self.sample_factor(self.contrast_range, "contrast")
        sequence_mean = augmented.mean(axis=(0, 1, 2, 3), keepdims=True)
        augmented = (augmented - sequence_mean) * contrast + sequence_mean

        saturation = self.sample_factor(self.saturation_range, "saturation")
        luminance_weights = np.array((0.2989, 0.5870, 0.1140), dtype=augmented.dtype)
        luminance = (augmented * luminance_weights.reshape(1, 1, 1, 3)).sum(axis=-1, keepdims=True)
        augmented = luminance + (augmented - luminance) * saturation

        gamma = self.sample_factor(self.gamma_range, "gamma")
        augmented = np.power(np.clip(augmented, 0, 1), gamma)

        noise_std = self.sample_factor(self.noise_std_range, "noise standard deviation")
        shared_noise = np.random.normal(0, noise_std, size=(1, *augmented.shape[1:])).astype(augmented.dtype)
        augmented = np.clip(augmented + shared_noise, 0, 1)

        return list(augmented)


class TrainDatasetLoad(Dataset):
    def __init__(self, path_dir, set_intrinsic_matrix, transform, downsample=1, max_length=3, num_iter=500,
                 svmask_threshold=20, sequence_augmentation=None):
        self.path_dir = path_dir
        self.transform = transform
        self.downsample = downsample
        self.max_length = max_length
        self.num_iter = num_iter
        self.svmask_threshold = svmask_threshold
        self.intrinsic_matrix = set_intrinsic_matrix
        self.sequence_augmentation = sequence_augmentation

        self.data_imgs = []
        self.data_masks = []
        self.train_data_num = []

        for data_index in os.listdir(self.path_dir):
            image_dir = os.path.join(self.path_dir, data_index, "imgs")
            if not os.path.isdir(image_dir):
                continue

            temp_imgs = [
                image_name for image_name in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, image_name))
            ]
            temp_imgs_sorted = strsort(temp_imgs)

            if len(temp_imgs_sorted)<max_length:
                continue

            temp_imgs = []
            for i in range(len(temp_imgs_sorted)):
                temp_imgs.append(os.path.join(image_dir, temp_imgs_sorted[i]))

            temp_train_data_num = max(1, len(temp_imgs) - self.max_length + 1)

            self.data_imgs.append(temp_imgs)
            self.train_data_num.append(temp_train_data_num)

            mask_path = os.path.join(self.path_dir, data_index, "mask.png")
            if os.path.exists(mask_path):
                self.data_masks.append(mask_path)
            else:
                self.data_masks.append("None")

        self.data_num = len(self.data_imgs)
        if self.data_num == 0:
            raise ValueError("No valid training sequences were found.")

    def __len__(self):
        return self.num_iter

    def __getitem__(self, index):
        rd_data = random.randint(0, self.data_num - 1)
        rd_index = random.randint(0, self.train_data_num[rd_data] - 1)

        imgs_path = self.data_imgs[rd_data][rd_index:rd_index + self.max_length]

        imgs = []
        for img_path in imgs_path:
            img = cv2.imread(img_path).astype(np.float32) / 255.0
            img = cv2.resize(img, (0, 0), fx=1.0 / self.downsample, fy=1.0 / self.downsample,
                             interpolation=cv2.INTER_NEAREST)
            imgs.append(img)

        if self.sequence_augmentation is not None:
            imgs = self.sequence_augmentation(imgs)

        H, W, C = imgs[0].shape
        imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs, 0)

        imgs_temp = []
        for img_path in imgs_path:
            img = cv2.imread(img_path, 0).astype(np.float32)
            img = cv2.resize(img, (0, 0), fx=1.0 / self.downsample, fy=1.0 / self.downsample,
                             interpolation=cv2.INTER_NEAREST)
            imgs_temp.append(img)

        sv_masks = []
        for i in range(len(imgs_temp) - 1):
            im1 = imgs_temp[i]
            im2 = imgs_temp[i + 1]

            sv_mask = np.abs(im1 - im2)
            if self.svmask_threshold == 0:
                sv_mask = np.ones_like(sv_mask)
            else:
                sv_mask = np.where(sv_mask < self.svmask_threshold, sv_mask / self.svmask_threshold, 1)

            sv_mask = np.expand_dims(sv_mask, 0)
            sv_mask = torch.from_numpy(sv_mask)
            sv_masks.append(sv_mask)

        sv_masks = torch.stack(sv_masks, 0)

        mask_path = self.data_masks[rd_data]
        if mask_path == "None":
            boundary = np.ones((1, H, W), dtype=np.float32)
        else:
            boundary = cv2.imread(mask_path, 0)
            boundary = boundary.astype(np.float32) / 255
            boundary = cv2.resize(boundary, (0, 0), fx=1.0 / self.downsample, fy=1.0 / self.downsample,
                                  interpolation=cv2.INTER_NEAREST)
            boundary = np.expand_dims(boundary, 0)

        return imgs, self.intrinsic_matrix, boundary, sv_masks


class TestDatasetLoad(Dataset):
    def __init__(self, path_dir, transform=None, downsample=2, Max_Length=3, svmask_threshold=20):
        self.path_dir = path_dir
        self.transform = transform
        self.downsample = downsample
        self.maxlength = Max_Length
        self.svmask_threshold = svmask_threshold

        self.data_imgs = []

        temp_imgs = [
            image_name for image_name in os.listdir(self.path_dir + "/imgs/")
            if os.path.isfile(os.path.join(self.path_dir, "imgs", image_name))
        ]
        temp_imgs_sorted = strsort(temp_imgs)

        for i in range(len(temp_imgs_sorted)):
            self.data_imgs.append(self.path_dir + "/imgs/" + temp_imgs_sorted[i])
        if os.path.exists(self.path_dir + "/mask.png"):
            self.data_mask = self.path_dir + "/mask.png"
        else:
            self.data_mask = "None"

    def __len__(self):
        return len(self.data_imgs) // (self.maxlength - 1)

    def __getitem__(self, index):

        data_max_index = len(self.data_imgs)

        img_paths = self.data_imgs[
                    min(index * 2, min(index * 2 + 3, data_max_index) - 3):min(index * 2 + 3, data_max_index)]

        imgs = []

        for img_path in img_paths:
            img = cv2.imread(img_path).astype(np.float32) / 255.0
            img = cv2.resize(img, (0, 0), fx=1.0 / self.downsample, fy=1.0 / self.downsample,
                             interpolation=cv2.INTER_NEAREST)
            H, W, C = img.shape
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 0)

        imgs_temp = []
        for img_path in img_paths:
            img = cv2.imread(img_path,0).astype(np.float32)
            img = cv2.resize(img, (0, 0), fx=1.0 / self.downsample, fy=1.0 / self.downsample,
                             interpolation=cv2.INTER_NEAREST)
            imgs_temp.append(img)

        sv_masks = []
        for i in range(len(imgs_temp) - 1):
            im1 = imgs_temp[i]
            im2 = imgs_temp[i + 1]

            sv_mask = np.abs(im1 - im2)
            if self.svmask_threshold == 0:
                sv_mask = np.ones_like(sv_mask)
            else:
                sv_mask = np.where(sv_mask < self.svmask_threshold, sv_mask / self.svmask_threshold, 1)

            sv_mask = np.expand_dims(sv_mask, 0)
            sv_mask = torch.from_numpy(sv_mask)
            sv_masks.append(sv_mask)

        sv_masks = torch.stack(sv_masks, 0)

        mask_path = self.data_mask
        if mask_path == "None":
            boundary = np.ones((1, H, W), dtype=np.float32)
        else:
            boundary = cv2.imread(mask_path, 0)
            boundary = boundary.astype(np.float32) / 255
            boundary = cv2.resize(boundary, (0, 0), fx=1.0 / self.downsample, fy=1.0 / self.downsample,
                                  interpolation=cv2.INTER_NEAREST)
            boundary = np.expand_dims(boundary, 0)

        return img_paths, imgs, sv_masks, boundary
