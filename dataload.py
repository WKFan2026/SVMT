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


class TrainDatasetLoad(Dataset):
    def __init__(self, path_dir, set_intrinsic_matrix, transform, downsample=1, max_length=3, num_iter=500, automask_threshold=20):
        self.path_dir = path_dir
        self.transform = transform
        self.downsample = downsample
        self.max_length = max_length
        self.num_iter = num_iter
        self.automask_threshold = automask_threshold
        self.intrinsic_matrix = set_intrinsic_matrix


        #all data
        self.data_num = len(os.listdir(self.path_dir))
        self.data_imgs = []
        self.data_masks = []

        # all groups (three frames)
        self.train_data_num = []

        for data_index in os.listdir(self.path_dir):

            temp_imgs = os.listdir(self.path_dir + data_index + "/imgs/")
            temp_imgs_sorted = strsort(temp_imgs)

            if len(temp_imgs_sorted)<max_length:
                continue

            temp_imgs = []
            for i in range(len(temp_imgs_sorted)):
                temp_imgs.append(self.path_dir + data_index + "/imgs/" + temp_imgs_sorted[i])

            temp_train_data_num = max(1, len(temp_imgs) - self.max_length + 1)

            self.data_imgs.append(temp_imgs)
            self.train_data_num.append(temp_train_data_num)

            # 存放mask，分为有和无，无的话后面迭代的时候返回一个全1的mask
            if os.path.exists(self.path_dir + data_index + "/mask.png"):
                self.data_masks.append(self.path_dir + data_index + "/mask.png")
            else:
                self.data_masks.append("None")

    def __len__(self):
        return self.num_iter

    def __getitem__(self, index):
        # random data
        rd_data = random.randint(0, self.data_num - 1)
        # random data start
        rd_index = random.randint(0, self.train_data_num[rd_data] - 1)

        # imgs load
        imgs_path = self.data_imgs[rd_data][rd_index:rd_index + self.max_length]

        imgs = []
        for img_path in imgs_path:
            img = cv2.imread(img_path).astype(np.float32) / 255.0
            img = cv2.resize(img, (0, 0), fx=1.0 / self.downsample, fy=1.0 / self.downsample,
                             interpolation=cv2.INTER_NEAREST)
            H, W, C = img.shape
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 0)

        #svmask
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
            sv_mask = np.where(sv_mask < self.automask_threshold, sv_mask / self.automask_threshold, 1)

            sv_mask = np.expand_dims(sv_mask, 0)
            sv_mask = torch.from_numpy(sv_mask)
            sv_masks.append(sv_mask)

        sv_masks = torch.stack(sv_masks, 0)

        #img mask
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
    def __init__(self, path_dir, transform=None, downsample=2, Max_Length=3, automask_threshold=20):
        self.path_dir = path_dir
        self.transform = transform
        self.downsample = downsample
        self.maxlength = Max_Length
        self.automask_threshold = automask_threshold

        self.data_imgs = []

        temp_imgs = os.listdir(self.path_dir + "/imgs/")
        temp_imgs_sorted = strsort(temp_imgs)

        for i in range(len(temp_imgs_sorted)):
            self.data_imgs.append(self.path_dir + "/imgs/" + temp_imgs_sorted[i])
        print(self.data_imgs)

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

        # cal mask
        sv_masks = []
        for i in range(len(imgs_temp) - 1):
            im1 = imgs_temp[i]
            im2 = imgs_temp[i + 1]

            sv_mask = np.abs(im1 - im2)
            sv_mask = np.where(sv_mask < self.automask_threshold, sv_mask / self.automask_threshold, 1)

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