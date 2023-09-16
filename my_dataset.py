import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
import torch


def readNii(path):
    img = sitk.ReadImage(path)
    # 查看图片深度和尺寸
    data = sitk.GetArrayFromImage(img)
    return np.array(data)


class SSOCTADataset(Dataset):
    def __init__(self, root: str, train: bool, test: bool, transforms=None):
        super(SSOCTADataset, self).__init__()

        data_root = os.path.join(root)
        assert os.path.exists(data_root), f'path {data_root} does not exist.'
        quality_train = pd.read_csv('data/train/most_1_type_and_excellent_quality_train.txt', header=None)
        quality_train = np.array(quality_train).reshape(-1)

        quality_test = pd.read_csv('data/test/most_1_type_and_excellent_quality_test.txt', header=None)
        quality_test = np.array(quality_test).reshape(-1)

        self.transforms = transforms
        # 测试情况下的数据集
        if test:
            # self.images = readNii(os.path.join(data_root, 'test', 'most_1_type_image_test.nii.gz'))
            # self.masks = readNii(os.path.join(data_root, 'test', 'most_1_type_mask_test.nii.gz'))
            # self.label = quality_test
            self.images = readNii('data/test/image.nii.gz')
            self.masks = readNii('data/test/mask.nii.gz')
            self.label = quality_test
        # 训练情况下的数据集
        elif train:
            # self.images = readNii(os.path.join(data_root, 'train', 'most_1_type_image_train.nii.gz'))
            # self.masks = readNii(os.path.join(data_root, 'train', 'most_1_type_mask_train.nii.gz'))
            # self.label = quality_train
            self.images = readNii('data/train/image.nii.gz')
            self.masks = readNii('data/train/mask.nii.gz')
            self.label = quality_train

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx, :, :])
        mask = Image.fromarray(self.masks[idx, :, :])
        category = torch.tensor(self.label[idx])

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return img, mask, category

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_images = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_images, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
