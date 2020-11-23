import os
import sys
import random
import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset

sys.path.append(os.path.split(sys.path[0])[0])


# parameters
size = 48


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir, test=False):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(
            map(lambda x: x.replace('volume', 'segmentation').replace('.nii', '.nii.gz'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

        self.test = test

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # read dcm and segmentation
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # min-max normalization
        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        if self.test is False:
            start_slice = random.randint(0, ct_array.shape[0] - size)
        else:
            start_slice = 25
        end_slice = start_slice + size - 1

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):
        return len(self.ct_list)
