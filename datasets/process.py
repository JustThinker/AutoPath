import os
import sys
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

sys.path.append(os.path.split(sys.path[0])[0])


parser = argparse.ArgumentParser(description='pre-process LiTS data')
parser.add_argument('--data_root', required=True, type=str, help='root path of LiTS data')
parser.add_argument('--dataset_root', type=str, default='../data/LiTS', help='root path of saving processed data')
parser.add_argument('--size', type=int, default=48, help='minimum slices')
parser.add_argument('--down_scale', type=float, default=0.5, help='factor of downsample')
parser.add_argument('--expand_slice', type=int, default=20, help='expand slices')
parser.add_argument('--thickness', default=1, help='thickness of data')
parser.add_argument('--upper', default=250, help='Window width upper bound')
parser.add_argument('--lower', default=-200, help='Window width lower bound')
args = parser.parse_args()


def process(data_set_path, ct_path, seg_path):
    if os.path.exists(data_set_path):
        shutil.rmtree(data_set_path)

    new_ct_path = os.path.join(data_set_path, 'ct')
    new_seg_dir = os.path.join(data_set_path, 'seg')

    os.mkdir(data_set_path)
    os.mkdir(new_ct_path)
    os.mkdir(new_seg_dir)

    for file in tqdm(os.listdir(ct_path)):

        ct = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        # fuse segmentation of liver and tumor
        seg_array[seg_array > 0] = 1

        ct_array[ct_array > args.upper] = args.upper
        ct_array[ct_array < args.lower] = args.lower

        # resample dcm
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / args.thickness, args.down_scale, args.down_scale), order=3)
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / args.thickness, 1, 1), order=0)

        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        start_slice = max(0, start_slice - args.expand_slice)
        end_slice = min(seg_array.shape[0] - 1, end_slice + args.expand_slice)

        # these cases will be skip, if their amounts of total slices are less than minimum slices
        if end_slice - start_slice + 1 < args.size:
            print('!!!!!!!!!!!!!!!!')
            print(file, 'have too little slice', ct_array.shape[0])
            print('!!!!!!!!!!!!!!!!')
            continue

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / args.down_scale), ct.GetSpacing()[1] * int(1 / args.down_scale), args.thickness))

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], args.thickness))
    
        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))
        

if __name__ == '__main__':
    train_path = os.path.join(args.dataset_root, 'train')
    data_train_ct_path = os.path.join(args.data_root, 'ct', 'train')
    data_train_seg_path = os.path.join(args.data_root, 'seg', 'train')

    process(train_path, data_train_ct_path, data_train_seg_path)

    test_path = os.path.join(args.dataset_root, 'test')
    data_test_ct_path = os.path.join(args.data_root, 'ct', 'test')
    data_test_seg_path = os.path.join(args.data_root, 'seg', 'test')

    process(test_path, data_test_ct_path, data_test_seg_path)
