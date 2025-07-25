# Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

# This file is part of the implementation as described in the CVPR 2017 paper:
# Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
# Please see the file LICENSE.txt for the license governing this code.

import numpy as np
import scipy.io as sio
import os
import h5py
import pandas as pd
import base64
import argparse
import torch
import torch.nn as nn
from model.get_model import BSN

from torch.utils.data import DataLoader

from util.generator import np2tensor, tensor2np
from util.config_parse import ConfigParser
from util.file_manager import FileManager
from util.logger import Logger
from util.model_need_tools import set_denoiser, test_dataloader_process, crop_test, self_ensemble, set_status

from DataDeal.dnd_submission.bundle_submissions import bundle_submissions_srgb
from DataDeal.dnd_submission.dnd_denoise import denoise_srgb
from DataDeal.dnd_submission.pytorch_wrapper import pytorch_denoiser
from DataDeal.Data_loader import SIDD, SIDD_benchmark, SIDD_val, DND, preped_RN_data

args = argparse.ArgumentParser()
args.add_argument('-c', '--config', default='config/SIDD', type=str)
args.add_argument('-e', '--ckpt_epoch', default=0, type=int)
args.add_argument('-g', '--gpu', default='4', type=str)
args.add_argument('--save_folder', default='output/DND_test_out', type=str)
args.add_argument('--pretrained', default='/home/lab/wyj/mmbsn/output/120_single_no_attn_finetune/checkpoint/SIDD_MMBSN_028.pth',
                  type=str)
args.add_argument('--thread', default=4, type=int)
args.add_argument('--self_en', action='store_true')
args.add_argument('--test_dir', default='/data/wyj/dataset/DND/', type=str)
args.add_argument('-rd', '--data_root_dir',
                  default='/data/wyj/dataset', type=str)

args = args.parse_args()

assert args.config is not None, 'config file path is needed'

cfg = ConfigParser(args)

# device setting
if cfg['gpu'] == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

test_cfg = cfg['test']
ckpt_cfg = cfg['checkpoint']
status_len = 13
output_folder = cfg['save_folder']
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

print(cfg['pretrained'])
# logger = Logger()
# logger.highlight(logger.get_start_msg())
status = set_status('test')
denoiser = set_denoiser(checkpoint_path=cfg['pretrained'], cfg=cfg)
# status = set_status('test%03d'%cfg['ckpt_epoch'])
if cfg['self_en']:
    denoiser = lambda *input_data: self_ensemble(denoiser, *input_data)
elif 'crop' in cfg['test']:
    denoiser = lambda *input_data: crop_test(denoiser, *input_data, size=cfg['test']['crop'])


def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf


def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx, yy, bb]
    return sigma


def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0, bb]
    return sigma


def denoise_raw(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all raw images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the parameters of the noise level
                  function (nlf["a"], nlf["b"]) and a mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:
        pass

    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_raw', '%04d.mat' % (i + 1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['Inoisy']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3]].copy()
            Idenoised_crop = Inoisy_crop.copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)
            for yy in range(2):
                for xx in range(2):
                    nlf["sigma"] = load_sigma_raw(info, i, k, yy, xx)
                    Inoisy_crop_c = Inoisy_crop[yy:H:2, xx:W:2].copy()
                    Idenoised_crop_c = denoiser(Inoisy_crop_c, nlf)
                    Idenoised_crop[yy:H:2, xx:W:2] = Idenoised_crop_c
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder, '%04d_%02d.mat' % (i + 1, k + 1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k + 1, 20))
        print('[%d/%d] %s done\n' % (i + 1, 50, filename))


# def denoise_srgb(denoiser, data_folder, out_folder):
#     '''
#     Utility function for denoising all bounding boxes in all sRGB images of
#     the DND dataset.
#
#     denoiser      Function handle
#                   It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch
#                   and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
#     data_folder   Folder where the DND dataset resides
#     out_folder    Folder where denoised output should be written to
#     '''
#     try:
#         os.makedirs(out_folder)
#     except:
#         pass
#
#     print('model loaded\n')
#     # load info
#     infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
#     info = infos['info']
#     bb = info['boundingboxes']
#     print('info loaded\n')
#     # process data
#     for i in range(50):
#         filename = os.path.join(data_folder, 'images_srgb', '%04d.mat' % (i + 1))
#         img = h5py.File(filename, 'r')
#         Inoisy = np.float32(np.array(img['InoisySRGB']).T)
#         # bounding box
#         ref = bb[0][i]
#         boxes = np.array(info[ref]).T
#         for k in range(20):
#             idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
#             Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
#             H = Inoisy_crop.shape[0]
#             W = Inoisy_crop.shape[1]
#             nlf = load_nlf(info, i)
#             for yy in range(2):
#                 for xx in range(2):
#                     nlf["sigma"] = load_sigma_srgb(info, i, k)
#                     # Idenoised_crop = denoiser(Inoisy_crop, nlf)
#                     Idenoised_crop = denoiser(Inoisy_crop)
#             # save denoised data
#             Idenoised_crop = np.float32(Idenoised_crop)
#             save_file = os.path.join(out_folder, '%04d_%02d.mat' % (i + 1, k + 1))
#             sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
#             print('%s crop %d/%d' % (filename, k + 1, 20))
#         print('[%d/%d] %s done\n' % (i + 1, 50, filename))


def denoise_srgb(denoiser, data_folder, out_folder):
    """
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy) where Inoisy is a noisy image patch
                  in tensor format with shape (b, c, h, w)
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    """
    try:
        os.makedirs(out_folder)
    except Exception:
        pass

    # 确保输入与模型参数位于同一设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')

    # 处理每张图像（共50张）
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat' % (i + 1))
        img = h5py.File(filename, 'r')
        # 读取噪声图像，注意 T 转置后得到 (H, W, 3)
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        Inoisy = np.ascontiguousarray(Inoisy)
        # 转换为张量，得到 (1, 3, H, W)，并移动到与模型相同的设备上
        Inoisy_tensor = torch.tensor(Inoisy).permute(2, 0, 1).unsqueeze(0).to(device)

        # 获取当前图像的 bounding box 信息
        ref = bb[0][i]
        boxes = np.array(info[ref]).T

        for k in range(20):
            # idx 顺序为 [y_start, y_end, x_start, x_end]
            idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
            # 裁剪时对 height（维度2）和 width（维度3）进行索引，保留所有通道
            Inoisy_crop = Inoisy_tensor[:, :, idx[0]:idx[1], idx[2]:idx[3]].clone()

            H, W = Inoisy_crop.shape[2], Inoisy_crop.shape[3]
            nlf = load_nlf(info, i)

            # 原代码中两层循环只用于加载 nlf["sigma"]，此处仅调用一次 denoiser 即可
            nlf["sigma"] = load_sigma_srgb(info, i, k)
            Idenoised_crop = denoiser(Inoisy_crop)

            # 将 denoiser 输出从 GPU 转回 CPU，并转换为 numpy 数组（格式为 H x W x 3）
            Idenoised_crop_np = Idenoised_crop.squeeze().permute(1, 2, 0).cpu().numpy()
            save_file = os.path.join(out_folder, '%04d_%02d.mat' % (i + 1, k + 1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop_np})
            print('%s crop %d/%d' % (filename, k + 1, 20))

        print('[%d/%d] %s done\n' % (i + 1, 50, filename))



if __name__ == '__main__':
    denoise_srgb(denoiser=denoiser, data_folder='/data/wyj/dataset/DND', out_folder='./DND/test')