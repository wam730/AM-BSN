# import wget
import os
import scipy.io
import numpy as np
import pandas as pd
import base64
import argparse
import torch
import torch.nn as nn
from model.get_model import BSN

import torch
import torch.autograd as autograd
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
args.add_argument('-g', '--gpu', default='0', type=str)
args.add_argument('-sf','--save_folder', default='output/SIDD_test_out', type=str)
args.add_argument('-p','--pretrained', default='/home/lab/wyj/mmbsn/output/120_single_no_attn_finetune/checkpoint/SIDD_MMBSN_028.pth', type=str)
args.add_argument('--thread', default=8, type=int)
args.add_argument('--self_en', action='store_true')
args.add_argument('--test_dir', default='./dataset/test_data', type=str)
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


# def my_srgb_denoiser(x):
#     """
#     使用你的 PyTorch 去噪模型对 sRGB 图像块进行去噪。
#     输入: x 是一个 NumPy 数组，形状为 (H, W, C)（高度，宽度，通道数）。
#     输出: 返回去噪后的 NumPy 数组，形状与输入相同。
#     """
#     # 转换输入数据为 PyTorch 张量
#     x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().cuda()  # 转换为 (1, C, H, W)
#
#     # 使用模型进行去噪
#     with torch.no_grad():
#         y_tensor = denoiser(x_tensor)
#
#     # 转换输出张量为 NumPy 数组
#     y = y_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 转回 (H, W, C)
#
#     # 确保输出的 dtype 与输入一致
#     y = y.astype(x.dtype)
#
#     return y


def my_srgb_denoiser(x):
    """
    使用你的 PyTorch 去噪模型对 sRGB 图像块进行去噪。
    输入: x 是一个 NumPy 数组，形状为 (H, W, C)（高度，宽度，通道数），类型为 uint8。
    输出: 返回去噪后的 NumPy 数组，形状与输入相同，类型与输入一致。
    """
    # 确保输入为 uint8
    if x.dtype == np.uint8:
        # 转换为 float32，但保留原始值域 [0, 255]
        x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().cuda()
    else:
        raise ValueError("Input must be of type uint8.")

    # 使用模型进行去噪
    with torch.no_grad():
        y_tensor = denoiser(x_tensor)

    # 转换输出张量为 NumPy 数组
    y = y_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 确保输出的类型和输入一致（转换回 uint8）
    y = y.clip(0, 255).astype(np.uint8)

    return y


def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    decoded_array = decoded_array.reshape(array_shape)
    return decoded_array


# Download input file, if needed.
# url = 'https://competitions.codalab.org/my/datasets/download/0d8a1e68-155d-4301-a8cd-9b829030d719'
input_file = '/data/wyj/dataset/SIDD/BenchmarkNoisyBlocksSrgb.mat'
if os.path.exists(input_file):
    print(f'{input_file} exists. No need to download it.')
else:
    print('Downloading input file BenchmarkNoisyBlocksSrgb.mat...')
    wget.download(url, input_file)
    print('Downloaded successfully.')

# Read inputs.
key = 'BenchmarkNoisyBlocksSrgb'
inputs = scipy.io.loadmat(input_file)
inputs = inputs[key]
print(f'inputs.shape = {inputs.shape}')
print(f'inputs.dtype = {inputs.dtype}')

# Denoising.
output_blocks_base64string = []
for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j, :, :, :]
        # print(inputs[1, 1, :, :, :])
        print('{}-{}:{}'.format(i, j, i * 32 + j + 1))
        out_block = my_srgb_denoiser(in_block)
        assert in_block.shape == out_block.shape
        assert in_block.dtype == out_block.dtype
        out_block_base64string = array_to_base64string(out_block)
        output_blocks_base64string.append(out_block_base64string)

# Save outputs to .csv file.
output_file = './output/SubmitSrgb_benchmark_sasl.csv'
print(f'Saving outputs to {output_file}')
output_df = pd.DataFrame()
n_blocks = len(output_blocks_base64string)
print(f'Number of blocks = {n_blocks}')
output_df['ID'] = np.arange(n_blocks)
output_df['BLOCK'] = output_blocks_base64string

output_df.to_csv(output_file, index=False)

# TODO: Submit the output file SubmitSrgb.csv at
# kaggle.com/competitions/sidd-benchmark-srgb-psnr
print('TODO: Submit the output file SubmitSrgb.csv at', output_file)
print('kaggle.com/competitions/sidd-benchmark-srgb-psnr')

print('Done.')
