import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss


eps = 1e-6

# ============================ #
#  Self-reconstruction loss    #
# ============================ #



@regist_loss
class self_L1():
    def TV_loss(self, x):
        """
        计算批量图像的Total Variation Loss。
    
        参数:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)
    
        返回:
            torch.Tensor: TV损失值
        """
        # 计算水平方向的差异
        h_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
        # 计算垂直方向的差异
        v_diff = x[:, :, :-1, :] - x[:, :, 1:, :]
    
        # 计算TV损失（使用L2范数，即平方和）
        tv_loss = torch.sum(h_diff ** 2) + torch.sum(v_diff ** 2)
    
        # 可选：使用L1范数（绝对值和）
        # tv_loss = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(v_diff))
    
        return tv_loss

    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        # print("use this loss function")
        return F.l1_loss(output, target_noisy)  # + 0.05 * self.TV_loss(output)

@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output, target_noisy)
