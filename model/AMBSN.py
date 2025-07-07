import torch
import torch.nn as nn
import numbers
import torch.nn.functional as F
import numpy as np
import math
from pdb import set_trace as stx
from einops import rearrange
from .masks import CentralMaskedConv2d, XMaskedConv2d, CustomMaskedConv2d, angle45MaskedConv2d, angle135MaskedConv2d
from .restormer_arch import DTB

class AMBSN(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, DCL1_num=2, DCL2_num=7, mask_type='o_x'):
        """
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        """
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        ly0 = []
        ly0 += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly0 += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly0)  # 128

        DCL_number1 = 2  # DCL1_num  # the number of DCL in DC_branchl
        DCL_number2 = 7  # DCL2_num  # the number of DCL in DC_branchl2

        self.branch_local = DC_branchl(2, base_ch, 'central', DCL_number1)
        self.branch_global = DC_branchl(3, base_ch, 'x', DCL_number1)

        ly_c = []
        ly_c += [nn.Conv2d(base_ch, base_ch, kernel_size=1)]  # 256
        ly_c += [nn.ReLU(inplace=True)]
        self.conv2_1 = nn.Sequential(*ly_c)  # 128
        self.conv2_2 = nn.Sequential(*ly_c)  # 128

        self.dc_branchl2_local = DC_branchl2(2, base_ch, DCL_number2)
        self.dc_branchl2_global = DC_branchl2(3, base_ch, DCL_number2)

        ly = []
        ly += [nn.Conv2d(base_ch * 4, base_ch, kernel_size=1)]  # 128*4
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)
        FM = []

        local_FM, local_cat = self.branch_local(x)
        global_FM, global_cat = self.branch_global(x)
        FM.append(local_FM)
        FM.append(global_FM)

        conv2_1 = self.conv2_1(local_cat)  # 128
        dc_branchl2_local = self.dc_branchl2_local(conv2_1)  # 128

        conv2_2 = self.conv2_2(global_cat)  # 128
        dc_branchl2_global = self.dc_branchl2_global(conv2_2)  # 128

        FM.append(dc_branchl2_global)
        FM.append(dc_branchl2_local)

        cat3 = torch.cat(FM, dim=1)  # 128+128+128+128 = 4*128

        return self.tail(cat3)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, mask_type, num_module):
        super().__init__()

        ly0 = []
        ly1_1 = []
        ly1_2 = []
        ly2 = []

        if mask_type == 'x':
            ly0 += [XMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'c':
            ly0 += [CustomMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a45':
            ly0 += [angle45MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a135':
            ly0 += [angle135MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        else:
            ly0 += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        ly0 += [nn.ReLU(inplace=True)]
        ly0 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly0 += [nn.ReLU(inplace=True)]
        ly0 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly0 += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly0)

        ly1_1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly1_1 += [nn.ReLU(inplace=True)]
        self.conv1_1 = nn.Sequential(*ly1_1)
        self.conv1_2 = nn.Sequential(*ly1_1)

        ly2 += [DCl(stride, in_ch) for _ in range(num_module)]
        ly2 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly2 += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly2)

        ly1_2 += [nn.Conv2d(in_ch * 2, in_ch, kernel_size=1)]
        ly1_2 += [nn.ReLU(inplace=True)]
        self.conv1_3 = nn.Sequential(*ly1_2)

    def forward(self, x):

        y0 = self.head(x)

        conv1_1 = self.conv1_1(y0)
        y1 = self.body(conv1_1)
        cat0 = torch.cat([conv1_1, y1], dim=1)

        conv1_3 = self.conv1_3(cat0)

        conv1_2 = self.conv1_2(y0)

        return conv1_2, conv1_3


class DC_branchl2(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        ly = []
        ly += [DCl(stride, in_ch) for _ in range(num_module)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DC_branchl2_attn(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        ly = []
        ly += [DTB(dim=in_ch,num_blocks=[num_module], stride=stride, heads=[4])]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)
