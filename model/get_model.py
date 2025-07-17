from functools import lru_cache
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.cluster.hierarchy import single

from util.generator import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, randomArrangement, \
    inverseRandomArrangement, inv_random_pd, random_pd

from .APBSN import APBSN
from .CSCBSN import CSCBSN
from .AMBSN import AMBSN


class BSN(nn.Module):

    @classmethod
    @lru_cache(maxsize=1)
    def bsn_model(cls):
        return cls()

    def __init__(self, type='AMBSN', pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,
                 in_ch=3, bsn_base_ch=128, bsn_num_module=9, DCL1_num=2, DCL2_num=7, mask_type='o_fsz', shuffle=False, TTA=True):
        """
        Args:
            type           : BSN model type
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
            mask_type      : types of mask
            shuffle        : Whether to randomly shuffle the sub-images after PD
            TTA            : Whether to use the FEI strategy or not
        """
        super().__init__()

        # network hyperparameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        self.shuffle = shuffle
        self.all_time = 0
        self.n = 0
        self.TTA = TTA

        # define network
        if type == 'APBSN':
            self.bsn = APBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'CSCBSN':
            self.bsn = CSCBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'AMBSN':
            self.bsn = AMBSN(in_ch, in_ch, bsn_base_ch, DCL1_num, DCL2_num, mask_type)
        else:
            raise NotImplementedError('bsn type %s is not implemented' % type)

    def forward(self, img, pd=None, shuffle=None):
        """
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        """
        # default pd factor is training factor (a)
        if shuffle is None:
            shuffle = self.shuffle

        if pd is None:
            pd = self.pd_a

        # do PD
        if pd > 1:
            if shuffle:
                shuffle_idx, pd_img = random_pd(img, f=pd, pad=self.pd_pad)
                # pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
            else:
                pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
                shuffle_idx = None
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))
            shuffle_idx = None

        # forward blind-spot network
        pd_img_denoised = self.bsn(pd_img)

        # do inverse PD
        if pd > 1:
            if shuffle:
                # img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
                img_pd_bsn = inv_random_pd(pd_img_denoised, shuffle_idx, f=pd, pad=self.pd_pad)
            else:
                img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]

        return img_pd_bsn

    def denoise(self, x, TTA=None):
        """
        Denoising process for inference.
        """
        b, c, h, w = x.shape
        if TTA is None:
            TTA = self.TTA

        if TTA:
            # ============== PD = 2 ====================
            # start_time = time.time()
            img_pd2_bsn = x
            if h % self.pd_b != 0:
                img_pd2_bsn = F.pad(img_pd2_bsn, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
            if w % self.pd_b != 0:
                img_pd2_bsn = F.pad(img_pd2_bsn, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
            img_pd2_bsn = self.forward(img_pd2_bsn, pd=2)

            # ============== PD = 5 =====================
            img_pd5_bsn = x
            if h % self.pd_a != 0:
                img_pd5_bsn = F.pad(img_pd5_bsn, (0, 0, 0, self.pd_a - h % self.pd_a), mode='constant', value=0)
            if w % self.pd_a != 0:
                img_pd5_bsn = F.pad(img_pd5_bsn, (0, self.pd_a - w % self.pd_a, 0, 0), mode='constant', value=0)
            img_pd5_bsn = self.forward(img_pd5_bsn, pd=5)
            img_pd5_bsn = img_pd5_bsn[:, :, :h, :w]

            # ============== FUSE ====================
            # 加权融合
            img_pd_bsn = torch.add(torch.mul(img_pd5_bsn, 0.2), torch.mul(img_pd2_bsn, 0.8))

        else:
            if h % self.pd_b != 0:
                x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
            if w % self.pd_b != 0:
                x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
            img_pd_bsn = self.forward(img=x, pd=self.pd_b, shuffle=False)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            # img_pd_bsn = self.forward(img=img_pd_bsn, pd=self.pd_b)
            return img_pd_bsn[:, :, :h, :w]
        else:
            denoised = torch.empty(*x.shape, self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
            return torch.mean(denoised, dim=-1)
