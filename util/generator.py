from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping & chnage type to uint8
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)

    return peak_signal_noise_ratio(img1, img2, data_range=255)


def ssim(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping
    img2 = np.clip(img2, 0, 255)
    img1 = np.clip(img1, 0, 255)

    return structural_similarity(img1, img2, multichannel=True, data_range=255, channel_axis=2)


def np2tensor(n: np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2, 0, 1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2, 0, 1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s' % (n.shape,))


def tensor2np(t: torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    t = t.cpu().detach()

    # gray
    if len(t.shape) == 2:
        return t.permute(1, 2, 0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1, 2, 0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0, 2, 3, 1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s' % (t.shape,))


def imwrite_tensor(t, name='test.png'):
    cv2.imwrite('./%s' % name, tensor2np(t.cpu()))


def imread_tensor(name='test'):
    return np2tensor(cv2.imread('./%s' % name))


def rot_hflip_img(img: torch.Tensor, rot_times: int = 0, hflip: int = 0):
    '''
    rotate '90 x times degree' & horizontal flip image
    (shape of img: b,c,h,w or c,h,w)
    '''
    b = 0 if len(img.shape) == 3 else 1
    # no flip
    if hflip % 2 == 0:
        # 0 degrees
        if rot_times % 4 == 0:
            return img
        # 90 degrees
        elif rot_times % 4 == 1:
            return img.flip(b + 1).transpose(b + 1, b + 2)
        # 180 degrees
        elif rot_times % 4 == 2:
            return img.flip(b + 2).flip(b + 1)
        # 270 degrees
        else:
            return img.flip(b + 2).transpose(b + 1, b + 2)
    # horizontal flip
    else:
        # 0 degrees
        if rot_times % 4 == 0:
            return img.flip(b + 2)
        # 90 degrees
        elif rot_times % 4 == 1:
            return img.flip(b + 1).flip(b + 2).transpose(b + 1, b + 2)
        # 180 degrees
        elif rot_times % 4 == 2:
            return img.flip(b + 1)
        # 270 degrees
        else:
            return img.transpose(b + 1, b + 2)


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c,
                                                                                                           w + 2 * f * pad,
                                                                                                           h + 2 * f * pad)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(b, c,
                                                                                                                 w + 2 * f * pad,
                                                                                                                 h + 2 * f * pad)


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f,
                                                                                              h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)

def random_pd(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    """
    pixel-shuffle down-sampling (PD) with random shuffling of sub-images while preserving RGB channels
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    """
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0:
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c,
                                                                                                           w + 2 * f * pad,
                                                                                                           h + 2 * f * pad)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)

        # Reshape to separate RGB channels and sub-images
        # 将张量重塑为 [b, 3, f*f, h//f, w//f] 的形状
        # 其中3是RGB通道数，f*f是子图像数量
        unshuffled = unshuffled.view(b, 3, f * f, h // f, w // f)

        # Generate random indices for shuffling sub-images
        shuffle_idx = torch.randperm(f * f)

        # 只打乱子图像的顺序，保持RGB通道不变
        # unshuffled = unshuffled.index_select(2, shuffle_idx)
        unshuffled = unshuffled[:, :, shuffle_idx, :, :]

        # Reshape back to original format
        unshuffled = unshuffled.view(b, c * f * f, h // f, w // f)

        if pad != 0:
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return shuffle_idx, unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3,
                                                                                                    5).reshape(b, c,
                                                                                                               w + 2 * f * pad,
                                                                                                               h + 2 * f * pad)


def inv_random_pd(x: torch.Tensor, shuffle_idx: torch.Tensor, f: int, pad: int = 0):
    """
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        shuffle_idx (Tensor) : indices used for shuffling in random_pd
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    """
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b, c, w, h = x.shape

        # Calculate inverse shuffle indices
        inverse_idx = torch.argsort(shuffle_idx)

        # Reshape to match the format in random_pd
        # before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f,
        #                                                                                       h // f)

        # Reshape to separate RGB channels and sub-images
        # before_shuffle = before_shuffle.view(b, 3, f * f, w // f, h // f)

        before_shuffle = (
            x.view(b, c, f, w // f, f, h // f)  # f x f
            .permute(0, 1, 2, 4, 3, 5)  # (b, c, f, f, w//f, h//f)
            .reshape(b, c * f * f, w // f, h // f)  # (b, c * f * f, w // f, h // f)
            .view(b, 3, f * f, w // f, h // f)  # 调整通道为 RGB 格式
        )

        # Apply inverse shuffling to restore original sub-image order
        before_shuffle = before_shuffle[:, :, inverse_idx, :, :]
        # before_shuffle = before_shuffle.index_select(2, inverse_idx)

        # Reshape back to original format
        before_shuffle = before_shuffle.view(b, c * f * f, w // f, h // f)

        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


# def random_pd(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
#     '''
#     pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
#     Args:
#         x (Tensor) : input tensor
#         f (int) : factor of PD
#         pad (int) : number of pad between each down-sampled images
#         pad_value (float) : padding value
#     Return:
#         pd_x (Tensor) : down-shuffled image tensor with pad or not
#     '''
#     # single image tensor
#     if len(x.shape) == 3:
#         c, w, h = x.shape
#         unshuffled = F.pixel_unshuffle(x, f)
#         if pad != 0:
#             unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
#         return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c,
#                                                                                                            w + 2 * f * pad,
#                                                                                                            h + 2 * f * pad)
#     # batched image tensor
#     else:
#         b, c, w, h = x.shape
#         unshuffled = F.pixel_unshuffle(x, f)
#
#         b, c_f2, h_f, w_f = unshuffled.shape
#         unshuffled = unshuffled.view(b, c, f * f, w // f, h // f)
#         shuffle_idx = torch.randperm(f * f)
#         unshuffled = unshuffled[:, :, shuffle_idx, :, :]
#         unshuffled = unshuffled.view(b, c_f2, h_f, w_f)
#
#         if pad != 0:
#             unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
#         return shuffle_idx, unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3,
#                                                                                                     5).reshape(b, c,
#                                                                                                                w + 2 * f * pad,
#                                                                                                                h + 2 * f * pad)
#
#
# def inv_random_pd(x: torch.Tensor, shuffle_idx: torch.Tensor, f: int, pad: int = 0):
#     '''
#     inverse of pixel-shuffle down-sampling (PD)
#     see more details about PD in pixel_shuffle_down_sampling()
#     Args:
#         x (Tensor) : input tensor
#         f (int) : factor of PD
#         pad (int) : number of pad will be removed
#         param shuffle_idx:
#     '''
#     # single image tensor
#     if len(x.shape) == 3:
#         c, w, h = x.shape
#         before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
#         if pad != 0:
#             before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
#         return F.pixel_shuffle(before_shuffle, f)
#     # batched image tensor
#     else:
#         b, c, w, h = x.shape
#
#         inverse_idx = torch.argsort(shuffle_idx)
#
#         before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f,
#                                                                                               h // f)
#         before_shuffle = before_shuffle[:, inverse_idx, :, :]
#
#         if pad != 0:
#             before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
#         return F.pixel_shuffle(before_shuffle, f)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def get_gaussian_2d_filter(window_size, sigma, channel=1, device=torch.device('cpu')):
    '''
    return 2d gaussian filter window as tensor form
    Arg:
        window_size : filter window size
        sigma : standard deviation
    '''
    gauss = torch.ones(window_size, device=device)
    for x in range(window_size): gauss[x] = exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
    gauss = gauss.unsqueeze(1)
    #gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=device).unsqueeze(1)
    filter2d = gauss.mm(gauss.t()).float()
    filter2d = (filter2d / filter2d.sum()).unsqueeze(0).unsqueeze(0)
    return filter2d.expand(channel, 1, window_size, window_size)


def get_mean_2d_filter(window_size, channel=1, device=torch.device('cpu')):
    '''
    return 2d mean filter as tensor form
    Args:
        window_size : filter window size
    '''
    window = torch.ones((window_size, window_size), device=device)
    window = (window / window.sum()).unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size)


def mean_conv2d(x, window_size=None, window=None, filter_type='gau', sigma=None, keep_sigma=False, padd=True):
    '''
    color channel-wise 2d mean or gaussian convolution
    Args:
        x : input image
        window_size : filter window size
        filter_type(opt) : 'gau' or 'mean'
        sigma : standard deviation of gaussian filter
    '''
    b_x = x.unsqueeze(0) if len(x.shape) == 3 else x

    if window is None:
        if sigma is None: sigma = (window_size - 1) / 6
        if filter_type == 'gau':
            window = get_gaussian_2d_filter(window_size, sigma=sigma, channel=b_x.shape[1], device=x.device)
        else:
            window = get_mean_2d_filter(window_size, channel=b_x.shape[1], device=x.device)
    else:
        window_size = window.shape[-1]

    if padd:
        pl = (window_size - 1) // 2
        b_x = F.pad(b_x, (pl, pl, pl, pl), 'reflect')

    m_b_x = F.conv2d(b_x, window, groups=b_x.shape[1])

    if keep_sigma:
        m_b_x /= (window ** 2).sum().sqrt()

    if len(x.shape) == 4:
        return m_b_x
    elif len(x.shape) == 3:
        return m_b_x.squeeze(0)
    else:
        raise ValueError('input image shape is not correct')


def randomArrangement(x: torch.Tensor, pd_factor: int):
    """
    xiaoyu 2023/5/23\n
    用于将pd降采样结果进一步打乱\n
    :param x: pd降采样结果， 大小为[b,c,h,w]
    :param pd_factor: 表示每行有多少个小图像块
    :return: [1]打乱的结果， [2]用于还原打乱结果的映射
    """
    #np.random.permutation是一个随机排列函数,就是将输入的数据进行随机排列
    seq2random = np.random.permutation(range(0, pd_factor * pd_factor))  # 置乱数组 seq2random[i]表示x第i个块应该放在random_x的哪一块

    random2seq = np.zeros_like(seq2random)  # random2seq[i]表示random_x的第i块应该被放在x的哪一块
    for i in range(0, len(seq2random)):  #恢复数组
        random2seq[seq2random[i]] = i

    random_x = torch.zeros_like(x)

    b, c, h, w = x.shape
    assert h % pd_factor == 0, "dim[-2] of input x %d cannot be divided by %d" % (h, pd_factor)  #确保能被整除
    assert w % pd_factor == 0, "dim[-1] of input x %d cannot be divided by %d" % (w, pd_factor)
    sub_h = h // pd_factor  #h//有多少小块 =子块的高
    sub_w = w // pd_factor
    idx = 0

    for i in range(0, pd_factor):
        for j in range(0, pd_factor):
            random_idx = seq2random[idx]
            random_j = random_idx % pd_factor
            random_i = random_idx // pd_factor
            random_x[:, :, random_i * sub_h:(random_i + 1) * sub_h, random_j * sub_w:(random_j + 1) * sub_w] = x[:, :,
                                                                                                               i * sub_h:(
                                                                                                                                 i + 1) * sub_h,
                                                                                                               j * sub_w:(
                                                                                                                                 j + 1) * sub_w]  #没看懂
            idx += 1

    return random_x, random2seq


def inverseRandomArrangement(random_x: torch.Tensor, random2seq: np.ndarray, pd_factor: int):
    """
    xiaoyu 2023/5/23 \n
    用于还原被randomArrangement(...)打乱的张量 \n
    :param random_x: 乱序张量
    :param random2seq: 逆映射
    :param pd_factor: 表示每行有多少个小图像块
    :return: [1]顺序正确的张量
    """

    x = torch.zeros_like(random_x)

    b, c, h, w = random_x.shape
    assert h % pd_factor == 0, "dim[-2] of input x %d cannot be divided by %d" % (h, pd_factor)
    assert w % pd_factor == 0, "dim[-1] of input x %d cannot be divided by %d" % (w, pd_factor)
    sub_h = h // pd_factor
    sub_w = w // pd_factor

    random_idx = 0

    for random_i in range(0, pd_factor):
        for random_j in range(0, pd_factor):
            idx = random2seq[random_idx]
            j = idx % pd_factor
            i = idx // pd_factor
            x[:, :, i * sub_h:(i + 1) * sub_h, j * sub_w:(j + 1) * sub_w] = random_x[:, :,
                                                                            random_i * sub_h:(random_i + 1) * sub_h,
                                                                            random_j * sub_w:(random_j + 1) * sub_w]
            random_idx += 1

    return x
