# AM-BSN
Official PyTorch implementation of "Enhancing Self-Supervised Image Denoising with Asymmetric Mask Blind-Spot Networks."

![图片描述](./images/example.png)

## Abstract
Image denoising is a fundamental task in image processing and computer vision. Traditional supervised methods heavily rely on large sets of noisy-clean image pairs, which are impractical to obtain for real-world noisy images. Self-supervised denoising methods offer a viable alternative but often struggle with spatial correlation in noise. In this paper, we introduce the Asymmetric Mask Blind-Spot Network (AM-BSN), designed to disrupt spatial correlations of large-scale noise in real-world images. Our network features a dual-branch architecture: a local branch employing a 3x3 central mask convolution for fine detail recovery, and a global branch utilizing a 5x5 'X'-shaped mask convolution and dilated convolutions for global structure reconstruction. Experimental results on real-world datasets demonstrate that AM-BSN outperforms state-of-the-art methods, achieving a PSNR of 37.90 dB and an SSIM of 0.885 on the SIDD benchmark. This research advances self-supervised denoising techniques, providing a practical solution for real-world applications.

## Parameters
|   __Models__   |                                    __SIDD Validation__                                   |__Parameters__ |
|:----------:|:-----------------------------------------------------------------------------------------------:|:-------:|
| AP-BSN |                                             35.91/0.870                                             |   3.7M   |
| MM-BSN |                                             37.38/0.882                                             |   5.3M   |
| AM-BSN |                                             37.54/0.884                                             |   3.8M   |

## Setup
### Requirements

Our experiments are done with:

- Python 3.10.13
- PyTorch 2.1.1+cu118
- numpy 1.26.3
- opencv 4.10.0.84
- scikit-image 0.24.0

### Directory
The data used in the training of our method is consistent with that of the APBSN. For the specific method of obtaining training data, please refer to [AP-BSN](https://github.com/wooseoklee4/AP-BSN).


## How to test
To test noisy images with pre-trained AM-BSN in gpu:0

Use [png2csv.py](https://github.com/wam730/AM-BSN/blob/main/png2csv.py) to convert the denoised images from the SIDD benchmark dataset into the CSV format required for the [Kaggle competition](https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr/leaderboard).
```
python test.py -c SIDD -g 0 --pretrained ./ckpt/AMBSN.pth --td [your noisy images dir]
```

## How to train
As soon as our paper is accepted, we will upload the training code immediately.

## Acknowledgement
Part of our codes are adapted from [AP-BSN](https://github.com/wooseoklee4/AP-BSN) and [MMBSN](https://github.com/dannie125/MM-BSN). We are expressing gratitude for their work sharing.
