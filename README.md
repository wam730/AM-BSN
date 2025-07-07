# AM-BSN
Official PyTorch implementation of "Enhancing Self-Supervised Image Denoising with Asymmetric Mask Blind-Spot Networks."

## Abstract


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

## How to test
To test noisy images with pre-trained AM-BSN in gpu:0
```
python test.py -c SIDD -g 0 --pretrained ./ckpt/AMBSN.pth --td [your noisy images dir]
```

## How to train
As soon as our paper is accepted, we will upload the training code immediately.

## Acknowledgement
Part of our codes are adapted from [AP-BSN](https://github.com/wooseoklee4/AP-BSN) and [MMBSN](https://github.com/dannie125/MM-BSN). We are expressing gratitude for their work sharing.
