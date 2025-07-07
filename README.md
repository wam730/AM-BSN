# AM-BSN
Official PyTorch implementation of "Enhancing Self-Supervised Image Denoising with Asymmetric Mask Blind-Spot Networks."

## Acknowledgement
Part of our codes are adapted from [AP-BSN](https://github.com/wooseoklee4/AP-BSN) and [MMBSN](https://github.com/dannie125/MM-BSN)and we are expressing gratitude for their work sharing.

## How to test
To test noisy images with pre-trained AM-BSN in gpu:0
'''
python test.py -c SIDD -g 0 --pretrained ./ckpt/AMBSN.pth --td [your noisy images dir]
'''
