import torch.nn as nn
import torch

class DiamondMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)

        # Mask the center point and the diamond pattern around it
        center_y, center_x = kH // 2, kW // 2
        self.mask[:, :, center_y, center_x] = 0  # Center point

        # Mask points in the diamond pattern
        if kH >= 5 and kW >= 5:
            # Up and down two points from the center
            if center_y - 1 >= 0:
                self.mask[:, :, center_y - 1, center_x] = 0
            if center_y - 2 >= 0:
                self.mask[:, :, center_y - 2, center_x] = 0
            if center_y + 1 < kH:
                self.mask[:, :, center_y + 1, center_x] = 0
            if center_y + 2 < kH:
                self.mask[:, :, center_y + 2, center_x] = 0

            # Left and right two points from the center
            if center_x - 1 >= 0:
                self.mask[:, :, center_y, center_x - 1] = 0
            if center_x - 2 >= 0:
                self.mask[:, :, center_y, center_x - 2] = 0
            if center_x + 1 < kW:
                self.mask[:, :, center_y, center_x + 1] = 0
            if center_x + 2 < kW:
                self.mask[:, :, center_y, center_x + 2] = 0

            # Diagonal points
            if center_y - 1 >= 0 and center_x - 1 >= 0:
                self.mask[:, :, center_y - 1, center_x - 1] = 0  # Top-left
            if center_y - 1 >= 0 and center_x + 1 < kW:
                self.mask[:, :, center_y - 1, center_x + 1] = 0  # Top-right
            if center_y + 1 < kH and center_x - 1 >= 0:
                self.mask[:, :, center_y + 1, center_x - 1] = 0  # Bottom-left
            if center_y + 1 < kH and center_x + 1 < kW:
                self.mask[:, :, center_y + 1, center_x + 1] = 0  # Bottom-right

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class CrossMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)

        # Mask the center point and its neighboring points (up, down, left, right)
        center_y, center_x = kH // 2, kW // 2
        self.mask[:, :, center_y, center_x] = 0  # Center point
        if center_y - 1 >= 0:
            self.mask[:, :, center_y - 1, center_x] = 0  # Up
        if center_y + 1 < kH:
            self.mask[:, :, center_y + 1, center_x] = 0  # Down
        if center_x - 1 >= 0:
            self.mask[:, :, center_y, center_x - 1] = 0  # Left
        if center_x + 1 < kW:
            self.mask[:, :, center_y, center_x + 1] = 0  # Right

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class XMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)

        # Mask for the 135-degree diagonal (top-left to bottom-right)
        for i in range(kH):
            self.mask[:, :, i, i] = 0

        # Mask for the 45-degree diagonal (top-right to bottom-left)
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0
        # if kH == 5:
        #     self.mask[:, :, 1:-1, 1:-1] = 0
        # else:
        #     pass

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ColMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class RowMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fSzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, kW // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class SzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle135MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle45MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class chaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
            self.mask[:, :, kW - 1 - i, i] = 1
            self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fchaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
            self.mask[:, :, kW - 1 - i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class huiMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 1:-1, 1:-1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# class CustomMaskedConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # Ensure the kernel size is 5x5 as required
#         assert self.kernel_size == (5, 5), "This mask is designed for 5x5 kernels only."
#
#         # Initialize the mask with ones, matching the weight tensor's shape
#         self.register_buffer('mask', torch.ones_like(self.weight))
#
#         # Get kernel dimensions
#         kH, kW = self.kernel_size
#
#         # Create index grids for rows and columns
#         i = torch.arange(kH)
#         j = torch.arange(kW)
#         ii, jj = torch.meshgrid(i, j, indexing='ij')
#
#         # Define the masking condition:
#         # - Even rows (i % 2 == 0) mask even columns (j % 2 == 0)
#         # - Odd rows (i % 2 == 1) mask odd columns (j % 2 == 1)
#         condition = ((ii % 2 == 0) & (jj % 2 == 0)) | ((ii % 2 == 1) & (jj % 2 == 1))
#
#         # Create the 5x5 mask pattern
#         mask_pattern = torch.ones(kH, kW)
#         mask_pattern[condition] = 0  # Set masked positions to 0
#
#         # Apply the mask pattern to all output and input channels
#         self.mask[:, :, :, :] = mask_pattern
#
#     def forward(self, x):
#         # Apply the mask to the weights during the forward pass
#         self.weight.data *= self.mask
#         return super().forward(x)


# class CustomMaskedConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.register_buffer('mask', self.weight.data.clone())  # 初始化掩码
#         _, _, kH, kW = self.weight.size()
#
#         # 创建一个掩码并初始化为1
#         self.mask.fill_(1)
#
#         # 定义需要掩码的位置
#         mask_positions = [
#             (0, 0), (0, 2), (0, 4),
#             (1, 1), (1, 3),
#             (2, 0), (2, 2), (2, 4),
#             (3, 1), (3, 3),
#             (4, 0), (4, 2), (4, 4)
#         ]
#
#         # 将这些位置的掩码设置为0
#         for i, j in mask_positions:
#             self.mask[:, :, i, j] = 0
#
#     def forward(self, x):
#         self.weight.data *= self.mask  # 应用掩码
#         return super().forward(x)

class CustomMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()

        # Initialize mask with ones
        self.mask.fill_(1)

        # Define the positions that need to be masked (set to zero)
        mask_positions = [
            (0, 2),
            (1, 1), (1, 3),
            (2, 0), (2, 2), (2, 4),
            (3, 1), (3, 3),
            (4, 2)
        ]

        # Set the specified positions to zero in the mask
        for i, j in mask_positions:
            self.mask[:, :, i, j] = 0

    def forward(self, x):
        # Apply the mask to the weights
        self.weight.data *= self.mask
        return super().forward(x)