"""
U-Net 3D 모델
=============

의료 영상 세그멘테이션을 위한 고전적인 U-Net 아키텍처입니다.
5개 세그멘테이션 CNN 중 첫 번째 모델 (Model 1/5)

U-Net 특징:
- 대칭적인 인코더-디코더 구조
- Skip connections으로 세밀한 특징 보존
- 의료 영상에서 검증된 성능
"""

import torch
import torch.nn as nn
from .base import BaseSegmentationModel


class DoubleConv3D(nn.Module):
    """(Conv3D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2,
                                diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(BaseSegmentationModel):
    """
    췌장 세그멘테이션을 위한 3D U-Net

    논문: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    3D 버전으로 수정하여 구현되었습니다.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        bilinear: bool = True,
        **kwargs
    ):
        """
        U-Net 3D 모델 초기화

        Args:
            in_channels (int): 입력 채널 수 (CT의 경우 1)
            num_classes (int): 출력 클래스 수
            base_channels (int): 기본 채널 수. 각 레벨에서 2배씩 증가 (기본값 32)
            bilinear (bool): True이면 Upsample 사용, False이면 ConvTranspose3d 사용
        """
        super().__init__(in_channels, num_classes, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        U-Net 순전파
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_config(self):
        """
        모델 설정 정보 반환
        """
        config = super().get_config()
        config.update({
            "base_channels": self.base_channels,
            "bilinear": self.bilinear,
        })
        return config
