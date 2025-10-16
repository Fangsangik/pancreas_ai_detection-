"""
C2FNAS 모델
===========

Coarse-to-Fine Neural Architecture Search 모델
5개 세그멘테이션 CNN 중 다섯 번째 모델 (Model 5/5)

기존 세그멘테이션 프로젝트 설정 기반
"""

import torch
import torch.nn as nn
from .base import BaseSegmentationModel


class C2FNAS(BaseSegmentationModel):
    """
    췌장 세그멘테이션을 위한 C2FNAS

    특징:
    - Coarse-to-fine 세그멘테이션 전략
    - NAS로 최적화된 아키텍처
    - 다중 스케일 특징 추출

    Note:
        MONAI 라이브러리의 C2FNAS를 사용하거나
        커스텀 구현 가능

    사용 예시:
    ---------
    >>> model = C2FNAS(in_channels=1, num_classes=3)
    >>> output = model(ct_scan)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        **kwargs
    ):
        """
        C2FNAS 모델 초기화

        Args:
            in_channels (int): 입력 채널 수 (CT의 경우 1)
            num_classes (int): 출력 클래스 수
        """
        super().__init__(in_channels, num_classes, **kwargs)

        # TODO: C2FNAS 아키텍처 구현
        # MONAI에서 가져오거나 커스텀 구현
        #
        # from monai.networks.nets import C2FNAS as MonaiC2FNAS
        # self.model = MonaiC2FNAS(...)

        self.encoder = None
        self.decoder = None
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Coarse-to-fine 전략으로 순전파

        Args:
            x (torch.Tensor): 입력 CT 영상 [배치, 1, D, H, W]

        Returns:
            torch.Tensor: 세그멘테이션 마스크 [배치, 클래스 수, D, H, W]
        """
        # TODO: C2FNAS forward pass 구현
        # MONAI의 C2FNAS 통합 또는 커스텀 버전 구현
        raise NotImplementedError(
            "C2FNAS forward pass 구현 필요.\n"
            "다음 옵션 중 선택:\n"
            "1. MONAI 사용: from monai.networks.nets import C2FNAS\n"
            "2. 커스텀 구현: Coarse-to-fine 전략 직접 구현"
        )

    def get_config(self):
        """모델 설정 정보 반환"""
        return super().get_config()
