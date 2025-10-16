"""
세그멘테이션 모델 베이스 인터페이스
====================================

모든 세그멘테이션 모델을 위한 추상 베이스 클래스입니다.
5개의 CNN 아키텍처 간 일관성과 상호 교환성을 보장합니다.

주요 기능:
- 통일된 인터페이스 제공
- 모델 간 쉬운 교체 가능
- 통합된 학습/추론 파이프라인
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseSegmentationModel(nn.Module, ABC):
    """
    세그멘테이션 모델을 위한 추상 베이스 클래스

    5개의 세그멘테이션 CNN이 이 클래스를 반드시 상속해야 합니다:
    - 일관된 인터페이스 제공
    - 모델 간 쉬운 교체
    - 통합된 학습/추론 파이프라인 지원

    사용 예시:
    ---------
    >>> from pancreas_cancer_diagnosis.segmentation.models import UNet3D  # doctest: +SKIP
    >>> model = UNet3D(in_channels=1, num_classes=3)  # doctest: +SKIP
    >>> output = model(input_tensor)  # doctest: +SKIP
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, **kwargs):
        """
        베이스 세그멘테이션 모델 초기화

        Args:
            in_channels (int): 입력 채널 수 (CT 영상의 경우 기본값 1)
            num_classes (int): 세그멘테이션 클래스 수 (기본값 3: 배경, 췌장, 종양)
            **kwargs: 모델별 특화 파라미터

        Note:
            - CT 영상은 그레이스케일이므로 in_channels=1
            - 췌장암 진단의 경우 3클래스 (배경, 정상 췌장, 종양)
        """
        super().__init__()
        # 입력 채널 수 저장
        self.in_channels = in_channels
        # 출력 클래스 수 저장
        self.num_classes = num_classes
        # 모델 이름 자동 저장 (재현성을 위해)
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파(forward pass)

        이 메서드는 반드시 하위 클래스에서 구현되어야 합니다.

        Args:
            x (torch.Tensor): 입력 텐서 [배치, 채널, 깊이, 높이, 너비]
                             예: [2, 1, 96, 96, 96] (배치 2개, CT 1채널, 96^3 크기)

        Returns:
            torch.Tensor: 출력 텐서 [배치, 클래스 수, 깊이, 높이, 너비]
                         예: [2, 3, 96, 96, 96] (3클래스 세그멘테이션)

        Raises:
            NotImplementedError: 하위 클래스에서 구현하지 않은 경우

        Note:
            - 출력은 각 클래스에 대한 로짓(logits) 또는 확률값
            - 일반적으로 softmax는 loss 함수에서 적용
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        재현성을 위한 모델 설정 정보 반환

        학습된 모델의 하이퍼파라미터와 설정을 저장하여
        나중에 동일한 설정으로 모델을 재생성할 수 있게 합니다.

        Returns:
            Dict[str, Any]: 모델 설정 딕셔너리
                - model_name: 모델 클래스 이름
                - in_channels: 입력 채널 수
                - num_classes: 출력 클래스 수

        Example:
            >>> model = UNet3D()  # doctest: +SKIP
            >>> config = model.get_config()  # doctest: +SKIP
            >>> print(config)  # doctest: +SKIP
            {'model_name': 'UNet3D', 'in_channels': 1, 'num_classes': 3}
        """
        return {
            "model_name": self.model_name,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
        }

    def load_checkpoint(self, checkpoint_path: str):
        """
        체크포인트에서 모델 가중치 로드

        학습된 모델의 가중치를 불러와 추론이나 추가 학습에 사용합니다.

        Args:
            checkpoint_path (str): 체크포인트 파일 경로
                                  예: 'outputs/unet/best_model.pth'

        Raises:
            FileNotFoundError: 체크포인트 파일이 없는 경우
            RuntimeError: 가중치 로드 중 오류 발생 시

        Note:
            - 체크포인트는 'state_dict' 키를 포함해야 함
            - CPU에 먼저 로드한 후 필요시 GPU로 이동
            - 모델 구조가 체크포인트와 일치해야 함

        Example:
            >>> model = UNet3D()  # doctest: +SKIP
            >>> model.load_checkpoint('best_unet.pth')  # doctest: +SKIP
            Loaded checkpoint from best_unet.pth
        """
        # CPU로 체크포인트 로드 (장치 독립적)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # state_dict만 로드 (가중치 정보)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"체크포인트 로드 완료: {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path: str, **kwargs):
        """
        모델 가중치를 체크포인트로 저장

        학습된 모델의 가중치와 설정을 파일로 저장합니다.

        Args:
            checkpoint_path (str): 저장할 체크포인트 파일 경로
                                  예: 'outputs/unet/best_model.pth'
            **kwargs: 추가로 저장할 정보 (예: epoch, loss, metrics)

        Note:
            - state_dict: 모델 가중치
            - config: 모델 설정 (재현성을 위해)
            - kwargs: 학습 정보 (epoch, loss 등)

        Example:
            >>> model.save_checkpoint('best_unet.pth', epoch=100, val_dice=0.92)  # doctest: +SKIP
            체크포인트 저장 완료: best_unet.pth
        """
        # 저장할 체크포인트 딕셔너리 생성
        checkpoint = {
            'state_dict': self.state_dict(),      # 모델 가중치
            'config': self.get_config(),          # 모델 설정
            **kwargs                               # 추가 정보
        }
        # 파일로 저장
        torch.save(checkpoint, checkpoint_path)
        print(f"체크포인트 저장 완료: {checkpoint_path}")
