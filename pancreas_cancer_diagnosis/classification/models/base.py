"""
분류 모델 베이스 인터페이스
============================

모든 분류 모델을 위한 추상 베이스 클래스입니다.
단일 모델과 앙상블 아키텍처를 모두 지원합니다.

주요 기능:
- 통일된 분류 인터페이스
- 앙상블 모델 지원
- 세그멘테이션 출력과 쉬운 통합
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseClassificationModel(nn.Module, ABC):
    """
    분류 모델을 위한 추상 베이스 클래스

    일관된 인터페이스 보장:
    - 단일 분류 모델
    - 앙상블 모델
    - 세그멘테이션 출력과 쉬운 통합

    사용 예시:
    ---------
    >>> from pancreas_cancer_diagnosis.classification.models import ResNet3D
    >>> model = ResNet3D(in_channels=3, num_classes=2)  # 3채널 입력, 2클래스 (정상/암)
    >>> output = model(seg_mask)  # [B, 2] (로짓)
    >>> probs = model.predict_proba(seg_mask)  # [B, 2] (확률)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, **kwargs):
        """
        베이스 분류 모델 초기화

        Args:
            in_channels (int): 입력 채널 수
                              - 1: 원본 CT 영상
                              - 3: 세그멘테이션 마스크 (배경, 췌장, 종양)
            num_classes (int): 클래스 수 (기본값 2: 정상 vs 암)
            **kwargs: 모델별 특화 파라미터

        Note:
            - 세그멘테이션 출력을 입력으로 사용하는 경우 in_channels=3
            - 이진 분류 문제: 정상(0) vs 췌장암(1)
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
                             예: [2, 3, 96, 96, 96] (세그멘테이션 마스크)

        Returns:
            torch.Tensor: 출력 텐서 [배치, 클래스 수]
                         예: [2, 2] (로짓 또는 확률)

        Raises:
            NotImplementedError: 하위 클래스에서 구현하지 않은 경우

        Note:
            - 출력은 로짓(logits) 형태
            - 확률로 변환하려면 predict_proba() 사용
            - 클래스 예측은 predict() 사용
        """
        pass

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        확률 예측 반환

        로짓을 softmax를 통해 확률로 변환합니다.

        Args:
            x (torch.Tensor): 입력 텐서 [배치, 채널, 깊이, 높이, 너비]

        Returns:
            torch.Tensor: 확률 텐서 [배치, 클래스 수]
                         예: [[0.23, 0.77], [0.85, 0.15]]
                         (첫 번째 샘플: 정상 23%, 암 77%)

        Example:
            >>> model = ResNet3D()
            >>> probs = model.predict_proba(input_tensor)
            >>> print(probs)
            tensor([[0.23, 0.77],
                    [0.85, 0.15]])
        """
        # 순전파로 로짓 획득
        logits = self.forward(x)
        # softmax로 확률 변환
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        클래스 예측 반환

        가장 높은 확률을 가진 클래스의 인덱스를 반환합니다.

        Args:
            x (torch.Tensor): 입력 텐서

        Returns:
            torch.Tensor: 클래스 예측 [배치]
                         예: [1, 0] (첫 번째: 암, 두 번째: 정상)

        Example:
            >>> model = ResNet3D()
            >>> predictions = model.predict(input_tensor)
            >>> print(predictions)
            tensor([1, 0])  # 1=암, 0=정상
        """
        # 확률 계산
        probs = self.predict_proba(x)
        # 가장 높은 확률의 클래스 선택
        return torch.argmax(probs, dim=1)

    def get_config(self) -> Dict[str, Any]:
        """
        재현성을 위한 모델 설정 정보 반환

        학습된 모델의 하이퍼파라미터와 설정을 저장합니다.

        Returns:
            Dict[str, Any]: 모델 설정 딕셔너리
                - model_name: 모델 클래스 이름
                - in_channels: 입력 채널 수
                - num_classes: 출력 클래스 수

        Example:
            >>> model = ResNet3D(in_channels=3, num_classes=2)
            >>> config = model.get_config()
            >>> print(config)
            {'model_name': 'ResNet3D', 'in_channels': 3, 'num_classes': 2}
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
                                  예: 'outputs/resnet3d/best_model.pth'

        Raises:
            FileNotFoundError: 체크포인트 파일이 없는 경우
            RuntimeError: 가중치 로드 중 오류 발생 시

        Note:
            - 체크포인트는 'state_dict' 키를 포함해야 함
            - CPU에 먼저 로드한 후 필요시 GPU로 이동
            - 모델 구조가 체크포인트와 일치해야 함

        Example:
            >>> model = ResNet3D()
            >>> model.load_checkpoint('best_resnet.pth')
            체크포인트 로드 완료: best_resnet.pth
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
                                  예: 'outputs/resnet3d/best_model.pth'
            **kwargs: 추가로 저장할 정보 (예: epoch, accuracy, loss)

        Note:
            - state_dict: 모델 가중치
            - config: 모델 설정 (재현성을 위해)
            - kwargs: 학습 정보 (epoch, accuracy 등)

        Example:
            >>> model.save_checkpoint(
            ...     'best_resnet.pth',
            ...     epoch=50,
            ...     val_acc=0.95,
            ...     val_auroc=0.97
            ... )
            체크포인트 저장 완료: best_resnet.pth
        """
        # 저장할 체크포인트 딕셔너리 생성
        checkpoint = {
            'state_dict': self.state_dict(),      # 모델 가중치
            'config': self.get_config(),          # 모델 설정
            **kwargs                               # 추가 정보 (메트릭 등)
        }
        # 파일로 저장
        torch.save(checkpoint, checkpoint_path)
        print(f"체크포인트 저장 완료: {checkpoint_path}")
