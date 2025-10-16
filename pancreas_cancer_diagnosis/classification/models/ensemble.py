"""
앙상블 분류 모델
================

여러 분류 모델의 예측을 결합합니다.
다양한 앙상블 전략을 지원: 평균, 가중 평균, 투표, 스태킹

이 모델은 5개의 세그멘테이션 CNN 출력을 통합하여
최종 췌장암 진단을 내리는 핵심 컴포넌트입니다.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .base import BaseClassificationModel


class EnsembleClassifier(BaseClassificationModel):
    """
    여러 모델을 결합한 앙상블 분류기

    지원 기능:
    - 다중 분류 모델 통합
    - 5개 세그멘테이션 CNN 출력 처리
    - 다양한 앙상블 전략
    - 가중치 기반 결합
    - 불확실성 추정

    사용 예시:
    ---------
    >>> # 5개의 분류 모델 생성
    >>> models = [ResNet3D(), DenseNet3D(), ResNet3D(), DenseNet3D(), ResNet3D()]
    >>>
    >>> # 가중 평균 앙상블 생성
    >>> ensemble = EnsembleClassifier(
    ...     models=models,
    ...     ensemble_method='weighted',
    ...     weights=[0.25, 0.20, 0.20, 0.20, 0.15]
    ... )
    >>>
    >>> # 예측 수행
    >>> output = ensemble(input_tensor)
    """

    def __init__(
        self,
        models: List[BaseClassificationModel],
        ensemble_method: str = "average",
        weights: Optional[List[float]] = None,
        num_classes: int = 2,
        **kwargs
    ):
        """
        앙상블 분류기 초기화

        Args:
            models (List[BaseClassificationModel]): 분류 모델 리스트
                                                     일반적으로 5개 (세그멘테이션마다 하나)
            ensemble_method (str): 앙상블 방법
                - 'average': 단순 평균 (모든 모델 동등)
                - 'weighted': 가중 평균 (모델마다 다른 가중치)
                - 'voting': 하드 투표 (다수결)
                - 'stacking': 메타 분류기 학습
            weights (Optional[List[float]]): 각 모델의 가중치 (weighted 방법용)
                                            예: [0.25, 0.20, 0.20, 0.20, 0.15]
            num_classes (int): 클래스 수 (기본값 2: 정상/암)

        Note:
            - weights는 자동으로 정규화됨 (합=1.0)
            - stacking 방법은 메타 분류기가 추가로 학습됨
            - 5개 모델 사용 시 안정적인 예측 가능

        Example:
            >>> ensemble = EnsembleClassifier(
            ...     models=[model1, model2, model3, model4, model5],
            ...     ensemble_method='weighted',
            ...     weights=[0.3, 0.2, 0.2, 0.2, 0.1]  # 첫 번째 모델이 가장 중요
            ... )
        """
        super().__init__(in_channels=1, num_classes=num_classes, **kwargs)

        # 모델 리스트를 ModuleList로 저장 (PyTorch 최적화)
        self.models = nn.ModuleList(models)
        # 앙상블 방법 저장
        self.ensemble_method = ensemble_method
        # 모델 개수 저장
        self.num_models = len(models)

        # 가중치 설정
        if weights is None:
            # 가중치 미지정 시 모든 모델에 동일 가중치
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            # 가중치가 지정된 경우
            assert len(weights) == self.num_models, \
                f"가중치 개수({len(weights)})와 모델 개수({self.num_models})가 일치해야 합니다"
            self.weights = torch.tensor(weights)
            # 가중치 정규화 (합=1.0)
            self.weights = self.weights / self.weights.sum()

        # 스태킹 앙상블을 위한 메타 분류기
        if ensemble_method == "stacking":
            # 모든 모델의 출력을 입력으로 받아 최종 예측
            self.meta_classifier = nn.Linear(
                num_classes * self.num_models,  # 입력: 모든 모델의 예측
                num_classes                      # 출력: 최종 클래스
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        앙상블 순전파

        모든 모델의 예측을 수집하고 선택된 앙상블 방법으로 결합합니다.

        Args:
            x (torch.Tensor): 입력 텐서 [배치, 채널, 깊이, 높이, 너비]
                             또는 각 모델용 입력 리스트

        Returns:
            torch.Tensor: 앙상블 예측 [배치, 클래스 수]
                         예: [[2.3, -1.5]] → 로짓 형태

        Process:
            1. 각 모델로부터 개별 예측 수집
            2. 예측들을 텐서로 스택
            3. 선택된 앙상블 방법 적용
            4. 최종 예측 반환

        Example:
            >>> ensemble = EnsembleClassifier(models=[...])
            >>> output = ensemble(ct_scan)  # [1, 2]
            >>> probs = torch.softmax(output, dim=1)  # [[0.77, 0.23]]
        """
        # 모든 모델로부터 예측 수집
        predictions = []
        for model in self.models:
            pred = model(x)  # 각 모델의 예측
            predictions.append(pred)

        # 텐서로 스택: [모델 수, 배치, 클래스 수]
        predictions = torch.stack(predictions, dim=0)

        # 선택된 앙상블 방법 적용
        if self.ensemble_method == "average":
            return self._average_ensemble(predictions)
        elif self.ensemble_method == "weighted":
            return self._weighted_ensemble(predictions)
        elif self.ensemble_method == "voting":
            return self._voting_ensemble(predictions)
        elif self.ensemble_method == "stacking":
            return self._stacking_ensemble(predictions)
        else:
            raise ValueError(f"알 수 없는 앙상블 방법: {self.ensemble_method}")

    def _average_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        단순 평균 앙상블

        모든 모델의 예측을 동등하게 평균냅니다.
        가장 간단하지만 종종 효과적인 방법입니다.

        Args:
            predictions (torch.Tensor): 모든 모델의 예측 [모델 수, 배치, 클래스 수]

        Returns:
            torch.Tensor: 평균 예측 [배치, 클래스 수]

        Example:
            >>> # 3개 모델의 예측
            >>> preds = torch.tensor([
            ...     [[2.0, 1.0]],  # 모델 1
            ...     [[1.5, 1.5]],  # 모델 2
            ...     [[1.0, 2.0]]   # 모델 3
            ... ])
            >>> result = self._average_ensemble(preds)
            >>> # [[1.5, 1.5]] (평균)
        """
        return predictions.mean(dim=0)

    def _weighted_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        가중 평균 앙상블

        각 모델에 다른 가중치를 부여하여 평균냅니다.
        성능이 좋은 모델에 더 높은 가중치를 줄 수 있습니다.

        Args:
            predictions (torch.Tensor): 모든 모델의 예측 [모델 수, 배치, 클래스 수]

        Returns:
            torch.Tensor: 가중 평균 예측 [배치, 클래스 수]

        Note:
            - 가중치는 초기화 시 설정됨
            - 가중치 합은 자동으로 1.0으로 정규화됨
            - 검증 세트 성능 기반으로 가중치 설정 권장

        Example:
            >>> # weights = [0.5, 0.3, 0.2]
            >>> preds = torch.tensor([
            ...     [[2.0, 1.0]],  # 모델 1 (가중치 0.5)
            ...     [[1.0, 2.0]],  # 모델 2 (가중치 0.3)
            ...     [[0.0, 3.0]]   # 모델 3 (가중치 0.2)
            ... ])
            >>> result = self._weighted_ensemble(preds)
            >>> # [[1.3, 1.7]] (가중 평균)
        """
        # 가중치를 올바른 디바이스와 형태로 변환
        weights = self.weights.to(predictions.device).view(-1, 1, 1)
        # 가중 합 계산
        return (predictions * weights).sum(dim=0)

    def _voting_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        하드 투표 앙상블

        각 모델의 클래스 예측을 투표로 집계합니다.
        다수결 원칙을 따릅니다.

        Args:
            predictions (torch.Tensor): 모든 모델의 예측 [모델 수, 배치, 클래스 수]

        Returns:
            torch.Tensor: 투표 카운트 [배치, 클래스 수]

        Note:
            - 로짓을 확률로 변환 후 argmax로 클래스 선택
            - 각 클래스별 투표 수 반환
            - 가장 많은 투표를 받은 클래스가 최종 예측

        Example:
            >>> # 5개 모델의 예측
            >>> # 모델 1,2,3: 클래스 0 예측
            >>> # 모델 4,5: 클래스 1 예측
            >>> result = self._voting_ensemble(preds)
            >>> # [[3.0, 2.0]] → 클래스 0이 3표, 클래스 1이 2표
        """
        # 확률로 변환
        probs = torch.softmax(predictions, dim=2)
        # 각 모델의 클래스 예측 (argmax)
        votes = torch.argmax(probs, dim=2)  # [모델 수, 배치]

        # 투표 집계
        batch_size = predictions.shape[1]
        result = torch.zeros(batch_size, self.num_classes, device=predictions.device)

        # 각 모델의 투표를 카운트
        for i in range(self.num_models):
            for b in range(batch_size):
                result[b, votes[i, b]] += 1

        return result

    def _stacking_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        스태킹 앙상블

        메타 분류기를 사용하여 개별 모델의 예측을 결합합니다.
        가장 복잡하지만 가장 강력한 앙상블 방법입니다.

        Args:
            predictions (torch.Tensor): 모든 모델의 예측 [모델 수, 배치, 클래스 수]

        Returns:
            torch.Tensor: 메타 분류기의 예측 [배치, 클래스 수]

        Note:
            - 메타 분류기는 개별 모델의 예측을 입력으로 받음
            - 메타 분류기는 별도로 학습되어야 함
            - 교차 검증으로 학습 권장

        Example:
            >>> # 모든 모델의 예측을 하나로 연결
            >>> # [5 모델 × 2 클래스] = 10차원 입력
            >>> # 메타 분류기가 최종 2클래스 예측
        """
        # 모든 예측을 평탄화
        batch_size = predictions.shape[1]
        # [배치, 모델 수 × 클래스 수]로 변환
        stacked = predictions.permute(1, 0, 2).reshape(batch_size, -1)
        # 메타 분류기로 최종 예측
        return self.meta_classifier(stacked)

    def predict_proba_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        불확실성 추정을 포함한 예측

        여러 모델의 예측 분산을 불확실성으로 해석합니다.
        모델 간 의견 불일치가 클수록 불확실성이 높습니다.

        Args:
            x (torch.Tensor): 입력 텐서

        Returns:
            Dict[str, torch.Tensor]: 다음을 포함하는 딕셔너리
                - 'mean': 평균 예측 확률 [배치, 클래스 수]
                - 'std': 표준편차 (불확실성) [배치, 클래스 수]
                - 'individual': 개별 모델 예측 [모델 수, 배치, 클래스 수]

        Note:
            - 높은 std는 모델 간 의견 불일치 → 높은 불확실성
            - 낮은 std는 모델 간 합의 → 높은 확신
            - 임상 결정에 불확실성 정보 활용 가능

        Example:
            >>> result = ensemble.predict_proba_with_uncertainty(ct_scan)
            >>> print(result['mean'])  # [[0.77, 0.23]]
            >>> print(result['std'])   # [[0.05, 0.05]]  # 낮은 불확실성
            >>>
            >>> # 높은 불확실성 예시
            >>> # 모델1: [0.9, 0.1], 모델2: [0.1, 0.9] → std 높음
        """
        predictions = []
        # 각 모델의 확률 예측 수집
        for model in self.models:
            pred = model.predict_proba(x)
            predictions.append(pred)

        # 텐서로 스택
        predictions = torch.stack(predictions, dim=0)

        return {
            'mean': predictions.mean(dim=0),  # 평균 예측
            'std': predictions.std(dim=0),    # 불확실성
            'individual': predictions          # 개별 예측 (디버깅용)
        }

    def get_config(self):
        """
        앙상블 설정 정보 반환

        Returns:
            Dict: 설정 딕셔너리
                - model_name: 모델 이름
                - num_models: 모델 개수
                - ensemble_method: 앙상블 방법
                - weights: 각 모델의 가중치
        """
        config = super().get_config()
        config.update({
            "num_models": self.num_models,
            "ensemble_method": self.ensemble_method,
            "weights": self.weights.tolist(),
        })
        return config


class MultiInputEnsemble(EnsembleClassifier):
    """
    다중 입력 앙상블

    5개의 세그멘테이션 CNN 출력을 각각 처리하는 앙상블입니다.
    각 분류 모델이 하나의 세그멘테이션 출력을 처리하고,
    결과를 앙상블로 결합합니다.

    사용 시나리오:
    -------------
    1. 5개 세그멘테이션 CNN으로 췌장 영역 추출
       - UNet → seg_mask_1
       - ResUNet → seg_mask_2
       - VNet → seg_mask_3
       - AttentionUNet → seg_mask_4
       - C2FNAS → seg_mask_5

    2. 각 세그멘테이션 마스크에 대해 분류 모델 적용
       - ResNet3D(seg_mask_1) → pred_1
       - DenseNet3D(seg_mask_2) → pred_2
       - ResNet3D(seg_mask_3) → pred_3
       - DenseNet3D(seg_mask_4) → pred_4
       - ResNet3D(seg_mask_5) → pred_5

    3. 앙상블로 최종 진단
       - Ensemble(pred_1, ..., pred_5) → 정상 or 암

    Example:
        >>> # 5개 분류 모델 생성
        >>> models = [
        ...     ResNet3D(),   # seg_1용
        ...     DenseNet3D(), # seg_2용
        ...     ResNet3D(),   # seg_3용
        ...     DenseNet3D(), # seg_4용
        ...     ResNet3D()    # seg_5용
        ... ]
        >>>
        >>> # 다중 입력 앙상블 생성
        >>> ensemble = MultiInputEnsemble(models, num_seg_models=5)
        >>>
        >>> # 5개 세그멘테이션 출력으로 예측
        >>> seg_outputs = [seg1, seg2, seg3, seg4, seg5]
        >>> final_pred = ensemble(seg_outputs)
    """

    def __init__(
        self,
        models: List[BaseClassificationModel],
        num_seg_models: int = 5,
        **kwargs
    ):
        """
        다중 입력 앙상블 초기화

        Args:
            models (List[BaseClassificationModel]): 분류 모델 리스트
                                                     (세그멘테이션마다 하나)
            num_seg_models (int): 세그멘테이션 모델 수 (기본값 5)

        Raises:
            AssertionError: 모델 수가 세그멘테이션 수와 맞지 않을 때
        """
        super().__init__(models, **kwargs)
        self.num_seg_models = num_seg_models

        # 모델 개수 검증
        assert len(models) == num_seg_models, \
            f"모델 개수({len(models)})와 세그멘테이션 개수({num_seg_models})가 일치해야 합니다"

    def forward(self, seg_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        다중 세그멘테이션 입력으로 순전파

        각 세그멘테이션 출력을 해당 분류 모델에 전달하고
        결과를 앙상블로 결합합니다.

        Args:
            seg_outputs (List[torch.Tensor]): 5개 세그멘테이션 출력 리스트
                                              각각 [배치, 채널, 깊이, 높이, 너비]

        Returns:
            torch.Tensor: 앙상블 예측 [배치, 클래스 수]

        Raises:
            AssertionError: 세그멘테이션 출력 개수가 맞지 않을 때

        Example:
            >>> seg_outputs = [
            ...     unet_output,      # [1, 3, 96, 96, 96]
            ...     resunet_output,   # [1, 3, 96, 96, 96]
            ...     vnet_output,      # [1, 3, 96, 96, 96]
            ...     attunet_output,   # [1, 3, 96, 96, 96]
            ...     c2fnas_output     # [1, 3, 96, 96, 96]
            ... ]
            >>> prediction = ensemble(seg_outputs)  # [1, 2]
        """
        # 세그멘테이션 출력 개수 검증
        assert len(seg_outputs) == self.num_seg_models, \
            f"세그멘테이션 출력 {self.num_seg_models}개가 필요하지만 {len(seg_outputs)}개가 제공됨"

        # 각 모델에 해당 세그멘테이션 출력 전달
        predictions = []
        for i, (model, seg_out) in enumerate(zip(self.models, seg_outputs)):
            pred = model(seg_out)
            predictions.append(pred)

        # 텐서로 스택
        predictions = torch.stack(predictions, dim=0)

        # 선택된 앙상블 방법 적용
        if self.ensemble_method == "average":
            return self._average_ensemble(predictions)
        elif self.ensemble_method == "weighted":
            return self._weighted_ensemble(predictions)
        elif self.ensemble_method == "voting":
            return self._voting_ensemble(predictions)
        elif self.ensemble_method == "stacking":
            return self._stacking_ensemble(predictions)
        else:
            raise ValueError(f"알 수 없는 앙상블 방법: {self.ensemble_method}")
