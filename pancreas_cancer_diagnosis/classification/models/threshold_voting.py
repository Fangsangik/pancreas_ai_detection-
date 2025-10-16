"""
Threshold-based Voting Ensemble
================================

"5개 중 4개 이상에서 암(1)이 나오면 암으로 판정" 가설 구현

핵심 아이디어:
- 5개 세그멘테이션 모델 각각이 독립적으로 암/정상 분류
- 4개 이상의 모델이 암(1)으로 예측하면 최종 암 판정
- Majority voting보다 엄격한 기준 (3/5 → 4/5)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from .base import BaseClassificationModel


class ThresholdVotingEnsemble(BaseClassificationModel):
    """
    임계값 기반 투표 앙상블

    가설: "5개 세그멘테이션 모델 중 4개 이상에서 암이 검출되면 실제 암일 확률이 높다"

    워크플로우:
    ----------
    1. 5개 세그멘테이션 CNN으로 각각 마스크 생성
    2. 각 마스크를 개별 분류 모델에 입력
    3. 5개 분류 모델의 예측 수집
    4. 4개 이상이 암(1)으로 예측하면 암으로 판정

    장점:
    -----
    - 엄격한 기준으로 False Positive 감소
    - 의료 진단에서 중요한 신뢰성 확보
    - 해석 가능한 규칙 기반 판정

    단점:
    -----
    - False Negative 증가 가능성 (암을 놓칠 위험)
    - 유연성 부족 (확률 정보 손실)

    사용 시나리오:
    ------------
    - 특이도(Specificity)가 중요한 경우
    - 추가 검사를 위한 1차 스크리닝
    - 높은 확신이 필요한 임상 결정

    Example:
        >>> # 5개 분류 모델 준비
        >>> models = [ResNet3D(in_channels=3, num_classes=2) for _ in range(5)]
        >>>
        >>> # 앙상블 생성
        >>> ensemble = ThresholdVotingEnsemble(
        ...     models=models,
        ...     threshold=4,  # 5개 중 4개 이상
        ...     num_classes=2
        ... )
        >>>
        >>> # 5개 세그멘테이션 마스크 입력
        >>> seg_outputs = [torch.randn(1, 3, 96, 96, 96) for _ in range(5)]
        >>> result = ensemble.predict_with_confidence(seg_outputs)
        >>>
        >>> print(f"예측: {result['prediction']}")  # 0 or 1
        >>> print(f"투표 수: {result['vote_count']}/5")  # 예: 4/5
        >>> print(f"신뢰도: {result['confidence']:.2%}")  # 예: 80%
    """

    def __init__(
        self,
        models: List[nn.Module],
        threshold: int = 4,
        num_classes: int = 2,
        **kwargs
    ):
        """
        임계값 기반 투표 앙상블 초기화

        Args:
            models (List[nn.Module]): 5개 분류 모델 리스트
                                     각 모델은 세그멘테이션 마스크를 입력으로 받음
            threshold (int): 암 판정을 위한 최소 투표 수 (기본값: 4)
                            예: threshold=4 → 5개 중 4개 이상
            num_classes (int): 클래스 수 (기본값: 2, 정상/암)

        Raises:
            AssertionError: 모델 수가 5개가 아니거나 threshold가 유효하지 않을 때

        Note:
            - 모든 모델은 동일한 아키텍처일 필요는 없음
            - 각 모델은 독립적으로 학습 가능
            - threshold=4는 엄격한 기준 (80% 동의)
            - threshold=3은 일반적인 majority voting (60% 동의)
        """
        super().__init__(num_classes=num_classes, **kwargs)

        assert len(models) == 5, "정확히 5개의 분류 모델이 필요합니다"
        assert 1 <= threshold <= 5, f"threshold는 1~5 사이여야 합니다 (현재: {threshold})"

        self.models = nn.ModuleList(models)
        self.threshold = threshold
        self.num_models = len(models)

    def forward(self, seg_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        순전파: 5개 모델의 예측을 수집하고 투표

        Args:
            seg_outputs (List[torch.Tensor]): 5개 세그멘테이션 마스크
                                              각각 [배치, 채널, D, H, W]

        Returns:
            torch.Tensor: 최종 로짓 [배치, 클래스 수]
                         threshold 이상이면 클래스 1, 미만이면 클래스 0

        Note:
            - 각 모델의 로짓을 수집 후 투표로 변환
            - 실제 확률이 아닌 이진 결정 기반
        """
        assert len(seg_outputs) == 5, "5개의 세그멘테이션 출력이 필요합니다"

        batch_size = seg_outputs[0].size(0)
        device = seg_outputs[0].device

        # 각 모델의 예측 수집
        predictions = []
        for i, model in enumerate(self.models):
            logits = model(seg_outputs[i])  # [배치, 2]
            pred_class = torch.argmax(logits, dim=1)  # [배치]
            predictions.append(pred_class)

        # [5, 배치] → [배치, 5]
        predictions = torch.stack(predictions, dim=1)

        # 각 샘플에 대해 암(1) 투표 수 계산
        cancer_votes = (predictions == 1).sum(dim=1)  # [배치]

        # Threshold 기반 최종 판정
        final_prediction = (cancer_votes >= self.threshold).long()  # [배치]

        # 로짓 형태로 변환 (cross entropy loss 호환)
        logits = torch.zeros(batch_size, self.num_classes, device=device)
        logits[range(batch_size), final_prediction] = 1.0

        return logits

    def predict_with_confidence(
        self,
        seg_outputs: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        신뢰도와 함께 예측 수행

        Args:
            seg_outputs (List[torch.Tensor]): 5개 세그멘테이션 마스크

        Returns:
            Dict[str, torch.Tensor]:
                - 'prediction': 최종 예측 [배치] (0 또는 1)
                - 'vote_count': 암에 투표한 모델 수 [배치] (0~5)
                - 'confidence': 신뢰도 [배치] (투표 비율, 0.0~1.0)
                - 'individual_preds': 각 모델의 예측 [배치, 5]
                - 'individual_probs': 각 모델의 확률 [배치, 5, 2]

        Example:
            >>> result = ensemble.predict_with_confidence(seg_outputs)
            >>>
            >>> # 샘플 1: 5개 중 4개가 암으로 예측
            >>> print(result['prediction'][0])  # 1 (암)
            >>> print(result['vote_count'][0])  # 4
            >>> print(result['confidence'][0])  # 0.8 (80%)
            >>>
            >>> # 샘플 2: 5개 중 2개만 암으로 예측
            >>> print(result['prediction'][1])  # 0 (정상)
            >>> print(result['vote_count'][1])  # 2
            >>> print(result['confidence'][1])  # 0.4 (40%)
        """
        self.eval()

        with torch.no_grad():
            batch_size = seg_outputs[0].size(0)
            device = seg_outputs[0].device

            # 각 모델의 예측 및 확률 수집
            individual_preds = []
            individual_probs = []

            for i, model in enumerate(self.models):
                logits = model(seg_outputs[i])  # [배치, 2]
                probs = torch.softmax(logits, dim=1)  # [배치, 2]
                pred = torch.argmax(logits, dim=1)  # [배치]

                individual_preds.append(pred)
                individual_probs.append(probs)

            # [5, 배치] → [배치, 5]
            individual_preds = torch.stack(individual_preds, dim=1)
            # [5, 배치, 2] → [배치, 5, 2]
            individual_probs = torch.stack(individual_probs, dim=1)

            # 암(1) 투표 수 계산
            vote_count = (individual_preds == 1).sum(dim=1)  # [배치]

            # 최종 예측 (threshold 기반)
            final_prediction = (vote_count >= self.threshold).long()

            # 신뢰도 계산 (투표 비율)
            confidence = vote_count.float() / self.num_models  # [배치]

            return {
                'prediction': final_prediction,
                'vote_count': vote_count,
                'confidence': confidence,
                'individual_preds': individual_preds,
                'individual_probs': individual_probs
            }

    def get_diagnosis_report(
        self,
        seg_outputs: List[torch.Tensor],
        patient_id: str = None
    ) -> Dict:
        """
        임상 진단 보고서 생성

        Args:
            seg_outputs (List[torch.Tensor]): 5개 세그멘테이션 마스크
            patient_id (str): 환자 ID (선택사항)

        Returns:
            Dict: 진단 보고서
                - patient_id: 환자 ID
                - final_diagnosis: 최종 진단 ('암' 또는 '정상')
                - confidence_level: 신뢰도 수준 ('매우 높음', '높음', '중간', '낮음')
                - vote_result: 투표 결과 (예: "5개 중 4개 모델이 암으로 판정")
                - individual_results: 각 모델의 판정 결과
                - recommendation: 임상 권고사항

        Example:
            >>> report = ensemble.get_diagnosis_report(seg_outputs, patient_id="P001")
            >>>
            >>> print(f"환자: {report['patient_id']}")
            >>> print(f"진단: {report['final_diagnosis']}")
            >>> print(f"신뢰도: {report['confidence_level']}")
            >>> print(f"권고: {report['recommendation']}")

            출력:
            환자: P001
            진단: 암
            신뢰도: 높음
            권고: 추가 정밀 검사 및 전문의 상담 권장
        """
        result = self.predict_with_confidence(seg_outputs)

        # 단일 샘플만 처리 (배치의 첫 번째)
        prediction = result['prediction'][0].item()
        vote_count = result['vote_count'][0].item()
        confidence = result['confidence'][0].item()
        individual_preds = result['individual_preds'][0].cpu().numpy()

        # 최종 진단
        final_diagnosis = "췌장암 의심" if prediction == 1 else "정상"

        # 신뢰도 수준 분류
        if confidence >= 0.8:  # 4개 이상 / 5개
            confidence_level = "매우 높음"
        elif confidence >= 0.6:  # 3개 / 5개
            confidence_level = "높음"
        elif confidence >= 0.4:  # 2개 / 5개
            confidence_level = "중간"
        else:  # 0~1개 / 5개
            confidence_level = "낮음"

        # 투표 결과 텍스트
        vote_result = f"5개 모델 중 {vote_count}개가 암으로 판정"

        # 각 모델의 결과
        model_names = ["UNet", "ResUNet", "VNet", "AttentionUNet", "C2FNAS"]
        individual_results = []
        for i, (name, pred) in enumerate(zip(model_names, individual_preds)):
            result_text = "암" if pred == 1 else "정상"
            individual_results.append(f"{name}: {result_text}")

        # 임상 권고사항
        if prediction == 1:
            if confidence >= 0.8:
                recommendation = "즉시 추가 정밀 검사 및 전문의 상담 권장"
            else:
                recommendation = "추가 영상 검사 및 전문의 재평가 권장"
        else:
            if vote_count >= 2:
                recommendation = "정상 판정이나 일부 모델에서 이상 소견. 정기 검진 권장"
            else:
                recommendation = "정상 판정. 정기 검진 유지"

        return {
            'patient_id': patient_id or "Unknown",
            'final_diagnosis': final_diagnosis,
            'confidence_level': confidence_level,
            'confidence_score': f"{confidence:.1%}",
            'vote_result': vote_result,
            'threshold_used': f"{self.threshold}/{self.num_models}",
            'individual_results': individual_results,
            'recommendation': recommendation,
            'raw_confidence': confidence,
            'raw_vote_count': vote_count
        }

    def analyze_threshold_sensitivity(
        self,
        seg_outputs: List[torch.Tensor]
    ) -> Dict[int, Dict]:
        """
        다양한 threshold 값에 대한 민감도 분석

        Args:
            seg_outputs (List[torch.Tensor]): 5개 세그멘테이션 마스크

        Returns:
            Dict[int, Dict]: Threshold별 결과
                {
                    3: {'prediction': 1, 'vote_count': 4, 'diagnosis': '암'},
                    4: {'prediction': 1, 'vote_count': 4, 'diagnosis': '암'},
                    5: {'prediction': 0, 'vote_count': 4, 'diagnosis': '정상'}
                }

        Note:
            - Threshold 변화에 따른 판정 변화 확인
            - 임상 결정에 대한 민감도 평가

        Example:
            >>> analysis = ensemble.analyze_threshold_sensitivity(seg_outputs)
            >>>
            >>> print("Threshold 민감도 분석:")
            >>> for threshold, result in analysis.items():
            ...     print(f"  {threshold}/5: {result['diagnosis']} (투표: {result['vote_count']})")

            출력:
            Threshold 민감도 분석:
              3/5: 암 (투표: 4)
              4/5: 암 (투표: 4)
              5/5: 정상 (투표: 4)
        """
        result = self.predict_with_confidence(seg_outputs)
        vote_count = result['vote_count'][0].item()

        analysis = {}
        for threshold in range(1, 6):
            prediction = 1 if vote_count >= threshold else 0
            diagnosis = "췌장암 의심" if prediction == 1 else "정상"

            analysis[threshold] = {
                'prediction': prediction,
                'vote_count': vote_count,
                'diagnosis': diagnosis,
                'threshold_desc': f"{threshold}/{self.num_models} 이상"
            }

        return analysis


def create_threshold_ensemble(
    seg_model_names: List[str] = None,
    threshold: int = 4,
    **model_kwargs
) -> ThresholdVotingEnsemble:
    """
    Threshold Voting Ensemble 생성 헬퍼 함수

    Args:
        seg_model_names (List[str]): 세그멘테이션 모델 이름들
                                     기본값: ["unet", "resunet", "vnet", "attunet", "c2fnas"]
        threshold (int): 암 판정 임계값 (기본값: 4)
        **model_kwargs: 분류 모델 파라미터

    Returns:
        ThresholdVotingEnsemble: 구성된 앙상블

    Example:
        >>> ensemble = create_threshold_ensemble(
        ...     threshold=4,
        ...     in_channels=3,
        ...     num_classes=2
        ... )
    """
    if seg_model_names is None:
        seg_model_names = ["unet", "resunet", "vnet", "attunet", "c2fnas"]

    # TODO: 실제 모델 인스턴스 생성
    # 여기서는 플레이스홀더로 None 반환
    # 실제 구현 시 ResNet3D 등의 모델 생성

    from .resnet3d import ResNet3D

    models = []
    for name in seg_model_names:
        model = ResNet3D(num_classes=2, **model_kwargs)
        models.append(model)

    return ThresholdVotingEnsemble(
        models=models,
        threshold=threshold,
        num_classes=2
    )
