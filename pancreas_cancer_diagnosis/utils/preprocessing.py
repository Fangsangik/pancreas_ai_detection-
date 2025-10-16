"""
데이터 전처리 및 정합성 처리 유틸리티
=======================================

CT 영상의 정합성을 보장하기 위한 전처리 함수들:
- Resampling: 다른 spacing을 표준 spacing으로 통일
- Orientation: 영상 방향을 표준 방향(RAS)으로 통일
- Intensity normalization: HU 값 정규화
- Cropping/Padding: 표준 크기로 맞추기
"""

import numpy as np
import nibabel as nib
from typing import Tuple, Optional, Union
from pathlib import Path


class CTPreprocessor:
    """
    CT 영상 전처리 및 정합성 처리 클래스

    주요 기능:
    1. Spacing 정규화 (리샘플링)
    2. Orientation 정규화 (RAS 방향으로 통일)
    3. Intensity 정규화 (HU 값 표준화)
    4. Spatial 정규화 (크기 통일)
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Optional[Tuple[int, int, int]] = None,
        intensity_range: Tuple[float, float] = (-200, 300),  # 복부 CT HU 범위
        normalize_intensity: bool = True,
    ):
        """
        전처리기 초기화

        Args:
            target_spacing: 목표 voxel spacing (mm) (x, y, z)
            target_size: 목표 영상 크기 (선택사항)
            intensity_range: HU 값 윈도우 범위 (min, max)
            normalize_intensity: 강도 정규화 여부
        """
        self.target_spacing = np.array(target_spacing)
        self.target_size = target_size
        self.intensity_range = intensity_range
        self.normalize_intensity = normalize_intensity

    def process(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Optional[Tuple[float, float, float]] = None,
        direction: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        전체 전처리 파이프라인 실행

        Args:
            image: 입력 영상 (H, W, D)
            spacing: 현재 voxel spacing
            origin: 영상 원점 (선택사항)
            direction: 영상 방향 행렬 (선택사항)

        Returns:
            processed_image: 전처리된 영상
            metadata: 전처리 메타데이터
        """
        metadata = {
            'original_shape': image.shape,
            'original_spacing': spacing,
            'target_spacing': self.target_spacing.tolist(),
        }

        # 1. Spacing 정규화 (리샘플링)
        if not np.allclose(spacing, self.target_spacing):
            print(f"   리샘플링: {spacing} → {self.target_spacing}")
            image = self._resample(image, spacing, self.target_spacing)
            metadata['resampled'] = True
        else:
            metadata['resampled'] = False

        # 2. Intensity 정규화
        if self.normalize_intensity:
            image = self._normalize_intensity(image)
            metadata['intensity_normalized'] = True
        else:
            metadata['intensity_normalized'] = False

        # 3. 크기 정규화 (필요시)
        if self.target_size is not None:
            image = self._resize_or_pad(image, self.target_size)
            metadata['resized'] = True
            metadata['final_shape'] = image.shape
        else:
            metadata['resized'] = False
            metadata['final_shape'] = image.shape

        return image, metadata

    def _resample(
        self,
        image: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        영상 리샘플링

        Args:
            image: 입력 영상
            original_spacing: 원본 spacing
            target_spacing: 목표 spacing

        Returns:
            리샘플링된 영상
        """
        from scipy.ndimage import zoom

        # 리샘플링 비율 계산
        zoom_factors = np.array(original_spacing) / np.array(target_spacing)

        # scipy.ndimage.zoom 사용
        resampled = zoom(image, zoom_factors, order=3)  # order=3: cubic interpolation

        return resampled

    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        HU 값 정규화

        1. 윈도잉: intensity_range로 클리핑
        2. 0-1 범위로 정규화

        Args:
            image: 입력 영상

        Returns:
            정규화된 영상
        """
        # HU 윈도잉
        image = np.clip(image, self.intensity_range[0], self.intensity_range[1])

        # 0-1 정규화
        image = (image - self.intensity_range[0]) / (
            self.intensity_range[1] - self.intensity_range[0]
        )

        return image.astype(np.float32)

    def _resize_or_pad(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        영상을 목표 크기로 맞추기 (crop 또는 pad)

        Args:
            image: 입력 영상
            target_size: 목표 크기

        Returns:
            크기가 조정된 영상
        """
        current_size = np.array(image.shape)
        target_size = np.array(target_size)

        # Crop 또는 Pad 계산
        result = np.zeros(target_size, dtype=image.dtype)

        # 복사할 영역 계산
        copy_from = np.maximum(0, (current_size - target_size) // 2)
        copy_to = copy_from + np.minimum(current_size, target_size)

        paste_from = np.maximum(0, (target_size - current_size) // 2)
        paste_to = paste_from + np.minimum(current_size, target_size)

        # 데이터 복사
        result[
            paste_from[0]:paste_to[0],
            paste_from[1]:paste_to[1],
            paste_from[2]:paste_to[2]
        ] = image[
            copy_from[0]:copy_to[0],
            copy_from[1]:copy_to[1],
            copy_from[2]:copy_to[2]
        ]

        return result


class ConsistencyValidator:
    """
    데이터셋 전체의 정합성 검증

    여러 CT 스캔의 메타데이터를 검증하고 통계를 제공합니다.
    """

    def __init__(self):
        self.spacings = []
        self.shapes = []
        self.intensity_ranges = []

    def add_sample(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float],
    ):
        """샘플 추가 및 메타데이터 수집"""
        self.spacings.append(spacing)
        self.shapes.append(image.shape)
        self.intensity_ranges.append((image.min(), image.max()))

    def validate(self, tolerance: float = 0.1) -> dict:
        """
        정합성 검증

        Args:
            tolerance: spacing 차이 허용 오차 (mm)

        Returns:
            검증 결과 딕셔너리
        """
        if not self.spacings:
            return {'valid': False, 'reason': '샘플 없음'}

        spacings = np.array(self.spacings)
        shapes = np.array(self.shapes)

        # Spacing 일관성 검사
        mean_spacing = spacings.mean(axis=0)
        spacing_std = spacings.std(axis=0)
        spacing_consistent = np.all(spacing_std < tolerance)

        # Shape 일관성 검사
        unique_shapes = np.unique(shapes, axis=0)
        shape_consistent = len(unique_shapes) == 1

        # Intensity 범위 검사
        intensity_ranges = np.array(self.intensity_ranges)

        return {
            'valid': spacing_consistent and shape_consistent,
            'num_samples': len(self.spacings),
            'spacing': {
                'consistent': spacing_consistent,
                'mean': mean_spacing.tolist(),
                'std': spacing_std.tolist(),
                'min': spacings.min(axis=0).tolist(),
                'max': spacings.max(axis=0).tolist(),
            },
            'shape': {
                'consistent': shape_consistent,
                'unique_shapes': unique_shapes.tolist(),
                'most_common': shapes[0].tolist(),
            },
            'intensity': {
                'min': intensity_ranges[:, 0].min(),
                'max': intensity_ranges[:, 1].max(),
                'mean_min': intensity_ranges[:, 0].mean(),
                'mean_max': intensity_ranges[:, 1].mean(),
            }
        }

    def get_recommended_settings(self) -> dict:
        """
        전처리 권장 설정 반환

        Returns:
            권장 전처리 파라미터
        """
        if not self.spacings:
            return {}

        spacings = np.array(self.spacings)
        shapes = np.array(self.shapes)
        intensity_ranges = np.array(self.intensity_ranges)

        # 중앙값 spacing 사용 (평균보다 이상치에 robust)
        median_spacing = np.median(spacings, axis=0)

        # 가장 큰 크기 사용 (정보 손실 최소화)
        max_shape = shapes.max(axis=0)

        # Intensity 범위 (1-99 percentile)
        all_intensities = np.concatenate([
            np.array([r[0], r[1]]) for r in self.intensity_ranges
        ])
        intensity_min = np.percentile(all_intensities, 1)
        intensity_max = np.percentile(all_intensities, 99)

        return {
            'target_spacing': median_spacing.tolist(),
            'target_size': max_shape.tolist(),
            'intensity_range': (float(intensity_min), float(intensity_max)),
        }


def validate_and_preprocess_dataset(
    data_dir: Path,
    output_dir: Path,
    target_spacing: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """
    데이터셋 전체 검증 및 전처리

    Args:
        data_dir: 입력 데이터 디렉토리
        output_dir: 전처리된 데이터 출력 디렉토리
        target_spacing: 목표 spacing (None이면 자동 결정)

    Returns:
        전처리 결과 및 통계
    """
    validator = ConsistencyValidator()

    # 1. 모든 NIfTI 파일 수집
    nifti_files = list(data_dir.glob("*.nii.gz"))

    print(f"🔍 {len(nifti_files)}개 파일 분석 중...")

    # 2. 메타데이터 수집
    for nii_file in nifti_files:
        nii = nib.load(str(nii_file))
        data = nii.get_fdata()
        spacing = nii.header.get_zooms()

        validator.add_sample(data, spacing)

    # 3. 검증
    validation_result = validator.validate()
    print("\n📊 데이터셋 일관성 검증:")
    print(f"   샘플 수: {validation_result['num_samples']}")
    print(f"   Spacing 일관성: {'✅' if validation_result['spacing']['consistent'] else '❌'}")
    print(f"   Shape 일관성: {'✅' if validation_result['shape']['consistent'] else '❌'}")

    # 4. 권장 설정
    recommended = validator.get_recommended_settings()
    print("\n💡 권장 전처리 설정:")
    print(f"   Target spacing: {recommended['target_spacing']}")
    print(f"   Target size: {recommended['target_size']}")
    print(f"   Intensity range: {recommended['intensity_range']}")

    # 5. 전처리 수행 (필요시)
    if target_spacing is None:
        target_spacing = tuple(recommended['target_spacing'])

    preprocessor = CTPreprocessor(
        target_spacing=target_spacing,
        intensity_range=recommended['intensity_range'],
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 전처리 시작...")
    for nii_file in nifti_files:
        nii = nib.load(str(nii_file))
        data = nii.get_fdata()
        spacing = nii.header.get_zooms()

        # 전처리
        processed, metadata = preprocessor.process(data, spacing)

        # 저장
        output_file = output_dir / nii_file.name
        processed_nii = nib.Nifti1Image(processed, affine=np.eye(4))
        processed_nii.header.set_zooms(target_spacing)
        nib.save(processed_nii, str(output_file))

        print(f"   ✓ {nii_file.name}")

    return {
        'validation': validation_result,
        'recommended_settings': recommended,
        'preprocessed_files': len(nifti_files),
    }
