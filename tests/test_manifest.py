"""
Manifest 파일 테스트
"""
import pytest
import pandas as pd
from pathlib import Path


class TestManifest:
    """Manifest 파일 테스트 클래스"""

    @pytest.fixture
    def manifest_path(self):
        """Manifest 파일 경로 fixture"""
        return Path(__file__).parent.parent / "data" / "manifests" / "pancreas_ct_manifest.csv"

    def test_manifest_exists(self, manifest_path):
        """Manifest 파일이 존재하는지 확인"""
        assert manifest_path.exists(), f"Manifest 파일이 없습니다: {manifest_path}"

    def test_manifest_structure(self, manifest_path):
        """Manifest 파일 구조 확인"""
        df = pd.read_csv(manifest_path)

        # 필수 컬럼 확인
        required_columns = [
            'Subject ID',
            'Collection',
            'Study Description',
            'Study Date',
            'Modality',
            'Number of Images',
            'File Location'
        ]

        for col in required_columns:
            assert col in df.columns, f"필수 컬럼이 없습니다: {col}"

    def test_manifest_data(self, manifest_path):
        """Manifest 데이터 검증"""
        df = pd.read_csv(manifest_path)

        # 데이터가 비어있지 않은지 확인
        assert len(df) > 0, "Manifest에 데이터가 없습니다"

        # 환자 수 확인 (6명)
        assert len(df) == 6, f"예상 환자 수: 6, 실제: {len(df)}"

        # Modality가 모두 CT인지 확인
        assert all(df['Modality'] == 'CT'), "Modality가 CT가 아닌 데이터가 있습니다"

        # Number of Images가 양수인지 확인
        assert all(df['Number of Images'] > 0), "이미지 수가 0 이하인 데이터가 있습니다"

    def test_file_locations(self, manifest_path):
        """파일 경로가 유효한지 확인"""
        df = pd.read_csv(manifest_path)

        for idx, row in df.iterrows():
            file_path = Path(row['File Location'])
            # 경로가 절대 경로인지 확인
            assert file_path.is_absolute(), f"상대 경로입니다: {file_path}"
