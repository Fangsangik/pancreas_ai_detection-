"""
데이터 구조 테스트
"""
import pytest
from pathlib import Path


class TestDataStructure:
    """데이터 디렉토리 구조 테스트"""

    @pytest.fixture
    def project_root(self):
        """프로젝트 루트 디렉토리"""
        return Path(__file__).parent.parent

    def test_data_directory_exists(self, project_root):
        """data 디렉토리 존재 확인"""
        data_dir = project_root / "data"
        assert data_dir.exists(), "data 디렉토리가 없습니다"

    def test_required_subdirectories(self, project_root):
        """필수 하위 디렉토리 확인"""
        data_dir = project_root / "data"

        required_dirs = [
            "manifests",
            "nifti",
            "raw_dicom",
            "labels"
        ]

        for dirname in required_dirs:
            subdir = data_dir / dirname
            assert subdir.exists(), f"{dirname} 디렉토리가 없습니다: {subdir}"

    def test_manifest_directory(self, project_root):
        """manifests 디렉토리 확인"""
        manifests_dir = project_root / "data" / "manifests"

        # manifest 파일 확인
        manifest_file = manifests_dir / "pancreas_ct_manifest.csv"
        assert manifest_file.exists(), "pancreas_ct_manifest.csv 파일이 없습니다"
