"""src.config 기본 동작 테스트."""
from pathlib import Path

from src.config import PATHS, settings


def test_paths_under_project_root():
    """모든 경로가 프로젝트 루트 하위에 있어야 한다."""
    assert PATHS.raw_dir == PATHS.root / "data" / "raw"
    assert PATHS.processed_dir == PATHS.root / "data" / "processed"
    assert PATHS.competitors_dir == PATHS.root / "docs" / "competitors"
    assert isinstance(PATHS.root, Path)


def test_ensure_creates_dirs(tmp_path, monkeypatch):
    """ensure() 호출 시 디렉터리가 생성되어야 한다."""
    PATHS.ensure()
    assert PATHS.raw_dir.is_dir()
    assert PATHS.outputs_dir.is_dir()


def test_default_tokenizer():
    """기본 형태소 분석기는 kiwi 또는 whitespace 중 하나여야 한다."""
    assert settings.tokenizer in {"kiwi", "whitespace"}


def test_require_raises_when_missing():
    """미설정 비밀 값 요청 시 명확한 에러가 발생해야 한다."""
    import pytest

    # 테스트 환경에서 일반적으로 비어 있는 키 사용
    if not settings.openai_api_key:
        with pytest.raises(RuntimeError):
            settings.require("openai_api_key")
