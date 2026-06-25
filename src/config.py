"""
중앙 설정 모듈 (config.py)

- .env 파일에서 환경 변수를 읽어와 애플리케이션 전역에서 사용한다.
- 실제 API 키 / 내부 정보는 코드에 하드코딩하지 않고 .env 에서만 주입받는다.
- 경로(Path) 관련 상수를 한 곳에서 관리하여 모듈 간 일관성을 유지한다.

사용 예:
    from src.config import settings, PATHS
    print(settings.openai_model)
    print(PATHS.raw_dir)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# .env 로드 (프로젝트 루트 기준). 이미 설정된 OS 환경 변수는 덮어쓰지 않는다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get(key: str, default: str = "") -> str:
    """환경 변수를 문자열로 읽는다 (없으면 default)."""
    return os.getenv(key, default).strip()


# .env.example 의 placeholder 값. 실제 값으로 교체되지 않았는지 판별하는 데 사용.
_PLACEHOLDERS = {"your_key_here", "your_model_here", ""}


@dataclass(frozen=True)
class Paths:
    """프로젝트 디렉터리 경로 모음."""
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    docs_dir: Path = PROJECT_ROOT / "docs"
    competitors_dir: Path = PROJECT_ROOT / "docs" / "competitors"
    models_dir: Path = PROJECT_ROOT / "models"
    notebooks_dir: Path = PROJECT_ROOT / "notebooks"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    config_dir: Path = PROJECT_ROOT / "config"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"

    def ensure(self) -> None:
        """필요한 디렉터리가 없으면 생성한다."""
        for p in (
            self.raw_dir,
            self.processed_dir,
            self.competitors_dir,
            self.models_dir,
            self.outputs_dir,
            self.prompts_dir,
            self.config_dir,
            self.chroma_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """환경 변수 기반 애플리케이션 설정."""
    # OpenAI (답글 생성 / AI 리포트)
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _get("OPENAI_MODEL"))

    # 앱/스토어 식별자
    google_play_app_id: str = field(default_factory=lambda: _get("GOOGLE_PLAY_APP_ID"))
    app_store_app_id: str = field(default_factory=lambda: _get("APP_STORE_APP_ID"))
    app_store_country: str = field(default_factory=lambda: _get("APP_STORE_COUNTRY", "kr"))

    # 벡터 DB
    chroma_persist_dir: str = field(
        default_factory=lambda: _get("CHROMA_PERSIST_DIR", "./chroma_db")
    )

    # 형태소 분석기: "kiwi" | "whitespace"
    tokenizer: str = field(default_factory=lambda: _get("TOKENIZER", "kiwi").lower())

    # 실행 환경
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())
    env: str = field(default_factory=lambda: _get("ENV", "development").lower())

    @property
    def openai_enabled(self) -> bool:
        """OpenAI 키/모델이 실제 값으로 설정되어 답글 생성을 시도할 수 있는지 여부.

        미설정이면 reply_generator 등에서 룰 기반 fallback 으로 분기한다.
        """
        return (
            self.openai_api_key not in _PLACEHOLDERS
            and self.openai_model not in _PLACEHOLDERS
        )

    def require(self, key: str) -> str:
        """필수 비밀 값을 안전하게 가져온다. 없거나 placeholder면 명확한 에러를 던진다."""
        value = getattr(self, key, "")
        if value in _PLACEHOLDERS:
            raise RuntimeError(
                f"필수 환경 변수 '{key}' 가 설정되지 않았습니다. "
                f".env 파일을 확인하세요 (.env.example 참고)."
            )
        return value


# 전역 싱글턴 인스턴스
PATHS = Paths()
settings = Settings()


if __name__ == "__main__":
    # 디버그용: 키 값은 노출하지 않고 설정 여부만 출력한다.
    PATHS.ensure()
    print(f"PROJECT_ROOT     : {PATHS.root}")
    print(f"ENV              : {settings.env}")
    print(f"TOKENIZER        : {settings.tokenizer}")
    print(f"CHROMA_DIR       : {settings.chroma_persist_dir}")
    print(f"OPENAI_ENABLED   : {settings.openai_enabled}")
    print(f"OPENAI_MODEL     : {settings.openai_model or '(미설정)'}")
    print(f"GOOGLE_PLAY_APP  : {settings.google_play_app_id or '(미설정)'}")
    print(f"APP_STORE_APP    : {settings.app_store_app_id or '(미설정)'}")
