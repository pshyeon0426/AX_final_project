"""
데이터 출처 매니페스트 (source_manifest.py)

목적:
- 공개 리뷰 / 경쟁 앱 공개 문서를 "직접 수집"했다는 근거(출처·방법·약관 메모)를
  표 형태로 기록하여 재현성과 데이터 윤리를 입증한다.
- 실제 고객정보·내부 비공개 데이터는 절대 포함하지 않으며,
  공개된 출처(스토어 리뷰/설명/릴리즈 노트/뉴스)와 샘플 데이터만 기록한다.

사용 예:
    from src.source_manifest import write_manifest, SourceRecord
    write_manifest()                       # 기본 시드 + 빈 매니페스트 생성
    # 또는 직접 레코드를 구성해 저장
    write_manifest([SourceRecord(...), ...])
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.config import PATHS

# ----------------------------------------------------------------------
# 허용 값 (검증용)
# ----------------------------------------------------------------------
SOURCE_TYPES = ("review", "app_description", "release_note", "news", "sample")
COLLECTION_METHODS = (
    "google_play_scraper",
    "app_store_manual_csv",
    "manual_download",
    "sample_generated",
)

# 필수 컬럼 순서 (CSV 헤더와 동일)
COLUMNS = (
    "source_name",
    "source_type",
    "app_name",
    "url_or_file",
    "collection_method",
    "collected_at",
    "terms_note",
    "status",
)

MANIFEST_PATH = PATHS.outputs_dir / "source_manifest.csv"


def _now_iso() -> str:
    """수집 시각(UTC, ISO8601). 실제 수집 시점에 호출하여 기록한다."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SourceRecord:
    """데이터 출처 1건."""

    source_name: str
    source_type: str
    app_name: str
    url_or_file: str
    collection_method: str
    terms_note: str = ""
    status: str = "planned"  # planned | collected | failed | skipped
    collected_at: str = field(default_factory=_now_iso)

    def validate(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"source_type '{self.source_type}' 은 허용되지 않습니다. "
                f"허용: {SOURCE_TYPES}"
            )
        if self.collection_method not in COLLECTION_METHODS:
            raise ValueError(
                f"collection_method '{self.collection_method}' 은 허용되지 않습니다. "
                f"허용: {COLLECTION_METHODS}"
            )

    def as_row(self) -> dict[str, str]:
        return {k: asdict(self)[k] for k in COLUMNS}


def seed_records() -> list[SourceRecord]:
    """초기 수집 계획을 담은 시드 레코드.

    실제 식별자/URL 은 수집 시점에 채우고, 약관/스크래핑 이슈는 terms_note 에 남긴다.
    여기서는 하드코딩된 고객정보 없이 '계획' 상태의 골격만 제공한다.
    """
    return [
        SourceRecord(
            source_name="슈퍼SOL Google Play 리뷰",
            source_type="review",
            app_name="슈퍼SOL",
            url_or_file="https://play.google.com/store/apps/details?id=<APP_ID>",
            collection_method="google_play_scraper",
            terms_note="공개 리뷰만 수집, 작성자 식별정보 비식별화",
            status="planned",
        ),
        SourceRecord(
            source_name="슈퍼SOL App Store 리뷰",
            source_type="review",
            app_name="슈퍼SOL",
            url_or_file="data/raw/appstore_supersol_reviews.csv",
            collection_method="app_store_manual_csv",
            terms_note="manual csv fallback (스크래핑 제한 대응)",
            status="planned",
        ),
        SourceRecord(
            source_name="경쟁 앱 스토어 설명/소개",
            source_type="app_description",
            app_name="경쟁 금융앱",
            url_or_file="docs/competitors/<app>_description.md",
            collection_method="manual_download",
            terms_note="공개 스토어 설명 텍스트, 직접 열람·정리",
            status="planned",
        ),
        SourceRecord(
            source_name="경쟁 앱 릴리즈 노트",
            source_type="release_note",
            app_name="경쟁 금융앱",
            url_or_file="docs/competitors/<app>_release_notes.md",
            collection_method="manual_download",
            terms_note="공개 업데이트 내역 정리",
            status="planned",
        ),
        SourceRecord(
            source_name="개발/테스트용 샘플 리뷰",
            source_type="sample",
            app_name="슈퍼SOL",
            url_or_file="data/raw/sample_reviews.csv",
            collection_method="sample_generated",
            terms_note="파이프라인 검증용 합성 데이터, 실제 고객정보 아님",
            status="collected",
        ),
    ]


def write_manifest(
    records: list[SourceRecord] | None = None,
    path=MANIFEST_PATH,
) -> "object":
    """매니페스트 CSV 를 생성한다.

    Args:
        records: 기록할 출처 목록. None 이면 seed_records() 사용.
        path:    저장 경로. 기본 outputs/source_manifest.csv

    Returns:
        저장된 파일 경로(Path).
    """
    records = records if records is not None else seed_records()
    for r in records:
        r.validate()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        for r in records:
            writer.writerow(r.as_row())
    return path


def append_to_manifest(
    records: list[SourceRecord],
    path=MANIFEST_PATH,
) -> "object":
    """기존 매니페스트에 출처 레코드를 추가한다 (없으면 헤더 포함 새로 생성).

    수집/생성 모듈이 실행 결과를 누적 기록하는 데 사용한다.
    """
    for r in records:
        r.validate()

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(r.as_row())
    return path


if __name__ == "__main__":
    out = write_manifest()
    print(f"source_manifest 생성 완료: {out}")
    print(f"컬럼: {', '.join(COLUMNS)}")
