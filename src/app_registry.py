"""
리뷰 수집 대상 앱 레지스트리 (app_registry.py)

- config/apps.yaml 을 읽고/쓰는 헬퍼.
- 선택한 앱들을 Google Play / App Store(RSS) 에서 수집해 표준 스키마 DataFrame 으로
  반환한다(저장은 호출 측에서 결정 → 미리보기 후 적용/병합/버리기 흐름 지원).

UI(Streamlit '데이터 수집' 탭)와 파이프라인 양쪽에서 재사용한다.
"""
from __future__ import annotations

import logging

import pandas as pd
import yaml

from src.config import PATHS, settings
from src.collect_reviews import (
    COLUMNS,
    collect_app_store_reviews,
    collect_google_play_reviews,
)

logger = logging.getLogger("app_registry")

APPS_PATH = PATHS.config_dir / "apps.yaml"

DEFAULT_REGISTRY = {
    "our_app": "슈퍼SOL",
    "apps": [],
    "defaults": {"stores": ["google_play", "app_store"], "n_per_store": 300},
}


# ----------------------------------------------------------------------
# 레지스트리 입출력
# ----------------------------------------------------------------------
def load_registry(path=APPS_PATH) -> dict:
    """apps.yaml 을 로드한다(없으면 기본 구조)."""
    if not path.exists():
        return dict(DEFAULT_REGISTRY)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("our_app", "슈퍼SOL")
    data.setdefault("apps", [])
    data.setdefault("defaults", dict(DEFAULT_REGISTRY["defaults"]))
    return data


def save_registry(registry: dict, path=APPS_PATH) -> None:
    """레지스트리를 apps.yaml 로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 정규화: app_store ID 는 문자열로
    for a in registry.get("apps", []):
        if a.get("app_store") is not None:
            a["app_store"] = str(a["app_store"]).strip()
        a["google_play"] = str(a.get("google_play", "")).strip()
        a["name"] = str(a.get("name", "")).strip()
        a["is_our_app"] = bool(a.get("is_our_app", False))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, allow_unicode=True, sort_keys=False)
    logger.info("앱 레지스트리 저장: %s (%d개)", path, len(registry.get("apps", [])))


def apps_dataframe(registry: dict | None = None) -> pd.DataFrame:
    """레지스트리를 편집용 DataFrame 으로 변환(화면 data_editor 용)."""
    registry = registry or load_registry()
    rows = registry.get("apps", [])
    if not rows:
        return pd.DataFrame(columns=["name", "google_play", "app_store", "is_our_app"])
    return pd.DataFrame(rows)[["name", "google_play", "app_store", "is_our_app"]]


def registry_from_dataframe(df: pd.DataFrame, our_app: str, defaults: dict) -> dict:
    """편집된 DataFrame 을 레지스트리 dict 로 변환."""
    apps = []
    for _, r in df.iterrows():
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        apps.append({
            "name": name,
            "google_play": str(r.get("google_play", "") or "").strip(),
            "app_store": str(r.get("app_store", "") or "").strip(),
            "is_our_app": bool(r.get("is_our_app", False)),
        })
    return {"our_app": our_app, "apps": apps, "defaults": defaults}


def get_app(registry: dict, name: str) -> dict | None:
    for a in registry.get("apps", []):
        if a.get("name") == name:
            return a
    return None


# ----------------------------------------------------------------------
# 수집 (저장 안 함 → 미리보기용 DataFrame 반환)
# ----------------------------------------------------------------------
def collect_apps(
    selected: list[str],
    stores: list[str],
    n_per_store: int = 300,
    registry: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """선택한 앱들을 지정 스토어에서 수집해 통합 DataFrame + 수집 리포트를 반환한다.

    Returns:
        (df, report) — df 는 표준 스키마, report 는 앱·스토어별 건수/상태 목록.
    """
    registry = registry or load_registry()
    frames: list[pd.DataFrame] = []
    report: list[dict] = []

    for name in selected:
        app = get_app(registry, name)
        if not app:
            report.append({"app": name, "store": "-", "count": 0, "status": "미등록"})
            continue

        if "google_play" in stores and app.get("google_play"):
            gp = collect_google_play_reviews(app["google_play"], name, n=n_per_store)
            frames.append(gp)
            report.append({"app": name, "store": "google_play",
                           "count": int(len(gp)),
                           "status": "수집" if len(gp) else "결과 없음"})

        if "app_store" in stores and app.get("app_store"):
            aps = collect_app_store_reviews(app["app_store"], name, n=n_per_store)
            frames.append(aps)
            report.append({"app": name, "store": "app_store",
                           "count": int(len(aps)),
                           "status": "수집" if len(aps) else "결과 없음(수동 입력 권장)"})

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=list(COLUMNS))
    return df, report


# ----------------------------------------------------------------------
# 수동 입력 정규화 / 병합
# ----------------------------------------------------------------------
def normalize_manual(df: pd.DataFrame, default_app: str = "") -> pd.DataFrame:
    """수동 입력(CSV/편집) DataFrame 을 표준 스키마로 맞춘다(부족 컬럼은 채움)."""
    out = pd.DataFrame()
    for col in COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = ""
    if (out["app_name"].astype(str).str.strip() == "").all() and default_app:
        out["app_name"] = default_app
    if (out["store"].astype(str).str.strip() == "").all():
        out["store"] = "sample"  # 출처 미상 수동 입력
    return out[list(COLUMNS)]


def merge_reviews(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """기존 + 신규 리뷰를 표준 스키마로 병합(중복 제거)."""
    cols = list(COLUMNS)
    base = base.reindex(columns=cols) if not base.empty else pd.DataFrame(columns=cols)
    new = new.reindex(columns=cols)
    merged = pd.concat([base, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["app_name", "store", "review_text"])
    return merged.reset_index(drop=True)
