"""
앱 리뷰 수집 모듈 (collect_reviews.py)

설계 원칙:
- 실제 수집(Google Play / App Store)이 실패해도 **데모가 가능하도록** 항상
  샘플 리뷰로 fallback 한다 (파이프라인이 멈추지 않음).
- 모든 결과는 동일 스키마로 통합되어 `data/raw/review_raw.csv` 에 저장된다.
- 수집/생성 결과는 `outputs/source_manifest.csv` 에 근거로 기록된다.
- 실제 고객정보·내부정보는 포함하지 않으며, 샘플은 합성 데이터다.

CSV 스키마(필수 컬럼):
    app_name, store, rating, date, review_text, version, source_url

store 값: google_play | app_store | sample

사용 예:
    from src.collect_reviews import collect_all
    df = collect_all(google_play_app_id=None, n_per_store=300)   # 키 없으면 샘플 fallback
"""
from __future__ import annotations

import logging
import random
from datetime import date, timedelta

import pandas as pd

from src.config import PATHS, settings
from src.source_manifest import SourceRecord, append_to_manifest

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_reviews")

# ----------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------
COLUMNS = ("app_name", "store", "rating", "date", "review_text", "version", "source_url")
STORES = ("google_play", "app_store", "sample")

OUR_APP = "슈퍼SOL"

# 경쟁 앱 실제 ID (2026-06 기준, Google Play 검색 + Apple iTunes 검색 API 로 검증).
# 3~5개로 조정 가능. google_play=패키지명, app_store=숫자 ID.
COMPETITOR_APPS: dict[str, dict[str, str]] = {
    "KB스타뱅킹":   {"google_play": "com.kbstar.kbbank",        "app_store": "373742138"},
    "우리WON뱅킹":  {"google_play": "com.wooribank.smart.npib", "app_store": "1470181651"},
    "하나원큐":     {"google_play": "com.hanabank.oqf",         "app_store": "6743190232"},
    "카카오뱅크":   {"google_play": "com.kakaobank.channel",    "app_store": "1258016944"},
    "토스":         {"google_play": "viva.republica.toss",      "app_store": "839333328"},
}
DEFAULT_COMPETITORS = list(COMPETITOR_APPS.keys())

RAW_PATH = PATHS.raw_dir / "review_raw.csv"

# 재현성을 위한 시드 (샘플 생성 결과 고정)
_SEED = 20260620


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(COLUMNS))


# ----------------------------------------------------------------------
# 1) Google Play 수집
# ----------------------------------------------------------------------
def collect_google_play_reviews(app_id: str, app_name: str, n: int = 300) -> pd.DataFrame:
    """google-play-scraper 로 공개 리뷰를 수집한다.

    Args:
        app_id:   Google Play 패키지명 (예: com.shinhan.sbanking)
        app_name: 표시용 앱 이름 (예: 슈퍼SOL)
        n:        수집할 리뷰 수

    Returns:
        표준 스키마 DataFrame. 실패 시 빈 DataFrame 을 반환한다(예외를 올리지 않음).
    """
    if not app_id:
        logger.warning("[google_play] app_id 가 비어 있어 수집을 건너뜁니다.")
        return _empty_df()

    try:
        from google_play_scraper import Sort, reviews  # 지연 import
    except ImportError:
        logger.warning(
            "[google_play] google-play-scraper 미설치. 수집을 건너뜁니다. "
            "(pip install google-play-scraper)"
        )
        return _empty_df()

    try:
        result, _ = reviews(
            app_id,
            lang="ko",
            country="kr",
            sort=Sort.NEWEST,
            count=n,
        )
    except Exception as exc:  # 네트워크/차단/앱ID 오류 등
        logger.warning("[google_play] 수집 실패 (%r). 빈 결과 반환.", exc)
        return _empty_df()

    rows = []
    for r in result:
        rows.append(
            {
                "app_name": app_name,
                "store": "google_play",
                "rating": r.get("score"),
                "date": str(r.get("at", ""))[:10],
                "review_text": (r.get("content") or "").replace("\n", " ").strip(),
                "version": r.get("reviewCreatedVersion") or "",
                "source_url": f"https://play.google.com/store/apps/details?id={app_id}",
            }
        )
    logger.info("[google_play] %s 리뷰 %d건 수집", app_name, len(rows))
    return pd.DataFrame(rows, columns=list(COLUMNS))


# ----------------------------------------------------------------------
# 2) App Store 수집 (선택 기능 / 실패 가능)
# ----------------------------------------------------------------------
def collect_app_store_reviews(
    app_id: str | None,
    app_name: str,
    n: int = 300,
    manual_csv=None,
) -> pd.DataFrame:
    """App Store 리뷰를 수집한다. 자동 수집 실패 가능성이 높은 **선택 기능**.

    동작 순서:
        1) manual_csv 가 주어지고 존재하면 → 수동 CSV 사용 (우선)
        2) Apple 공개 iTunes RSS 리뷰 피드 수집 (requests 기반)
        3) 실패하면 → 빈 DataFrame 반환 (상위에서 샘플 fallback)

    참고: app-store-scraper 는 Python 3.13 비호환(requests==2.23 강제)이라
    사용하지 않고 공개 RSS 피드로 대체했다.
    """
    country = settings.app_store_country

    # 1) 수동 CSV 우선
    if manual_csv is not None:
        from pathlib import Path

        p = Path(manual_csv)
        if p.exists():
            try:
                df = pd.read_csv(p)
                df = _normalize_manual_csv(df, app_name)
                logger.info("[app_store] 수동 CSV 사용: %s (%d건)", p, len(df))
                return df
            except Exception as exc:
                logger.warning("[app_store] 수동 CSV 로드 실패 (%r).", exc)
        else:
            logger.warning("[app_store] 수동 CSV 경로 없음: %s", p)

    if not app_id:
        logger.warning("[app_store] app_id 없음 + 수동 CSV 없음 → 자동 수집 생략.")
        return _empty_df()

    # 2) iTunes RSS 공개 피드 (별도 라이브러리 불필요, Py3.13 호환)
    df = _collect_app_store_rss(app_id, app_name, country, n)
    if not df.empty:
        return df

    logger.warning("[app_store] RSS 수집 실패 → 수동 CSV/샘플로 전환 필요.")
    return _empty_df()


def _collect_app_store_rss(
    app_id: str, app_name: str, country: str, n: int = 300, max_pages: int = 10
) -> pd.DataFrame:
    """Apple 공개 iTunes RSS '고객 리뷰' 피드에서 리뷰를 수집한다.

    엔드포인트(공개):
        https://itunes.apple.com/{country}/rss/customerreviews/page={p}/id={app_id}/sortby=mostrecent/json
    페이지당 최대 50건, 최대 10페이지(약 500건). 실패 시 빈 DF 반환.
    """
    try:
        import requests  # 지연 import
    except ImportError:
        logger.warning("[app_store] requests 미설치로 RSS 수집 불가.")
        return _empty_df()

    rows: list[dict] = []
    base = "https://itunes.apple.com/{c}/rss/customerreviews/page={p}/id={id}/sortby=mostrecent/json"
    app_url = f"https://apps.apple.com/{country}/app/id{app_id}"
    headers = {"User-Agent": "Mozilla/5.0 (review-collector; research)"}

    for page in range(1, max_pages + 1):
        if len(rows) >= n:
            break
        url = base.format(c=country, p=page, id=app_id)
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            entries = resp.json().get("feed", {}).get("entry", [])
        except Exception as exc:
            logger.warning("[app_store] RSS page %d 수집 실패 (%r).", page, exc)
            break

        # 1페이지의 첫 entry 는 앱 메타데이터인 경우가 있어 'im:rating' 유무로 판별
        for e in entries:
            if "im:rating" not in e:
                continue
            rows.append(
                {
                    "app_name": app_name,
                    "store": "app_store",
                    "rating": int(e["im:rating"]["label"]),
                    "date": (e.get("updated", {}).get("label", "") or "")[:10],
                    "review_text": (
                        (e.get("title", {}).get("label", "") + " " +
                         e.get("content", {}).get("label", ""))
                        .replace("\n", " ").strip()
                    ),
                    "version": e.get("im:version", {}).get("label", ""),
                    "source_url": app_url,
                }
            )

    if rows:
        logger.info("[app_store] iTunes RSS %s 리뷰 %d건 수집", app_name, len(rows))
        return pd.DataFrame(rows[:n], columns=list(COLUMNS))
    logger.warning("[app_store] iTunes RSS 수집 결과 없음 (id=%s, country=%s).", app_id, country)
    return _empty_df()


def _normalize_manual_csv(df: pd.DataFrame, app_name: str) -> pd.DataFrame:
    """수동 다운로드 CSV 를 표준 스키마로 정규화한다."""
    out = pd.DataFrame()
    out["app_name"] = df.get("app_name", app_name)
    out["store"] = "app_store"
    out["rating"] = df.get("rating")
    out["date"] = df.get("date", "").astype(str).str[:10]
    out["review_text"] = df.get("review_text", df.get("review", "")).astype(str)
    out["version"] = df.get("version", "")
    out["source_url"] = df.get("source_url", "")
    return out[list(COLUMNS)]


# ----------------------------------------------------------------------
# 3) 샘플 리뷰 생성 (fallback / 데모)
# ----------------------------------------------------------------------
# (이슈 카테고리, 대표 평점, 리뷰 텍스트 후보)
_SAMPLE_TEMPLATES: list[tuple[str, int, list[str]]] = [
    ("로그인", 2, [
        "로그인이 자꾸 안 돼요. 비밀번호 맞는데 오류만 떠요.",
        "앱 켜면 로그인 화면에서 멈춰서 진입이 안 됩니다.",
        "지문 로그인 등록했는데 매번 풀려서 다시 해야 해요.",
    ]),
    ("인증", 2, [
        "본인 인증 단계에서 계속 실패합니다. 너무 번거로워요.",
        "공동인증서 등록이 안 돼서 진행이 막혀요.",
        "OTP 인증이 자꾸 시간 초과돼서 처음부터 다시 해야 합니다.",
    ]),
    ("속도", 2, [
        "앱이 너무 느려요. 화면 전환할 때마다 한참 기다립니다.",
        "실행 속도가 느리고 자주 버벅거려요.",
        "조회할 때 로딩이 너무 길어요. 개선 부탁드립니다.",
    ]),
    ("혜택", 4, [
        "혜택이 다양해서 좋아요. 적립 쏠쏠합니다.",
        "이벤트 혜택 챙기는 재미가 있네요.",
        "포인트 혜택은 좋은데 조건이 좀 복잡해요.",
    ]),
    ("송금", 3, [
        "송금은 편한데 가끔 지연될 때가 있어요.",
        "이체 한도 설정이 헷갈려요. 안내가 더 필요합니다.",
        "송금 즐겨찾기 기능이 편리하네요.",
    ]),
    ("투자", 3, [
        "투자 메뉴가 한눈에 안 들어와요. 정리가 필요해요.",
        "펀드 가입 과정이 직관적이라 좋았습니다.",
        "투자 수익률 화면이 자주 갱신이 안 돼요.",
    ]),
    ("UX", 3, [
        "메뉴가 너무 많아서 원하는 기능 찾기 어려워요.",
        "디자인은 깔끔한데 동선이 복잡합니다.",
        "글씨가 작고 버튼 위치가 불편해요.",
    ]),
    ("오류", 1, [
        "결제하다가 앱이 튕겨서 두 번 결제될 뻔했어요.",
        "업데이트 후 자꾸 강제 종료됩니다.",
        "특정 화면에서 계속 오류 코드가 떠요.",
    ]),
    ("보안우려", 2, [
        "보안이 걱정돼요. 알 수 없는 접속 알림이 왔어요.",
        "비밀번호 변경 안내가 자주 와서 불안합니다.",
        "개인정보 처리 방식이 명확하지 않은 것 같아요.",
    ]),
    ("문의성", 3, [
        "이체 수수료는 어떻게 면제받나요? 안내 부탁드립니다.",
        "카드 재발급은 앱에서 어디서 하나요?",
        "해외 결제 등록 방법이 궁금합니다.",
    ]),
    ("긍정", 5, [
        "전반적으로 편리하고 만족스러워요. 잘 쓰고 있습니다.",
        "하나의 앱으로 다 되니까 정말 편해요.",
        "업데이트되면서 훨씬 쓰기 좋아졌어요. 추천합니다.",
    ]),
]

_VERSIONS = ["1.0.0", "1.1.0", "1.1.2", "1.2.0", "2.0.0"]


def generate_sample_reviews(
    n: int = 1000,
    our_app: str = OUR_APP,
    competitors: list[str] | None = None,
    seed: int = _SEED,
) -> pd.DataFrame:
    """다양한 이슈를 포함한 합성 리뷰를 n건 생성한다 (실제 고객정보 아님).

    이슈 카테고리: 로그인/인증/속도/혜택/송금/투자/UX/오류/보안우려/문의성/긍정.

    Args:
        n:           생성 건수 (1,000건 이상 권장)
        our_app:     우리 앱 이름
        competitors: 경쟁 앱 목록 (3~5개)
        seed:        재현성을 위한 시드
    """
    competitors = competitors if competitors is not None else DEFAULT_COMPETITORS
    if not (3 <= len(competitors) <= 5):
        logger.warning("경쟁 앱은 3~5개 권장입니다 (현재 %d개).", len(competitors))

    rng = random.Random(seed)
    apps = [our_app] + competitors
    base_date = date(2025, 10, 1)  # 출시 초기 가정 시작일

    rows = []
    for _ in range(n):
        app_name = rng.choice(apps)
        category, base_rating, texts = rng.choice(_SAMPLE_TEMPLATES)
        text = rng.choice(texts)
        # 평점에 약간의 변동 부여 (1~5 범위 유지)
        rating = min(5, max(1, base_rating + rng.choice([-1, 0, 0, 1])))
        d = base_date + timedelta(days=rng.randint(0, 250))
        rows.append(
            {
                "app_name": app_name,
                "store": "sample",
                "rating": rating,
                "date": d.isoformat(),
                "review_text": f"[{category}] {text}",
                "version": rng.choice(_VERSIONS),
                "source_url": "sample_generated",
            }
        )

    logger.info("샘플 리뷰 %d건 생성 (앱 %d개, 카테고리 %d종)",
                len(rows), len(apps), len(_SAMPLE_TEMPLATES))
    return pd.DataFrame(rows, columns=list(COLUMNS))


# ----------------------------------------------------------------------
# 4) 통합 수집 오케스트레이션
# ----------------------------------------------------------------------
def _collect_one_app(
    app_name: str,
    gp_id: str | None,
    as_id: str | None,
    n_per_store: int,
    app_store_manual_csv=None,
) -> tuple[list[pd.DataFrame], list[SourceRecord]]:
    """앱 1개에 대해 Google Play + App Store 를 수집하고 매니페스트 레코드를 만든다."""
    frames: list[pd.DataFrame] = []
    manifest: list[SourceRecord] = []

    # Google Play
    if gp_id:
        gp = collect_google_play_reviews(gp_id, app_name, n=n_per_store)
        if not gp.empty:
            frames.append(gp)
            manifest.append(SourceRecord(
                source_name=f"{app_name} Google Play 리뷰",
                source_type="review", app_name=app_name,
                url_or_file=f"https://play.google.com/store/apps/details?id={gp_id}",
                collection_method="google_play_scraper",
                terms_note="공개 리뷰, 작성자 비식별화", status="collected",
            ))

    # App Store
    if as_id or app_store_manual_csv:
        aps = collect_app_store_reviews(as_id, app_name, n=n_per_store,
                                        manual_csv=app_store_manual_csv)
        if not aps.empty:
            frames.append(aps)
            method = "app_store_manual_csv" if app_store_manual_csv else "manual_download"
            manifest.append(SourceRecord(
                source_name=f"{app_name} App Store 리뷰",
                source_type="review", app_name=app_name,
                url_or_file=str(app_store_manual_csv or f"appstore id{as_id}"),
                collection_method=method,
                terms_note="manual csv fallback (자동 수집 제한 대응)"
                if app_store_manual_csv else "공개 RSS 리뷰 자동 수집",
                status="collected",
            ))

    return frames, manifest


def collect_all(
    google_play_app_id: str | None = None,
    app_store_app_id: str | None = None,
    app_store_manual_csv=None,
    our_app: str = OUR_APP,
    competitors: dict[str, dict[str, str]] | None = None,
    n_per_store: int = 300,
    sample_n: int = 1000,
    save: bool = True,
) -> pd.DataFrame:
    """우리 앱 + 경쟁 앱을 수집하고, 모두 실패하면 샘플로 보강하여 통합 저장한다.

    Args:
        competitors: {앱이름: {"google_play": pkg, "app_store": id}} 매핑.
                     None 이면 COMPETITOR_APPS(검증된 5개) 사용. {} 이면 우리 앱만.
    동작:
        - 각 앱마다 Google Play / App Store(RSS) 수집
        - 전부 비어 있으면 경고 후 generate_sample_reviews() 실행
        - 결과를 data/raw/review_raw.csv 로 저장, source_manifest.csv 에 기록
    """
    gp_id = google_play_app_id or settings.google_play_app_id
    as_id = app_store_app_id or settings.app_store_app_id
    competitors = competitors if competitors is not None else COMPETITOR_APPS

    frames: list[pd.DataFrame] = []
    manifest: list[SourceRecord] = []

    # 1) 우리 앱
    logger.info("=== 수집 시작: %s (우리 앱) ===", our_app)
    f, m = _collect_one_app(our_app, gp_id, as_id, n_per_store, app_store_manual_csv)
    frames += f
    manifest += m

    # 2) 경쟁 앱
    for name, ids in competitors.items():
        logger.info("=== 수집 시작: %s (경쟁 앱) ===", name)
        f, m = _collect_one_app(
            name, ids.get("google_play"), ids.get("app_store"), n_per_store
        )
        frames += f
        manifest += m

    # 3) Fallback: 실제 수집이 전부 비었으면 샘플 생성
    if not frames:
        logger.warning(
            "⚠️ 실제 수집 결과가 없습니다. 데모용 샘플 리뷰로 전환합니다 "
            "(generate_sample_reviews)."
        )
        sample = generate_sample_reviews(n=sample_n, our_app=our_app,
                                         competitors=list(competitors.keys()))
        frames.append(sample)
        manifest.append(SourceRecord(
            source_name="데모용 샘플 리뷰",
            source_type="sample", app_name=our_app,
            url_or_file=str(RAW_PATH),
            collection_method="sample_generated",
            terms_note=f"파이프라인/데모용 합성 데이터 {len(sample)}건, 실제 고객정보 아님",
            status="collected",
        ))

    df = pd.concat(frames, ignore_index=True)

    if save:
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
        logger.info("통합 리뷰 저장: %s (%d건)", RAW_PATH, len(df))
        append_to_manifest(manifest)

    return df


if __name__ == "__main__":
    out = collect_all()
    print(f"\n수집 완료: 총 {len(out)}건")
    print(f"\n[store 분포]\n{out['store'].value_counts().to_string()}")
    print(f"\n[앱 분포]\n{out['app_name'].value_counts().to_string()}")
    print(f"\n저장 경로: {RAW_PATH}")
