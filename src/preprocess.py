"""
리뷰 전처리 모듈 (preprocess.py)

목적:
- 공개 리뷰 원본(data/raw/review_raw.csv)을 분석 가능한 형태로 정제한다.
- 개인정보성 패턴(전화/이메일/계좌/생년월일/인증번호 등)을 [MASKED] 로 치환한다.
- 정제 품질 지표를 outputs/data_quality_summary.json 으로 남긴다.

원칙:
- 원본은 수정하지 않고 결과만 data/processed/ 에 저장한다.
- 마스킹된 리뷰는 답글 자동 생성 시 담당자 검수 대상으로 표시한다.

실행:
    python -m src.preprocess
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd

from src.config import PATHS, settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess")

# ----------------------------------------------------------------------
# 상수 / 경로
# ----------------------------------------------------------------------
RAW_PATH = PATHS.raw_dir / "review_raw.csv"
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
SUMMARY_PATH = PATHS.outputs_dir / "data_quality_summary.json"

REQUIRED_COLUMNS = ["app_name", "store", "rating", "date", "review_text", "version", "source_url"]

MIN_REVIEW_LEN = 5  # clean_text 기준 이보다 짧으면 제거

# ----------------------------------------------------------------------
# 개인정보(PII) 마스킹 패턴
# ----------------------------------------------------------------------
MASK = "[MASKED]"

# 순서 주의: 더 구체적인 패턴(계좌/주민등록/전화)을 먼저 적용한다.
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 이메일
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    # 주민등록번호 (생년월일 6자리 - 뒤 7자리)
    ("rrn", re.compile(r"\b\d{6}\s*-\s*[1-4]\d{6}\b")),
    # 전화번호 (휴대폰/지역번호, 하이픈/공백 허용)
    ("phone", re.compile(r"\b01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}\b")),
    ("phone2", re.compile(r"\b0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b")),
    # 계좌번호 (숫자-숫자-숫자 형태, 10자리 이상 그룹)
    ("account", re.compile(r"\b\d{2,6}[-]\d{2,6}[-]\d{2,7}(?:[-]\d{1,6})?\b")),
    # 생년월일 (YYYY.MM.DD / YYYY-MM-DD / YYYY년 MM월 DD일)
    ("birth", re.compile(r"\b(?:19|20)\d{2}\s*[.\-/년]\s*\d{1,2}\s*[.\-/월]\s*\d{1,2}\s*일?\b")),
    # 카드번호 (4-4-4-4)
    ("card", re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b")),
    # 인증번호 ('인증번호'/'OTP'/'코드' 뒤에 오는 4~8자리 숫자)
    ("otp", re.compile(r"(?:인증\s*번호|인증코드|OTP|otp|코드)\s*[:\-]?\s*\d{4,8}\b")),
    # 길이가 긴 순수 숫자열(11자리 이상) — 계좌/카드/전화 잔여분
    ("longnum", re.compile(r"\b\d{11,}\b")),
]

# ----------------------------------------------------------------------
# 텍스트 정리
# ----------------------------------------------------------------------
# 이모지 및 기타 기호/픽토그램 영역
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF"
    "\U00002190-\U000021FF"
    "\U0000FE00-\U0000FE0F"
    "]",
    flags=re.UNICODE,
)
# 한글/영문/숫자/기본 문장부호 외 특수문자
_SPECIAL_RE = re.compile(r"[^0-9A-Za-z가-힣\s.,!?~%()\[\]]")
_MULTISPACE_RE = re.compile(r"\s+")


def mask_pii(text: str) -> tuple[str, bool]:
    """텍스트에서 개인정보성 패턴을 [MASKED] 로 치환한다.

    Returns:
        (마스킹된 텍스트, 마스킹 발생 여부)
    """
    if not text:
        return text, False
    masked = text
    for _, pat in PII_PATTERNS:
        masked = pat.sub(MASK, masked)
    return masked, (masked != text)


def clean_text(text: str) -> str:
    """이모지/특수문자/반복 공백을 정리한다. [MASKED] 토큰은 보존한다."""
    if not text:
        return ""
    # [MASKED] 보호: 임시 토큰으로 치환 후 복원
    placeholder = "\x00MASK\x00"
    t = text.replace(MASK, placeholder)
    t = _EMOJI_RE.sub(" ", t)
    t = _SPECIAL_RE.sub(" ", t)
    t = _MULTISPACE_RE.sub(" ", t).strip()
    t = t.replace(placeholder, MASK)
    return t


# ----------------------------------------------------------------------
# 메인 전처리
# ----------------------------------------------------------------------
def preprocess(
    raw_path=RAW_PATH,
    clean_path=CLEAN_PATH,
    summary_path=SUMMARY_PATH,
    min_len: int = MIN_REVIEW_LEN,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """리뷰 원본을 정제하고 품질 리포트를 생성한다."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"입력 파일이 없습니다: {raw_path} (먼저 python -m src.collect_reviews 실행)"
        )

    df = pd.read_csv(raw_path)
    n_input = len(df)
    logger.info("입력 로드: %s (%d건)", raw_path, n_input)

    # 1) 필수 컬럼 검사
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    # 2) review_text 결측 처리
    n_null_text = int(df["review_text"].isna().sum() +
                      (df["review_text"].fillna("").astype(str).str.strip() == "").sum())
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["review_text"].str.strip() != ""].copy()

    # 3) 중복 처리 (앱+스토어+리뷰텍스트 기준)
    n_before_dup = len(df)
    df = df.drop_duplicates(subset=["app_name", "store", "review_text"]).copy()
    n_dup = n_before_dup - len(df)

    # 4) PII 마스킹
    masked_flags = []
    masked_texts = []
    for txt in df["review_text"]:
        m_txt, flag = mask_pii(txt)
        masked_texts.append(m_txt)
        masked_flags.append(flag)
    df["review_text"] = masked_texts
    df["has_pii_masked"] = masked_flags
    n_masked = int(sum(masked_flags))

    # 5) clean_text 생성
    df["clean_text"] = df["review_text"].apply(clean_text)

    # 6) 너무 짧은 리뷰 제거 (clean_text 기준)
    n_before_short = len(df)
    df = df[df["clean_text"].str.len() >= min_len].copy()
    n_short = n_before_short - len(df)

    # 7) date 표준화
    parsed = pd.to_datetime(df["date"], errors="coerce")
    n_date_fail = int(parsed.isna().sum())
    if n_date_fail:
        logger.warning("date 파싱 실패 %d건 (NaT 처리)", n_date_fail)
    df["date"] = parsed

    # 8) 리뷰 길이 분포 + 이상치 후보 (IQR)
    df["review_len"] = df["clean_text"].str.len()
    q1 = float(df["review_len"].quantile(0.25))
    q3 = float(df["review_len"].quantile(0.75))
    iqr = q3 - q1
    low_fence = max(0, q1 - 1.5 * iqr)
    high_fence = q3 + 1.5 * iqr
    df["len_outlier"] = (df["review_len"] < low_fence) | (df["review_len"] > high_fence)
    n_outlier = int(df["len_outlier"].sum())

    # 9) 답글 자동 생성 검수 대상: [MASKED] 포함 시 needs_human_review
    df["needs_human_review"] = df["has_pii_masked"]
    n_needs_review = int(df["needs_human_review"].sum())

    n_output = len(df)

    # 10) 품질 요약
    summary = {
        "input_rows": n_input,
        "output_rows": n_output,
        "removed_rows": n_input - n_output,
        "null_text_rows": n_null_text,
        "duplicate_rows_removed": n_dup,
        "too_short_rows_removed": n_short,
        "pii_masked_rows": n_masked,
        "needs_human_review_rows": n_needs_review,
        "date_parse_failed_rows": n_date_fail,
        "rates": {
            "null_text_rate": round(n_null_text / n_input, 4) if n_input else 0,
            "duplicate_rate": round(n_dup / n_input, 4) if n_input else 0,
            "date_parse_fail_rate": round(n_date_fail / n_output, 4) if n_output else 0,
            "pii_masked_rate": round(n_masked / n_output, 4) if n_output else 0,
            "outlier_rate": round(n_outlier / n_output, 4) if n_output else 0,
        },
        "length_distribution": {
            "min": int(df["review_len"].min()) if n_output else 0,
            "q1": q1,
            "median": float(df["review_len"].median()) if n_output else 0,
            "q3": q3,
            "max": int(df["review_len"].max()) if n_output else 0,
            "iqr": iqr,
            "low_fence": low_fence,
            "high_fence": high_fence,
        },
        "outlier_candidates": n_outlier,
        "by_store": df["store"].value_counts().to_dict(),
        "by_app": df["app_name"].value_counts().to_dict(),
    }

    if save:
        # CSV 저장 (date 는 ISO 문자열로)
        out = df.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(clean_path, index=False, encoding="utf-8-sig")
        logger.info("정제 결과 저장: %s (%d건)", clean_path, len(out))

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("품질 요약 저장: %s", summary_path)

    return df, summary


if __name__ == "__main__":
    _, summary = preprocess()
    print("\n=== 데이터 품질 요약 ===")
    print(f"입력 → 출력 : {summary['input_rows']} → {summary['output_rows']}건")
    print(f"결측 제거    : {summary['null_text_rows']}건")
    print(f"중복 제거    : {summary['duplicate_rows_removed']}건")
    print(f"짧은 리뷰 제거: {summary['too_short_rows_removed']}건")
    print(f"PII 마스킹   : {summary['pii_masked_rows']}건 (검수 대상)")
    print(f"날짜 파싱 실패: {summary['date_parse_failed_rows']}건")
    print(f"길이 이상치  : {summary['outlier_candidates']}건")
