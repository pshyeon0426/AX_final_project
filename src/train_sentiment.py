"""
감성 분류 baseline 모델 (train_sentiment.py)

목적:
- 리뷰를 긍정/중립/부정으로 분류하는 baseline 2종을 학습·비교한다.
    Baseline 1: DummyClassifier (최빈 클래스 예측) — 성능 하한선
    Baseline 2: TF-IDF(unigram) + LogisticRegression(기본값)

⚠️ 약지도 라벨 주의:
    라벨은 별점에서 규칙으로 부여한 '약지도(weak supervision)' 라벨이다.
        1~2점 = negative, 3점 = neutral, 4~5점 = positive
    이는 작성자의 실제 감정 정답(human label)이 아니라 근사치다.
    별점과 본문 감정이 불일치하는 경우(예: 5점인데 불만, 1점인데 칭찬)가 있으므로
    성능 수치는 '별점 라벨 재현도'로 해석해야 하며, 향후 일부 수기 라벨로 검증이 필요하다.

실행:
    python -m src.train_sentiment
"""
from __future__ import annotations

import json
import logging

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import PATHS, settings
from src.tokenizer import get_tokenizer

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_sentiment")

# ----------------------------------------------------------------------
# 경로 / 상수
# ----------------------------------------------------------------------
CLEAN_PATH = PATHS.processed_dir / "review_clean.csv"
COMPARISON_PATH = PATHS.outputs_dir / "model_comparison.csv"
REPORT_PATH = PATHS.outputs_dir / "model_report.md"
METRICS_PATH = PATHS.outputs_dir / "metrics.json"
MODEL_PATH = PATHS.models_dir / "sentiment_model.joblib"

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 모듈 레벨 토크나이저 (지연 초기화).
# Kiwi 객체는 pickle 이 안 되므로, TfidfVectorizer 에는 인스턴스 메서드 대신
# 모듈 레벨 함수 tokenize() 를 넘긴다. 함수는 import 경로로 직렬화되고,
# Kiwi 객체는 전역에 보관되어 모델 pickle 에 포함되지 않는다.
_TOKENIZER = None


def tokenize(text: str) -> list[str]:
    """TF-IDF 용 토큰화 함수 (pickle 가능). 내부적으로 전역 토크나이저 사용."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = get_tokenizer()
    return _TOKENIZER.tokenize(text)


def make_weak_label(rating: int) -> str:
    """별점 기반 '약지도' 라벨 생성 (실제 감정 정답 아님).

    1~2점 = negative, 3점 = neutral, 4~5점 = positive
    """
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def load_dataset(path=CLEAN_PATH) -> pd.DataFrame:
    """정제 리뷰를 로드하고 약지도 라벨을 부여한다."""
    if not path.exists():
        raise FileNotFoundError(
            f"정제 데이터가 없습니다: {path} (먼저 python -m src.preprocess 실행)"
        )
    df = pd.read_csv(path)
    # 분석 텍스트는 clean_text 사용, 결측/공백 제거
    df["text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)].copy()
    # ⚠️ 약지도 라벨 (별점 규칙 기반, 실제 감정 정답 아님)
    df["label"] = df["rating"].astype(int).apply(make_weak_label)
    return df


def _eval(name: str, y_true, y_pred) -> dict:
    """공통 평가 지표 계산."""
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def train(save: bool = True) -> dict:
    """baseline 2종을 학습·평가하고 결과를 저장한다."""
    df = load_dataset()
    logger.info("데이터셋: %d건", len(df))
    logger.info("라벨 분포(약지도): %s", df["label"].value_counts().to_dict())

    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("train=%d / test=%d", len(X_train), len(X_test))

    results: list[dict] = []

    # --- Baseline 1: DummyClassifier (최빈 클래스) ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    res_dummy = _eval("Baseline1_Dummy(most_frequent)", y_test, dummy.predict(X_test))
    results.append(res_dummy)
    logger.info("[Dummy] f1_macro=%.4f acc=%.4f", res_dummy["f1_macro"], res_dummy["accuracy"])

    # --- Baseline 2: TF-IDF(unigram) + LogisticRegression ---
    logger.info("토크나이저 backend: %s", get_tokenizer().backend)
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize,     # 모듈 레벨 함수 (pickle 가능)
            token_pattern=None,     # 커스텀 tokenizer 사용 시 경고 방지
            ngram_range=(1, 1),     # unigram
            min_df=2,
        )),
        ("clf", LogisticRegression(max_iter=1000)),  # 기본값
    ])
    lr_pipe.fit(X_train, y_train)
    res_lr = _eval("Baseline2_TFIDF+LogReg", y_test, lr_pipe.predict(X_test))
    results.append(res_lr)
    logger.info("[TFIDF+LR] f1_macro=%.4f acc=%.4f", res_lr["f1_macro"], res_lr["accuracy"])

    if save:
        _save_outputs(results, df, len(X_train), len(X_test))
        PATHS.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr_pipe, MODEL_PATH)
        logger.info("모델 저장: %s", MODEL_PATH)

    return {"results": results, "label_dist": df["label"].value_counts().to_dict()}


def _save_outputs(results, df, n_train, n_test) -> None:
    """비교 CSV / 리포트 MD / metrics.json 저장."""
    # 1) model_comparison.csv (confusion matrix 제외 평탄화)
    rows = [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results]
    comp = pd.DataFrame(rows)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    logger.info("비교표 저장: %s", COMPARISON_PATH)

    # 2) model_report.md
    best = max(results, key=lambda r: r["f1_macro"])
    label_dist = df["label"].value_counts().to_dict()
    lines = [
        "# 감성 분류 Baseline 모델 리포트",
        "",
        "## ⚠️ 라벨 정의 (약지도 / weak supervision)",
        "",
        "라벨은 **별점 기반 규칙**으로 부여한 약지도 라벨이며, **작성자의 실제 감정 "
        "정답이 아니다.**",
        "",
        "| 별점 | 라벨 |",
        "|------|------|",
        "| 1~2점 | negative |",
        "| 3점 | neutral |",
        "| 4~5점 | positive |",
        "",
        "> 별점과 본문 감정이 불일치할 수 있으므로(예: 고별점+불만), 아래 성능은 "
        "'별점 라벨 재현도'로 해석하고 향후 수기 라벨로 검증해야 한다.",
        "",
        "## 데이터",
        "",
        f"- 총 {len(df)}건 (train {n_train} / test {n_test})",
        f"- 라벨 분포: {label_dist}",
        "",
        "## 모델 비교",
        "",
        "| 모델 | accuracy | precision_macro | recall_macro | f1_macro |",
        "|------|----------|-----------------|--------------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision_macro']:.4f} "
            f"| {r['recall_macro']:.4f} | {r['f1_macro']:.4f} |"
        )

    lines += ["", "## Confusion Matrix (행=실제, 열=예측, 순서: negative/neutral/positive)", ""]
    for r in results:
        lines.append(f"### {r['model']}")
        lines.append("")
        lines.append("| 실제\\예측 | negative | neutral | positive |")
        lines.append("|-----------|----------|---------|----------|")
        for i, lab in enumerate(LABELS):
            cm = r["confusion_matrix"][i]
            lines.append(f"| {lab} | {cm[0]} | {cm[1]} | {cm[2]} |")
        lines.append("")

    lines += [
        "## 결론",
        "",
        f"- 최고 성능: **{best['model']}** (f1_macro={best['f1_macro']:.4f})",
        "- Dummy 대비 향상폭이 baseline 의 학습 효과를 의미한다.",
        "- 목표(예: macro F1 ≥ 0.75) 대비 격차는 향후 n-gram 확장, 클래스 불균형 "
        "보정, 임베딩 기반 모델로 개선 가능하다.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("리포트 저장: %s", REPORT_PATH)

    # 3) metrics.json 병합
    existing = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            logger.warning("metrics.json 로드 실패 (%r). 새로 생성.", exc)
    existing["sentiment_baseline"] = {
        "label_type": "weak_supervision_from_rating",
        "models": {
            r["model"]: {
                "accuracy": r["accuracy"],
                "precision_macro": r["precision_macro"],
                "recall_macro": r["recall_macro"],
                "f1_macro": r["f1_macro"],
            } for r in results
        },
        "best_model": best["model"],
        "best_f1_macro": best["f1_macro"],
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("metrics.json 병합 저장: %s", METRICS_PATH)


if __name__ == "__main__":
    # `python -m src.train_sentiment` 실행 시 이 모듈은 '__main__' 로 로드된다.
    # 그대로 train() 을 호출하면 TF-IDF 에 넣은 tokenize 함수가 '__main__.tokenize'
    # 로 pickle 되어 재로딩이 깨진다. 정식 모듈 경로로 다시 import 해 실행하면
    # tokenize.__module__ 이 'src.train_sentiment' 로 고정되어 안전하게 저장된다.
    from src.train_sentiment import train as _train

    out = _train()
    print("\n=== 감성 분류 baseline 비교 ===")
    print("※ 라벨은 별점 기반 약지도 라벨(실제 감정 정답 아님)")
    for r in out["results"]:
        print(f"  {r['model']}: f1_macro={r['f1_macro']:.4f}, acc={r['accuracy']:.4f}")
