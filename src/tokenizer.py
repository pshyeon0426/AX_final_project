"""
한국어 토크나이저 (kiwipiepy 우선, 실패 시 공백 기반 fallback)

설계 원칙:
- kiwipiepy 설치/로딩에 실패해도 분석 파이프라인이 중단되지 않도록 한다.
- 동일한 인터페이스 `tokenize(text) -> list[str]` 를 제공하여
  EDA/토픽분석/벡터화 등 상위 모듈이 백엔드 종류를 신경 쓰지 않게 한다.

사용 예:
    from src.tokenizer import get_tokenizer
    tok = get_tokenizer()           # 환경에 맞는 토크나이저 자동 선택
    tokens = tok.tokenize("슈퍼솔 로그인이 자꾸 안돼요")
    print(tok.backend)              # "kiwi" 또는 "whitespace"
"""
from __future__ import annotations

import re
from typing import Protocol

from src.config import settings

# 간단한 한국어 불용어 (필요 시 config/ 로 분리 확장 가능)
_DEFAULT_STOPWORDS = {
    "그리고", "그러나", "하지만", "그래서", "그냥", "정말", "너무", "진짜",
    "수", "것", "등", "들", "에", "를", "을", "이", "가", "은", "는", "도",
}

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


class Tokenizer(Protocol):
    backend: str

    def tokenize(self, text: str) -> list[str]:
        ...


class WhitespaceTokenizer:
    """공백/정규식 기반 fallback 토크나이저. 외부 의존성이 없다."""

    backend = "whitespace"

    def __init__(self, stopwords: set[str] | None = None, min_len: int = 2) -> None:
        self.stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS
        self.min_len = min_len

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = _TOKEN_RE.findall(str(text))
        return [
            t for t in tokens
            if len(t) >= self.min_len and t not in self.stopwords
        ]


class KiwiTokenizer:
    """kiwipiepy 기반 토크나이저. 명사/동사/형용사 등 내용어 위주로 추출."""

    backend = "kiwi"
    # 분석에 유의미한 품사 (체언/용언/외국어/숫자)
    _KEEP_TAGS = ("NNG", "NNP", "VV", "VA", "SL", "SH", "SN", "XR")

    def __init__(self, stopwords: set[str] | None = None, min_len: int = 2) -> None:
        from kiwipiepy import Kiwi  # 지연 import: 설치 안 됐으면 여기서 ImportError

        self._kiwi = Kiwi()
        self.stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS
        self.min_len = min_len

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        result = []
        for token in self._kiwi.tokenize(str(text)):
            if token.tag in self._KEEP_TAGS and len(token.form) >= self.min_len:
                if token.form not in self.stopwords:
                    result.append(token.form)
        return result


def get_tokenizer(prefer: str | None = None) -> Tokenizer:
    """환경에 맞는 토크나이저를 반환한다.

    Args:
        prefer: "kiwi" | "whitespace". None 이면 settings.tokenizer 사용.

    동작:
        - prefer 가 "kiwi" 이고 kiwipiepy 가 설치되어 있으면 KiwiTokenizer.
        - 설치 실패(ImportError 등)하거나 prefer 가 "whitespace" 이면
          WhitespaceTokenizer 로 자동 fallback 한다.
    """
    choice = (prefer or settings.tokenizer or "kiwi").lower()

    if choice == "whitespace":
        return WhitespaceTokenizer()

    try:
        return KiwiTokenizer()
    except Exception as exc:  # ImportError 외 초기화 실패도 포함
        import warnings

        warnings.warn(
            f"kiwipiepy 로딩 실패 ({exc!r}). 공백 기반 토크나이저로 fallback 합니다. "
            f"형태소 분석 품질이 낮아질 수 있습니다.",
            RuntimeWarning,
            stacklevel=2,
        )
        return WhitespaceTokenizer()


if __name__ == "__main__":
    tok = get_tokenizer()
    sample = "슈퍼솔 로그인이 자꾸 안되고 송금 속도가 너무 느려요"
    print(f"backend: {tok.backend}")
    print(f"tokens : {tok.tokenize(sample)}")
