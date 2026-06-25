# 디자인 가이드 (색·스타일 일관성)

대시보드·차트·발표자료·문서가 **같은 톤**을 유지하기 위한 기준입니다.
색 정의의 단일 진실원본은 `src/theme.py` 의 `PALETTE` 이며, Streamlit 위젯 색은
`.streamlit/config.toml` 의 `[theme]` 가 관리합니다.

## 1. 컬러 팔레트

| 역할 | HEX | 용도 |
|------|-----|------|
| Primary | `#2F6FE0` | 강조·기준 앱·버튼/링크 |
| Primary Dark | `#1E4FB0` | 보조 강조 |
| Accent | `#7BA7E0` | 보조 데이터 시리즈 |
| Ink | `#1F2733` | 제목·숫자·본문 |
| Muted text | `#6B7585` | 라벨·캡션 |
| OK (긍정/양호) | `#2E9E6B` | positive, 통과, 좋은 지표 |
| Neutral | `#A3ACBA` | neutral, 보조 |
| Warn (주의) | `#E0A52E` | 경고, 중간 지표 |
| Bad (부정/위험) | `#DD5C4E` | negative, 실패, 나쁜 지표 |
| Card border | `#E7EBF1` | 카드/구분선 |
| Grid | `#EEF1F5` | 차트 격자 |

**감성 색 고정**: positive=초록 / neutral=회색 / negative=빨강 (혼동 방지).

## 2. 사용 원칙

- **꽉 찬 원색 박스 지양**: 상태 표시는 소프트 칩(연한 배경 + 테두리, `chip_html`) 사용.
- **숫자/제목은 Ink** 기본, 색은 강조가 필요한 곳에만 절제해서.
- **차트**: `apply_plotly_theme()`(앱) / `apply_matplotlib_theme()`(노트북) 호출 →
  폰트·격자·배경이 자동 통일됨. 의미색(감성 등)은 개별 지정 우선.
- **버튼/위젯 강조색**: Streamlit `[theme] primaryColor` 가 `#2F6FE0` 로 통일.

## 3. 코드에서 쓰는 법

```python
# Streamlit / 차트
from src.theme import PALETTE, SENTIMENT_COLORS, SEQ, chip_html, apply_plotly_theme
apply_plotly_theme()
st.markdown(chip_html("상태 양호", "ok"), unsafe_allow_html=True)
fig = px.pie(..., color_discrete_map=SENTIMENT_COLORS)

# 노트북 / 리포트용 matplotlib figure
from src.theme import apply_matplotlib_theme
apply_matplotlib_theme()
```

## 4. 발표자료(PPT)·문서 적용 팁

- 표지/강조: Primary `#2F6FE0`, 본문 텍스트: Ink `#1F2733`, 보조: Muted `#6B7585`.
- 좋음/주의/위험 신호는 OK/Warn/Bad 3색만 사용(과한 색 추가 금지).
- 배경은 흰색 기준, 카드/박스는 `#E7EBF1` 테두리로 가볍게 구분.
