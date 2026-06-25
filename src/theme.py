"""
공용 디자인 테마 (theme.py)

목적:
- 대시보드/차트/노트북 등 모든 산출물이 **동일한 색·폰트·스타일**을 쓰도록
  팔레트와 테마 적용 헬퍼를 한 곳에서 관리한다(단일 진실원본).

사용:
    from src.theme import PALETTE, SENTIMENT_COLORS, SEQ, chip_html
    from src.theme import apply_plotly_theme, apply_matplotlib_theme
    apply_plotly_theme()        # Plotly(px) 전역 템플릿 적용
    apply_matplotlib_theme()    # 노트북/저장 figure 용
"""
from __future__ import annotations

# ----------------------------------------------------------------------
# 팔레트 (깔끔한 블루 + 중간 채도 세만틱) — 절제되되 칙칙하지 않게
# ----------------------------------------------------------------------
PALETTE = {
    "primary": "#2F6FE0",
    "primary_dark": "#1E4FB0",
    "accent": "#7BA7E0",
    "ink": "#1F2733",         # 본문/숫자 기본
    "muted_text": "#6B7585",  # 보조 텍스트
    "ok": "#2E9E6B",          # 긍정/양호
    "neutral": "#A3ACBA",     # 중립
    "warn": "#E0A52E",        # 주의
    "bad": "#DD5C4E",         # 부정/위험
    "card_bg": "#FFFFFF",
    "card_border": "#E7EBF1",
    "grid": "#EEF1F5",
}

SENTIMENT_COLORS = {
    "positive": PALETTE["ok"],
    "neutral": PALETTE["neutral"],
    "negative": PALETTE["bad"],
}

# 일반 범주형 차트용 색 순서
SEQ = [PALETTE["primary"], PALETTE["accent"], PALETTE["warn"],
       PALETTE["ok"], PALETTE["neutral"], PALETTE["primary_dark"]]

# 칩(배지)용 소프트 톤: (배경, 글자, 테두리)
CHIP = {
    "primary": ("#EAF1FC", "#2F6FE0", "#D5E3F8"),
    "ok": ("#EAF6F0", "#2E9E6B", "#D2EADF"),
    "bad": ("#FBECEA", "#C8483B", "#F2D7D2"),
    "neutral": ("#F1F3F6", "#5B6675", "#E2E6EC"),
}

FONT_FAMILY = "Malgun Gothic, AppleGothic, NanumGothic, sans-serif"


def chip_html(text: str, tone: str = "primary") -> str:
    """소프트 톤 칩(배지) HTML 을 반환한다(Streamlit st.markdown 용)."""
    bg, fg, bd = CHIP.get(tone, CHIP["neutral"])
    return (f"<span style='background:{bg};color:{fg};border:1px solid {bd};"
            f"padding:3px 11px;border-radius:999px;font-size:0.82rem;font-weight:500'>{text}</span>")


def apply_plotly_theme(set_default: bool = True):
    """Plotly 전역 템플릿을 등록한다. px/go 차트가 동일한 폰트·그리드·배경을 따른다.

    개별 차트가 color_discrete_map 등으로 지정한 의미색은 그대로 우선한다.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    tmpl = go.layout.Template()
    tmpl.layout.colorway = SEQ
    tmpl.layout.font = dict(family=FONT_FAMILY, color=PALETTE["ink"], size=13)
    tmpl.layout.paper_bgcolor = "#FFFFFF"
    tmpl.layout.plot_bgcolor = "#FFFFFF"
    tmpl.layout.title = dict(font=dict(color=PALETTE["ink"], size=15))
    axis = dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["card_border"],
                linecolor=PALETTE["card_border"], tickfont=dict(color=PALETTE["muted_text"]))
    tmpl.layout.xaxis = axis
    tmpl.layout.yaxis = axis
    tmpl.layout.legend = dict(font=dict(color=PALETTE["ink"], size=12))
    pio.templates["supersol"] = tmpl
    if set_default:
        pio.templates.default = "plotly_white+supersol"
    return tmpl


def apply_matplotlib_theme():
    """matplotlib rcParams 를 팔레트에 맞춘다(노트북/리포트 figure 저장용)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": PALETTE["card_border"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["muted_text"],
        "ytick.color": PALETTE["muted_text"],
        "axes.prop_cycle": cycler(color=SEQ),
        "font.family": ["Malgun Gothic", "AppleGothic", "NanumGothic", "sans-serif"],
        "axes.unicode_minus": False,
    })
