"""
株式推奨システム Webダッシュボード
"""
import streamlit as st
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from web.pages import home, performance, history, settings

# ページ設定
st.set_page_config(
    page_title="株式推奨システム",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stock-card {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("# 📊 ナビゲーション")

    page = st.radio(
        "ページを選択",
        ["🏠 ホーム", "📈 パフォーマンス", "📜 履歴", "⚙️ 設定"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### システム情報")
    st.info("""
    **株式推奨システム**

    テクニカル指標に基づく
    スイングトレード推奨

    - トレンド分析
    - モメンタム分析
    - 出来高分析
    - ボラティリティ分析
    """)

    st.markdown("---")
    st.markdown("### 免責事項")
    st.warning("""
    本システムは教育・研究目的です。
    投資判断は自己責任で行ってください。
    """)

# ページルーティング
if page == "🏠 ホーム":
    home.render()
elif page == "📈 パフォーマンス":
    performance.render()
elif page == "📜 履歴":
    history.render()
elif page == "⚙️ 設定":
    settings.render()
