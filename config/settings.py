"""
Configuration settings for Japanese stock swing trading recommender system
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_DIR = DATA_DIR / "db"
REPORTS_DIR = DATA_DIR / "reports"

# Ensure directories exist
for dir_path in [DATA_DIR, CACHE_DIR, DB_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# INDICATOR PARAMETERS
# =============================================================================

INDICATOR_PARAMS = {
    # Moving Averages
    "sma_short": 5,
    "sma_medium": 25,
    "sma_long": 75,

    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # RSI
    "rsi_period": 14,

    # Stochastic
    "stoch_k": 14,
    "stoch_d": 3,

    # Bollinger Bands
    "bb_period": 20,
    "bb_std": 2.0,

    # ATR
    "atr_period": 14,

    # Volume
    "volume_ma_period": 20,

    # OBV
    "obv_ma_period": 10,

    # ADX (Average Directional Index)
    "adx_period": 14,
}

# =============================================================================
# SCORING WEIGHTS (Initial values from literature - will be optimized)
# =============================================================================

SCORING_WEIGHTS = {
    "trend": 0.30,        # Trend indicators (SMA, MACD)
    "momentum": 0.25,     # Momentum indicators (RSI, Stochastic)
    "volume": 0.20,       # Volume indicators
    "volatility": 0.15,   # Volatility indicators (BB, ATR)
    "pattern": 0.10,      # Chart patterns
}

# Verify weights sum to 1.0
assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"

# =============================================================================
# SCORING COMPONENT PARAMETERS (for ScoringEngine)
# =============================================================================

SCORING_PARAMS = {
    # Trend scoring
    "trend": {
        "max_score": 45,  # Increased to accommodate ADX
        "price_above_sma": 7,
        "golden_cross": 8,
        "macd_above_signal": 8,
        "macd_histogram_positive": 7,
        "sma_aligned_bonus": 5,
        # ADX scoring (NEW)
        "adx_strong_trend": 8,        # ADX > 25 (strong trend)
        "adx_very_strong_trend": 10,  # ADX > 40 (very strong trend)
        "di_bullish_crossover": 5,    # +DI > -DI (bullish)
    },
    # Momentum scoring
    "momentum": {
        "max_score": 30,
        "rsi_optimal_min": 40,
        "rsi_optimal_max": 65,
        "rsi_optimal_score": 15,
        "rsi_oversold_min": 30,
        "rsi_oversold_max": 40,
        "rsi_oversold_score": 12,
        "rsi_room_min": 65,
        "rsi_room_max": 70,
        "rsi_room_score": 5,
        "stoch_bullish_crossover": 10,
        "stoch_overbought_threshold": 80,
        "stoch_not_overbought_score": 5,
    },
    # Volume scoring
    "volume": {
        "max_score": 25,
        "exceptional_threshold": 2.0,
        "exceptional_score": 20,
        "high_threshold": 1.5,
        "high_score": 12,
        "above_avg_threshold": 1.2,
        "above_avg_score": 6,
        "obv_uptrend_score": 5,
    },
    # Volatility scoring
    "volatility": {
        "max_score": 18,
        "near_lower_band_threshold": 0.3,
        "near_lower_band_score": 8,
        "below_middle_threshold": 0.5,
        "below_middle_score": 5,
        "tight_squeeze_threshold": 3,
        "tight_squeeze_score": 7,
        "moderate_squeeze_threshold": 5,
        "moderate_squeeze_score": 4,
        "low_atr_threshold": 3,
        "low_atr_score": 3,
    },
    # Pattern scoring
    "pattern": {
        "max_score": 10,
        "near_support_threshold": 0.02,
        "near_support_score": 5,
    },
}

# =============================================================================
# SCREENING CRITERIA
# =============================================================================

SCREENING_CRITERIA = {
    # Basic filters
    "min_market_cap": 10_000_000_000,  # 100億円以上
    "min_avg_volume": 100_000,          # 平均出来高10万株以上
    "min_price": 500,                   # 最低株価500円
    "max_price": 50_000,                # 最高株価5万円

    # Technical filters
    "min_score": 55,                    # 最低スコア55点（100点満点）【緩和版】
    "min_confidence": 0.65,             # 最低信頼度65%【緩和版】

    # Exclusions
    "exclude_gap_threshold": 0.05,      # 5%以上のギャップは除外
    "exclude_pre_earnings_days": 3,     # 決算3日前は除外

    # RSI boundaries
    "rsi_min": 20,                      # RSI下限（これ以下は除外）
    "rsi_max": 80,                      # RSI上限（これ以上は除外）
}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

RISK_PARAMS = {
    # Entry
    "max_recommendations": 5,           # 1日の最大推奨銘柄数
    "position_size_pct": 0.20,          # 1銘柄あたりの資金配分（20%）

    # Stop Loss
    "stop_loss_atr_multiplier": 2.0,    # ATR × 2でストップロス設定
    "max_loss_per_trade": 0.02,         # 1トレードあたり最大損失2%

    # Take Profit
    "profit_target_min": 0.05,          # 最低利益目標5%
    "profit_target_max": 0.15,          # 最高利益目標15%
    "trailing_stop_atr_multiplier": 3.0, # ATR × 3でトレーリングストップ

    # Holding Period
    "min_holding_days": 3,
    "max_holding_days": 15,

    # Re-evaluation
    "reeval_score_threshold": 50,       # 再評価でスコア50未満なら売却検討
}

# =============================================================================
# ENHANCED RISK MANAGEMENT (改善1-10)
# =============================================================================

ENHANCED_RISK_PARAMS = {
    # 1. Trailing Stop (改善1: トレーリングストップ)
    "trailing_stop_enabled": True,
    "trailing_stop_activation_pct": 0.03,  # 3%以上の含み益でトレーリング開始
    "trailing_stop_atr_multiplier": 1.5,   # ATR × 1.5でトレーリング
    "trailing_stop_pct": 0.02,             # 最高値から2%下落で決済

    # 2. Volatility-based Position Sizing (改善2: ボラティリティベースポジションサイジング)
    "volatility_position_sizing": True,
    "base_risk_per_trade": 0.02,           # 1トレードあたり基本リスク2%
    "min_position_pct": 0.10,              # 最小ポジションサイズ10%
    "max_position_pct": 0.30,              # 最大ポジションサイズ30%
    "volatility_adjustment_factor": 1.5,   # ボラティリティ調整係数

    # 6. Improved Stop Loss (改善6: スイングロー基準ストップロス)
    "swing_low_stop_enabled": True,
    "swing_low_lookback": 5,               # 過去5日のスイングロー
    "swing_low_buffer_pct": 0.005,         # スイングロー - 0.5%

    # 11. Dynamic Stop Loss (改善11: 動的ストップロス) - NEW
    "dynamic_stop_loss_enabled": True,
    "dynamic_stop_regimes": {
        "bull": {
            "atr_multiplier": 1.5,          # 強気相場: タイトなSL（ATR×1.5）
            "profit_target_pct": 0.12,      # 利益目標12%
            "trailing_activation_pct": 0.04, # 4%でトレーリング開始
        },
        "bear": {
            "atr_multiplier": 2.5,          # 弱気相場: ゆるめのSL（ATR×2.5）
            "profit_target_pct": 0.08,      # 利益目標8%
            "trailing_activation_pct": 0.03, # 3%でトレーリング開始
        },
        "sideways": {
            "atr_multiplier": 2.0,          # 横ばい相場: 標準SL（ATR×2.0）
            "profit_target_pct": 0.10,      # 利益目標10%
            "trailing_activation_pct": 0.035,# 3.5%でトレーリング開始
        },
        "high_volatility": {
            "atr_multiplier": 3.0,          # 高ボラ相場: 広めのSL（ATR×3.0）
            "profit_target_pct": 0.15,      # 利益目標15%
            "trailing_activation_pct": 0.05, # 5%でトレーリング開始
        },
    },
    "adx_based_adjustment": True,           # ADXに基づくSL調整
    "adx_strong_trend_threshold": 30,       # 強いトレンド判定閾値
    "adx_strong_trend_sl_tighten": 0.8,     # 強いトレンド時はSLを20%タイトに

    # 9. Compound Interest (改善9: 複利最適化)
    "compound_enabled": True,
    "compound_reinvest_pct": 0.50,         # 利益の50%を再投資
    "compound_threshold": 0.10,            # 10%以上の利益で複利適用
}

# =============================================================================
# MARKET REGIME DETECTION (改善7: マーケットレジーム検出)
# =============================================================================

MARKET_REGIME_PARAMS = {
    "enabled": True,
    "lookback_days": 20,
    "sma_short": 20,
    "sma_long": 50,

    # レジーム判定閾値
    "bull_threshold": 0.02,       # SMA20がSMA50を2%以上上回る → 強気相場
    "bear_threshold": -0.02,      # SMA20がSMA50を2%以上下回る → 弱気相場
    "volatility_threshold": 0.025, # 日次ボラティリティ2.5%以上 → 高ボラ相場

    # レジーム別調整
    "bull_position_multiplier": 1.2,   # 強気相場: ポジション1.2倍
    "bear_position_multiplier": 0.5,   # 弱気相場: ポジション0.5倍
    "high_vol_position_multiplier": 0.7, # 高ボラ相場: ポジション0.7倍

    # レジーム別エントリー閾値
    "bull_min_score": 55,          # 強気相場: スコア55以上でエントリー
    "bear_min_score": 75,          # 弱気相場: スコア75以上でエントリー
    "sideways_min_score": 65,      # 横ばい相場: スコア65以上でエントリー
}

# =============================================================================
# MULTI-TIMEFRAME ANALYSIS (改善4: マルチタイムフレーム分析)
# =============================================================================

MULTI_TIMEFRAME_PARAMS = {
    "enabled": True,
    "weekly_trend_confirmation": True,
    "monthly_trend_confirmation": True,  # 月足確認を追加
    "weekly_sma_period": 10,          # 週足10週移動平均
    "monthly_sma_period": 6,          # 月足6ヶ月移動平均

    # タイムフレーム一致ボーナス
    "daily_weekly_alignment_bonus": 10,      # 日足・週足の方向一致でスコア+10
    "all_timeframe_alignment_bonus": 20,     # 日足・週足・月足全一致でスコア+20
    "two_timeframe_alignment_bonus": 12,     # 2つのタイムフレーム一致でスコア+12
    "weekly_monthly_alignment_bonus": 8,     # 週足・月足の一致ボーナス

    # 週足条件
    "weekly_bullish_conditions": [
        "price_above_weekly_sma",      # 株価が週足SMAより上
        "weekly_sma_rising",           # 週足SMAが上昇中
    ],

    # 月足条件
    "monthly_bullish_conditions": [
        "price_above_monthly_sma",     # 株価が月足SMAより上
        "monthly_sma_rising",          # 月足SMAが上昇中
    ],
}

# =============================================================================
# VOLUME BREAKOUT (改善5: 出来高ブレイクアウト)
# =============================================================================

VOLUME_BREAKOUT_PARAMS = {
    "enabled": True,
    "volume_spike_threshold": 2.0,     # 平均出来高の2倍以上
    "price_breakout_pct": 0.02,        # 2%以上の上昇
    "breakout_confirmation_days": 2,   # 2日連続で確認
    "breakout_score_bonus": 15,        # ブレイクアウト確認でスコア+15
}

# =============================================================================
# ADDITIONAL FILTERS (改善8: 追加フィルター)
# =============================================================================

ADDITIONAL_FILTERS = {
    "enabled": True,

    # RSI divergence (RSIダイバージェンス)
    "rsi_divergence_enabled": True,
    "rsi_divergence_lookback": 10,
    "rsi_divergence_bonus": 8,

    # Moving average convergence (移動平均収束)
    "ma_convergence_enabled": True,
    "ma_convergence_threshold": 0.02,  # SMA5とSMA25が2%以内に収束
    "ma_convergence_bonus": 5,

    # Sector strength (セクター強度)
    "sector_strength_enabled": True,
    "sector_strength_bonus": 5,

    # Recent performance filter (直近パフォーマンスフィルター)
    "recent_performance_enabled": True,
    "recent_performance_days": 5,
    "recent_drawdown_max": -0.10,      # 直近5日で-10%以上下落した銘柄は除外

    # Earnings filter (決算フィルター) - NEW
    "earnings_filter_enabled": True,
    "earnings_blackout_days_before": 3,  # 決算発表3日前から除外
    "earnings_blackout_days_after": 2,   # 決算発表2日後まで除外
    "earnings_volatility_spike_threshold": 3.0,  # ATRの3倍以上の変動は決算関連と推定
    "earnings_calendar_file": None,       # 決算カレンダーファイルパス（オプション）

    # High volatility filter (高ボラティリティ除外)
    "high_volatility_filter_enabled": True,
    "high_volatility_atr_multiplier": 2.5,  # 平均ATRの2.5倍以上は除外
    "gap_filter_threshold": 0.05,          # 5%以上のギャップは除外
}

# =============================================================================
# MACHINE LEARNING SCORING (改善10: 機械学習スコアリング)
# =============================================================================

ML_SCORING_PARAMS = {
    "enabled": False,  # デフォルトは無効（モデル学習後に有効化）
    "model_type": "lightgbm",  # lightgbm or xgboost
    "model_path": REPORTS_DIR / "ml_model.pkl",
    "feature_columns": [
        "RSI_14", "MACD", "MACD_Histogram", "BB_Percent",
        "Stoch_K", "Stoch_D", "Volume_Ratio", "ATR_Percent",
        "SMA_5", "SMA_25", "SMA_75", "OBV_Trend",
    ],
    "ml_score_weight": 0.30,           # MLスコアの重み30%
    "traditional_score_weight": 0.70,  # 従来スコアの重み70%
    "min_confidence_threshold": 0.60,  # MLの信頼度閾値
}

# =============================================================================
# BACKTESTING CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = {
    # Data range
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",

    # Walk-forward analysis periods
    "training_months": 18,              # 18ヶ月のトレーニング期間
    "test_months": 6,                   # 6ヶ月のテスト期間

    # Trading costs
    "commission_rate": 0.001,           # 0.1% per trade
    "slippage_rate": 0.0005,            # 0.05% slippage

    # Initial capital
    "initial_capital": 1_000_000,       # 100万円

    # Portfolio constraints
    "max_positions": 5,                 # 最大同時保有5銘柄
}

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

OPTIMIZATION_CONFIG = {
    # Parameter space for grid search
    "param_space": {
        "trend": [0.20, 0.25, 0.30, 0.35, 0.40],
        "momentum": [0.15, 0.20, 0.25, 0.30, 0.35],
        "volume": [0.10, 0.15, 0.20, 0.25, 0.30],
        "volatility": [0.10, 0.15, 0.20, 0.25],
        "pattern": [0.05, 0.10, 0.15, 0.20],
    },

    # Constraints
    "weight_sum": 1.0,
    "min_weight": 0.05,
    "max_weight": 0.45,

    # Optimization objective (composite score)
    "objective_weights": {
        "win_rate": 0.35,
        "profit_factor": 0.25,
        "sharpe_ratio": 0.20,
        "max_drawdown": 0.15,
        "holding_days": 0.05,
    },

    # Bayesian optimization
    "n_initial_points": 20,             # Top N from grid search
    "n_iterations": 50,                 # Bayesian optimization iterations

    # Success criteria
    "target_win_rate": 0.60,
    "target_profit_factor": 2.0,
    "target_sharpe_ratio": 1.5,
    "target_max_drawdown": 0.15,
}

# =============================================================================
# DATA FETCHING
# =============================================================================

DATA_FETCH_CONFIG = {
    # Data source
    "source": "stooq",  # Using Stooq via pandas-datareader

    # Retry settings
    "max_retries": 3,
    "retry_delay": 2,  # seconds

    # Cache settings
    "cache_enabled": True,
    "cache_expiry_hours": 24,

    # Rate limiting
    "requests_per_minute": 30,
}

# =============================================================================
# REPORTING
# =============================================================================

REPORT_CONFIG = {
    # Output format
    "formats": ["cli", "json", "csv", "html"],

    # CLI display
    "max_display_recommendations": 5,
    "use_colors": True,

    # Chart generation
    "include_charts": True,
    "chart_format": "png",
    "chart_dpi": 150,
}

# =============================================================================
# LOGGING
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "recommender.log",
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
