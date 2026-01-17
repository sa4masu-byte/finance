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
# SCREENING CRITERIA
# =============================================================================

SCREENING_CRITERIA = {
    # Basic filters
    "min_market_cap": 10_000_000_000,  # 100億円以上
    "min_avg_volume": 100_000,          # 平均出来高10万株以上
    "min_price": 500,                   # 最低株価500円
    "max_price": 50_000,                # 最高株価5万円

    # Technical filters
    "min_score": 65,                    # 最低スコア65点（100点満点）
    "min_confidence": 0.70,             # 最低信頼度70%

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
    "source": "yfinance",

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
