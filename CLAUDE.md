# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Japanese stock swing trading recommendation system that uses technical indicators and machine learning optimization to identify profitable swing trading opportunities (3-15 day holding periods).

**Core Workflow:**
1. Fetch historical stock data from Stooq
2. Calculate technical indicators (trend, momentum, volume, volatility, pattern)
3. Score stocks using weighted composite scoring
4. Optimize weights via grid search + Bayesian optimization
5. Validate using walk-forward analysis
6. Generate trading recommendations

## Essential Commands

### Running Weight Optimization
```bash
# Main optimization workflow (grid search + Bayesian + walk-forward validation)
python scripts/run_optimization.py

# Expected runtime: 30-60 minutes depending on stock count
# Outputs saved to: data/reports/best_weights.json
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

### Testing
```bash
# Run tests (if tests exist)
pytest

# Run with coverage
pytest --cov=src --cov=backtesting
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

## High-Level Architecture

### Four-Layer Design

1. **Data Layer** (`src/data/`)
   - `fetcher.py`: StockDataFetcher - Fetches from Stooq with automatic caching
   - `cache.py`: Cache management using Parquet format
   - **Symbol format**: Converts `7203.T` → `7203.JP` for Stooq API

2. **Analysis Layer** (`src/analysis/`)
   - `indicators.py`: TechnicalIndicators class
   - Calculates 20+ indicators across 5 categories
   - **Fallback strategy**: Uses pandas-ta if available, otherwise manual calculations
   - All calculations are vectorized using pandas/numpy

3. **Backtesting Layer** (`backtesting/`)
   - `scoring_engine.py`: ScoringEngine - Converts indicators → composite score (0-100)
   - `backtest_engine.py`: BacktestEngine - Simulates trading with realistic costs
   - `optimizer.py`: WeightOptimizer - Two-phase optimization (grid → Bayesian)

4. **Configuration** (`config/`)
   - `settings.py`: Central configuration (indicators, risk params, optimization)
   - `stock_universe.json`: Monitored stocks with metadata

### Key Data Flow

```
Stock Symbols → StockDataFetcher → OHLCV DataFrame
                                        ↓
                          TechnicalIndicators.calculate_all()
                                        ↓
                            DataFrame with 20+ indicators
                                        ↓
                          ScoringEngine.calculate_score()
                                        ↓
                    Composite Score (trend 30%, momentum 25%, etc.)
                                        ↓
                          BacktestEngine.run_backtest()
                                        ↓
                        BacktestResults (win rate, Sharpe, etc.)
                                        ↓
                          WeightOptimizer.grid_search()
                          + bayesian_optimization()
                                        ↓
                            Optimal weights saved to JSON
```

## Critical Implementation Details

### Scoring System Architecture

The `ScoringEngine` uses a **multi-stage weighted scoring** approach:

1. **Component Scores** (each 0-100):
   - `_calculate_trend_score()`: SMA crossovers, MACD signals (max 30 points)
   - `_calculate_momentum_score()`: RSI zones, Stochastic (max 25 points)
   - `_calculate_volume_score()`: Volume ratios, OBV trends (max 20 points)
   - `_calculate_volatility_score()`: Bollinger Bands, ATR (max 15 points)
   - `_calculate_pattern_score()`: Support/resistance (max 10 points)

2. **Weighted Composite**:
   - Final score = Σ(component_score × weight)
   - Weights sum to 1.0 (validated in settings.py)
   - Default weights from literature, optimized via backtesting

3. **Confidence Calculation**:
   - Based on indicator availability (not all indicators may be computable)
   - Only recommend if confidence ≥ 70% and score ≥ 65

### Optimization Algorithm

**Two-Phase Approach** (backtesting/optimizer.py):

**Phase 1: Grid Search**
- Tests 500-800 weight combinations from parameter space
- Constraints: weights ∈ [0.05, 0.45], sum = 1.0
- Evaluates each via backtest on training period
- Scores using composite objective (35% win rate + 25% profit factor + 20% Sharpe + 15% drawdown + 5% holding days)

**Phase 2: Bayesian Optimization**
- Initializes with top 20 results from grid search
- Uses Gaussian Process (scikit-optimize) for efficient exploration
- 50 iterations to refine optimal weights
- Returns best weights + validation score

**Phase 3: Walk-Forward Validation**
- Tests optimized weights on 3 out-of-sample periods
- Detects overfitting (performance should be consistent)
- Reports averaged metrics across all test periods

### Backtesting Engine Details

The `BacktestEngine` simulates realistic trading:

1. **Position Management**:
   - Max 5 concurrent positions (configurable)
   - Position size: 20% of capital per stock
   - Entry: Highest-scored stocks first

2. **Exit Logic** (checked daily in order):
   - Stop loss: Price ≤ entry - (2 × ATR)
   - Profit target: Price ≥ entry × 1.15 (15% gain)
   - Max holding: 15 days
   - Score deterioration: Re-evaluated score < 50

3. **Cost Model**:
   - Commission: 0.1% per trade
   - Slippage: 0.05% per trade
   - Applied on both entry and exit

4. **Trade Class**:
   - Tracks entry/exit price, dates, shares, stop/target
   - Calculates return%, P&L, holding days
   - Records exit reason for analysis

## Important Configuration

### config/settings.py Structure

- `INDICATOR_PARAMS`: Technical indicator periods (SMA, RSI, MACD, etc.)
- `SCORING_WEIGHTS`: Initial category weights (overridden by optimization)
- `SCREENING_CRITERIA`: Minimum score, confidence, RSI boundaries
- `RISK_PARAMS`: Position sizing, stop loss, profit targets, holding periods
- `BACKTEST_CONFIG`: Capital, max positions, commission/slippage rates
- `OPTIMIZATION_CONFIG`: Parameter space, constraints, objective weights
- `DATA_FETCH_CONFIG`: Stooq settings, caching, rate limiting

### Key Files Generated

- `data/cache/*.parquet`: Cached stock data (gzipped Parquet)
- `data/reports/best_weights.json`: Current optimal weights
- `data/reports/optimization_results_*.json`: Full optimization history

## Development Guidelines

### When Modifying Scoring Logic

1. Update the appropriate `_calculate_*_score()` method in `scoring_engine.py`
2. Ensure score normalization to 0-100 range
3. Update max_score constant if adding/removing components
4. Re-run optimization to find new optimal weights: `python scripts/run_optimization.py`

### When Adding New Indicators

1. Add calculation method to `TechnicalIndicators` class (src/analysis/indicators.py)
2. Include both pandas-ta and manual fallback implementations
3. Add parameters to `INDICATOR_PARAMS` in config/settings.py
4. Update `get_latest_indicators()` to include new indicator
5. Integrate into appropriate scoring component in `scoring_engine.py`

### When Modifying Optimization

1. Parameter space defined in `OPTIMIZATION_CONFIG["param_space"]`
2. Objective weights control what's optimized for (win rate vs. Sharpe, etc.)
3. Walk-forward periods hardcoded in `run_optimization.py` (lines 67-74)
4. Success criteria checked at end of optimization (60% win rate, 2.0 PF, 1.5 Sharpe)

### Working with Stock Data

- **Symbol format**: Code handles both `.T` and `.JP` suffixes automatically
- **Caching**: Data cached for 24 hours, use `force_refresh=True` to bypass
- **Rate limiting**: Max 30 requests/minute to Stooq
- **Validation**: Checks for required columns, null values (<10%), positive prices
- **Date order**: Stooq returns reverse chronological; code auto-sorts ascending

## Common Gotchas

1. **Indicator NaN values**: Early rows lack sufficient data for indicators (e.g., SMA_75 needs 75 days). This is expected; scoring handles missing values gracefully.

2. **Weight constraint violations**: If manually setting weights, ensure they sum to 1.0 within 0.001 tolerance (checked by assertion).

3. **Bayesian optimization dependency**: Requires scikit-optimize. If missing, falls back to grid search best result.

4. **pandas-ta availability**: Code works without it (uses manual calculations). However, pandas-ta is faster if available.

5. **Walk-forward date ranges**: Hardcoded in run_optimization.py. Update these when extending data range or changing validation strategy.

6. **Parquet cache corruption**: If cache issues occur, delete `data/cache/*.parquet` and re-fetch.

## System Requirements

- Python 3.10+
- 500MB+ disk space for data cache
- Internet connection for Stooq data fetches
- 2GB+ RAM for optimization (processes all stocks × indicators in memory)

## Implementation Status

Currently implemented:
- ✅ Data fetching with caching
- ✅ 20+ technical indicators
- ✅ Composite scoring engine
- ✅ Backtesting engine with realistic costs
- ✅ Two-phase weight optimization
- ✅ Walk-forward validation

Not yet implemented (referenced in README):
- 🔄 Daily recommendation system (scripts/daily_scan.py stub)
- 🔄 Web dashboard (dependencies installed, not implemented)
- 🔄 Reporting module (src/reporting/ structure exists)
