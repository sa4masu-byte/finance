"""
Sector Rotation and Correlation Analysis Module

This module provides:
1. Sector rotation detection based on relative strength
2. Stock correlation analysis within and across sectors
3. Sector momentum scoring for position sizing
4. Risk-adjusted sector allocation recommendations

Based on research:
- Sector rotation strategies (Stangl et al., 2009)
- Momentum effects in sector ETFs (Moskowitz & Grinblatt, 1999)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SectorPhase(Enum):
    """Market cycle phases for sector rotation"""
    EARLY_EXPANSION = "early_expansion"     # Recovery phase - favor cyclicals
    LATE_EXPANSION = "late_expansion"       # Growth phase - favor tech, consumer
    EARLY_CONTRACTION = "early_contraction" # Slowdown - favor defensives
    LATE_CONTRACTION = "late_contraction"   # Recession - favor utilities, healthcare


@dataclass
class SectorMetrics:
    """Sector performance metrics"""
    sector: str
    momentum_1m: float      # 1-month momentum
    momentum_3m: float      # 3-month momentum
    momentum_6m: float      # 6-month momentum
    relative_strength: float # Relative to market
    volatility: float       # 20-day volatility
    correlation_to_market: float
    sector_score: float     # Combined score 0-100


# Japanese stock sectors mapping
SECTOR_MAPPING = {
    # Technology
    "6758.T": "Technology",   # Sony
    "6861.T": "Technology",   # Keyence
    "6981.T": "Technology",   # Murata
    "9984.T": "Technology",   # SoftBank Group

    # Automotive
    "7203.T": "Automotive",   # Toyota
    "7267.T": "Automotive",   # Honda

    # Financial
    "8306.T": "Financial",    # MUFG
    "8316.T": "Financial",    # SMFG

    # Telecom
    "9432.T": "Telecom",      # NTT

    # Healthcare/Pharma
    "4502.T": "Healthcare",   # Takeda

    # Consumer/Retail
    "9983.T": "Consumer",     # Fast Retailing
    "7974.T": "Consumer",     # Nintendo

    # Materials/Industrial
    "4063.T": "Materials",    # Shin-Etsu Chemical

    # Services
    "6098.T": "Services",     # Recruit
}

# Sector rotation order in economic cycle
SECTOR_ROTATION_ORDER = [
    "Technology",   # Early expansion
    "Consumer",     # Mid expansion
    "Automotive",   # Mid expansion
    "Services",     # Late expansion
    "Materials",    # Late expansion/early contraction
    "Financial",    # Transition
    "Telecom",      # Defensive
    "Healthcare",   # Defensive/late contraction
]


class SectorAnalyzer:
    """Sector rotation and correlation analysis"""

    def __init__(
        self,
        sector_mapping: Optional[Dict[str, str]] = None,
        lookback_periods: Dict[str, int] = None,
    ):
        """
        Initialize sector analyzer

        Args:
            sector_mapping: Dict mapping stock symbols to sectors
            lookback_periods: Dict with momentum lookback periods
        """
        self.sector_mapping = sector_mapping or SECTOR_MAPPING
        self.lookback_periods = lookback_periods or {
            "short": 20,    # 1 month
            "medium": 60,   # 3 months
            "long": 120,    # 6 months
        }
        self._sector_data: Dict[str, pd.DataFrame] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def get_sector(self, symbol: str) -> str:
        """Get sector for a stock symbol"""
        return self.sector_mapping.get(symbol, "Unknown")

    def load_sector_data(self, stock_data: Dict[str, pd.DataFrame]) -> None:
        """
        Load stock data and aggregate by sector

        Args:
            stock_data: Dict mapping symbols to OHLCV DataFrames
        """
        self._sector_data = {}

        # Group stocks by sector
        sector_stocks: Dict[str, List[pd.DataFrame]] = {}
        for symbol, df in stock_data.items():
            sector = self.get_sector(symbol)
            if sector == "Unknown":
                continue

            if sector not in sector_stocks:
                sector_stocks[sector] = []

            # Normalize prices to 100 at start for comparison
            if len(df) > 0:
                normalized = df.copy()
                normalized["Close"] = (df["Close"] / df["Close"].iloc[0]) * 100
                sector_stocks[sector].append(normalized)

        # Calculate sector average performance
        for sector, dfs in sector_stocks.items():
            if len(dfs) > 0:
                # Align DataFrames by date and average
                combined = pd.concat(
                    [df[["Close"]].rename(columns={"Close": f"stock_{i}"})
                     for i, df in enumerate(dfs)],
                    axis=1
                )
                combined["Sector_Close"] = combined.mean(axis=1)
                self._sector_data[sector] = combined

    def calculate_sector_momentum(self, sector: str) -> Dict[str, float]:
        """
        Calculate momentum metrics for a sector

        Args:
            sector: Sector name

        Returns:
            Dict with momentum metrics
        """
        if sector not in self._sector_data:
            return {
                "momentum_1m": 0.0,
                "momentum_3m": 0.0,
                "momentum_6m": 0.0,
            }

        df = self._sector_data[sector]
        close = df["Sector_Close"]

        current = close.iloc[-1] if len(close) > 0 else 0

        # Calculate momentum as percentage change
        def safe_momentum(periods: int) -> float:
            if len(close) > periods:
                prev = close.iloc[-periods-1]
                return ((current - prev) / prev) * 100 if prev != 0 else 0.0
            return 0.0

        return {
            "momentum_1m": safe_momentum(self.lookback_periods["short"]),
            "momentum_3m": safe_momentum(self.lookback_periods["medium"]),
            "momentum_6m": safe_momentum(self.lookback_periods["long"]),
        }

    def calculate_relative_strength(
        self,
        sector: str,
        market_data: pd.DataFrame
    ) -> float:
        """
        Calculate relative strength vs market

        Args:
            sector: Sector name
            market_data: Market index OHLCV data

        Returns:
            Relative strength ratio (>1 = outperforming)
        """
        if sector not in self._sector_data or len(market_data) == 0:
            return 1.0

        sector_df = self._sector_data[sector]
        lookback = self.lookback_periods["medium"]

        if len(sector_df) < lookback or len(market_data) < lookback:
            return 1.0

        # Sector performance
        sector_close = sector_df["Sector_Close"]
        sector_return = (
            (sector_close.iloc[-1] - sector_close.iloc[-lookback]) /
            sector_close.iloc[-lookback]
        )

        # Market performance
        market_close = market_data["Close"]
        market_return = (
            (market_close.iloc[-1] - market_close.iloc[-lookback]) /
            market_close.iloc[-lookback]
        )

        # Relative strength
        if market_return != 0:
            return (1 + sector_return) / (1 + market_return)
        return 1.0

    def calculate_sector_volatility(self, sector: str) -> float:
        """
        Calculate sector volatility (annualized)

        Args:
            sector: Sector name

        Returns:
            Annualized volatility percentage
        """
        if sector not in self._sector_data:
            return 0.0

        df = self._sector_data[sector]
        if len(df) < 20:
            return 0.0

        close = df["Sector_Close"]
        returns = close.pct_change().dropna()

        if len(returns) < 20:
            return 0.0

        # 20-day volatility, annualized
        daily_vol = returns.tail(20).std()
        return daily_vol * np.sqrt(252) * 100

    def calculate_correlation_matrix(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between stocks

        Args:
            stock_data: Dict mapping symbols to OHLCV DataFrames

        Returns:
            Correlation matrix DataFrame
        """
        returns_data = {}

        for symbol, df in stock_data.items():
            if len(df) > 20:
                returns = df["Close"].pct_change().dropna()
                returns_data[symbol] = returns

        if len(returns_data) < 2:
            return pd.DataFrame()

        # Align by date
        returns_df = pd.DataFrame(returns_data)
        self._correlation_matrix = returns_df.corr()

        return self._correlation_matrix

    def get_sector_correlation(
        self,
        sector1: str,
        sector2: str
    ) -> float:
        """
        Get average correlation between two sectors

        Args:
            sector1: First sector name
            sector2: Second sector name

        Returns:
            Average correlation coefficient
        """
        if self._correlation_matrix is None:
            return 0.0

        # Get symbols for each sector
        sector1_symbols = [
            s for s, sec in self.sector_mapping.items()
            if sec == sector1 and s in self._correlation_matrix.columns
        ]
        sector2_symbols = [
            s for s, sec in self.sector_mapping.items()
            if sec == sector2 and s in self._correlation_matrix.columns
        ]

        if not sector1_symbols or not sector2_symbols:
            return 0.0

        # Calculate average cross-sector correlation
        correlations = []
        for s1 in sector1_symbols:
            for s2 in sector2_symbols:
                if s1 != s2:
                    correlations.append(self._correlation_matrix.loc[s1, s2])

        return np.mean(correlations) if correlations else 0.0

    def detect_market_phase(
        self,
        market_data: pd.DataFrame
    ) -> SectorPhase:
        """
        Detect current market cycle phase

        Args:
            market_data: Market index OHLCV data

        Returns:
            Current SectorPhase
        """
        if len(market_data) < 120:
            return SectorPhase.LATE_EXPANSION

        close = market_data["Close"]

        # Calculate short and long term momentum
        sma_20 = close.rolling(20).mean()
        sma_60 = close.rolling(60).mean()

        current_close = close.iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_60 = sma_60.iloc[-1]

        # Determine trend direction
        short_trend = current_close > current_sma_20
        long_trend = current_sma_20 > current_sma_60

        # Momentum acceleration
        momentum_20 = (current_close - close.iloc[-20]) / close.iloc[-20]
        momentum_60 = (current_close - close.iloc[-60]) / close.iloc[-60]

        # Phase detection logic
        if short_trend and long_trend:
            if momentum_20 > momentum_60 * 0.5:
                return SectorPhase.EARLY_EXPANSION
            else:
                return SectorPhase.LATE_EXPANSION
        elif not short_trend and long_trend:
            return SectorPhase.EARLY_CONTRACTION
        elif not short_trend and not long_trend:
            return SectorPhase.LATE_CONTRACTION
        else:
            # Short trend up but long trend down - transition
            return SectorPhase.EARLY_EXPANSION

    def get_recommended_sectors(
        self,
        market_phase: SectorPhase
    ) -> List[str]:
        """
        Get recommended sectors for current market phase

        Args:
            market_phase: Current market phase

        Returns:
            List of recommended sector names
        """
        phase_sectors = {
            SectorPhase.EARLY_EXPANSION: ["Technology", "Consumer", "Automotive"],
            SectorPhase.LATE_EXPANSION: ["Services", "Materials", "Consumer"],
            SectorPhase.EARLY_CONTRACTION: ["Financial", "Telecom", "Healthcare"],
            SectorPhase.LATE_CONTRACTION: ["Healthcare", "Telecom", "Consumer"],
        }
        return phase_sectors.get(market_phase, list(self.sector_mapping.values()))

    def calculate_sector_score(
        self,
        sector: str,
        market_data: Optional[pd.DataFrame] = None
    ) -> SectorMetrics:
        """
        Calculate comprehensive sector score

        Args:
            sector: Sector name
            market_data: Optional market index data for relative strength

        Returns:
            SectorMetrics with all calculations
        """
        momentum = self.calculate_sector_momentum(sector)
        volatility = self.calculate_sector_volatility(sector)

        relative_strength = 1.0
        if market_data is not None:
            relative_strength = self.calculate_relative_strength(sector, market_data)

        # Calculate correlation to market (if available)
        correlation_to_market = 0.5  # Default
        if self._correlation_matrix is not None and market_data is not None:
            sector_symbols = [
                s for s, sec in self.sector_mapping.items()
                if sec == sector and s in self._correlation_matrix.columns
            ]
            if sector_symbols:
                # Use average of sector stock correlations
                correlation_to_market = 0.5  # Placeholder

        # Combined sector score (0-100)
        # Weight: 40% momentum, 30% relative strength, 30% inverse volatility
        momentum_score = min(max(
            (momentum["momentum_3m"] + 10) / 20 * 40,  # Normalize -10% to +10%
            0
        ), 40)

        rs_score = min(max(
            (relative_strength - 0.9) / 0.2 * 30,  # 0.9 to 1.1 range
            0
        ), 30)

        vol_score = min(max(
            (30 - volatility) / 30 * 30,  # Lower vol = higher score
            0
        ), 30)

        sector_score = momentum_score + rs_score + vol_score

        return SectorMetrics(
            sector=sector,
            momentum_1m=momentum["momentum_1m"],
            momentum_3m=momentum["momentum_3m"],
            momentum_6m=momentum["momentum_6m"],
            relative_strength=relative_strength,
            volatility=volatility,
            correlation_to_market=correlation_to_market,
            sector_score=sector_score,
        )

    def get_all_sector_scores(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, SectorMetrics]:
        """
        Calculate scores for all sectors

        Args:
            stock_data: Dict mapping symbols to OHLCV DataFrames
            market_data: Optional market index data

        Returns:
            Dict mapping sector names to SectorMetrics
        """
        self.load_sector_data(stock_data)
        self.calculate_correlation_matrix(stock_data)

        sectors = set(self.sector_mapping.values())
        scores = {}

        for sector in sectors:
            scores[sector] = self.calculate_sector_score(sector, market_data)

        return scores

    def get_stock_sector_adjustment(
        self,
        symbol: str,
        stock_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Get position size adjustment factor based on sector analysis

        Args:
            symbol: Stock symbol
            stock_data: Dict mapping symbols to OHLCV DataFrames
            market_data: Optional market index data

        Returns:
            Adjustment multiplier (0.5 to 1.5)
        """
        sector = self.get_sector(symbol)
        if sector == "Unknown":
            return 1.0

        # Calculate sector scores if not done
        if not self._sector_data:
            self.load_sector_data(stock_data)

        metrics = self.calculate_sector_score(sector, market_data)

        # Score-based adjustment
        # Score 0-33: 0.5-0.8x, 34-66: 0.8-1.2x, 67-100: 1.2-1.5x
        if metrics.sector_score < 33:
            adjustment = 0.5 + (metrics.sector_score / 33) * 0.3
        elif metrics.sector_score < 67:
            adjustment = 0.8 + ((metrics.sector_score - 33) / 33) * 0.4
        else:
            adjustment = 1.2 + ((metrics.sector_score - 67) / 33) * 0.3

        return round(adjustment, 2)

    def rank_sectors(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank sectors by score

        Args:
            stock_data: Dict mapping symbols to OHLCV DataFrames
            market_data: Optional market index data

        Returns:
            List of (sector_name, score) tuples, sorted by score descending
        """
        scores = self.get_all_sector_scores(stock_data, market_data)

        ranked = [
            (sector, metrics.sector_score)
            for sector, metrics in scores.items()
        ]

        return sorted(ranked, key=lambda x: x[1], reverse=True)


# Convenience functions
def get_sector_for_stock(symbol: str) -> str:
    """Get sector for a stock symbol"""
    return SECTOR_MAPPING.get(symbol, "Unknown")


def is_defensive_sector(sector: str) -> bool:
    """Check if sector is considered defensive"""
    defensive = {"Healthcare", "Telecom", "Consumer"}
    return sector in defensive


def is_cyclical_sector(sector: str) -> bool:
    """Check if sector is considered cyclical"""
    cyclical = {"Technology", "Automotive", "Materials", "Financial"}
    return sector in cyclical
