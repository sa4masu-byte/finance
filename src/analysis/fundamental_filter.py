"""
Fundamental Analysis Filter Module

Provides fundamental data filtering for stock selection:
- P/E Ratio (Price-to-Earnings)
- PBR (Price-to-Book Ratio)
- Dividend Yield
- ROE (Return on Equity)
- Market Cap filtering

Data sources:
- Yahoo Finance API (via yfinance)
- Manual data input for unavailable stocks
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Fundamental data for a stock"""
    symbol: str
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pbr: Optional[float] = None  # Price-to-Book Ratio
    dividend_yield: Optional[float] = None  # As percentage
    roe: Optional[float] = None  # Return on Equity
    market_cap: Optional[float] = None  # In JPY
    revenue_growth: Optional[float] = None  # YoY growth
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if fundamental data is valid (not stale)"""
        return (datetime.now() - self.last_updated) < timedelta(days=7)

    @property
    def quality_score(self) -> float:
        """Calculate quality score based on fundamentals (0-100)"""
        score = 50  # Base score

        # P/E scoring (10-25 is good for growth stocks)
        if self.pe_ratio is not None:
            if 0 < self.pe_ratio < 10:
                score += 15  # Value stock
            elif 10 <= self.pe_ratio <= 25:
                score += 10  # Reasonable
            elif 25 < self.pe_ratio <= 40:
                score += 0   # Growth premium
            elif self.pe_ratio > 40:
                score -= 10  # Expensive
            elif self.pe_ratio < 0:
                score -= 15  # Negative earnings

        # PBR scoring (1-3 is reasonable)
        if self.pbr is not None:
            if 0 < self.pbr < 1:
                score += 10  # Undervalued
            elif 1 <= self.pbr <= 2:
                score += 5
            elif self.pbr > 3:
                score -= 5

        # Dividend yield (>2% is attractive)
        if self.dividend_yield is not None:
            if self.dividend_yield >= 4:
                score += 10
            elif self.dividend_yield >= 2:
                score += 5

        # ROE (>10% is good)
        if self.roe is not None:
            if self.roe >= 15:
                score += 15
            elif self.roe >= 10:
                score += 10
            elif self.roe >= 5:
                score += 5
            elif self.roe < 0:
                score -= 10

        # Debt/Equity (lower is better)
        if self.debt_to_equity is not None:
            if self.debt_to_equity < 0.5:
                score += 5
            elif self.debt_to_equity > 2:
                score -= 10

        return max(0, min(100, score))


# Pre-defined fundamental data for major Japanese stocks
# (Since real-time API access may be limited)
DEFAULT_FUNDAMENTALS: Dict[str, Dict] = {
    "7203.T": {  # Toyota
        "pe_ratio": 10.5,
        "pbr": 1.1,
        "dividend_yield": 2.8,
        "roe": 12.5,
        "market_cap": 45_000_000_000_000,
    },
    "7267.T": {  # Honda
        "pe_ratio": 8.5,
        "pbr": 0.7,
        "dividend_yield": 3.5,
        "roe": 9.8,
        "market_cap": 8_000_000_000_000,
    },
    "6758.T": {  # Sony
        "pe_ratio": 18.2,
        "pbr": 2.3,
        "dividend_yield": 0.6,
        "roe": 14.2,
        "market_cap": 18_000_000_000_000,
    },
    "6861.T": {  # Keyence
        "pe_ratio": 45.0,
        "pbr": 8.5,
        "dividend_yield": 0.3,
        "roe": 20.5,
        "market_cap": 16_000_000_000_000,
    },
    "6981.T": {  # Murata
        "pe_ratio": 22.0,
        "pbr": 2.5,
        "dividend_yield": 1.5,
        "roe": 12.0,
        "market_cap": 5_500_000_000_000,
    },
    "8306.T": {  # MUFG
        "pe_ratio": 12.0,
        "pbr": 0.8,
        "dividend_yield": 3.2,
        "roe": 7.5,
        "market_cap": 18_000_000_000_000,
    },
    "8316.T": {  # SMFG
        "pe_ratio": 11.5,
        "pbr": 0.7,
        "dividend_yield": 3.8,
        "roe": 7.0,
        "market_cap": 9_500_000_000_000,
    },
    "9432.T": {  # NTT
        "pe_ratio": 12.5,
        "pbr": 1.6,
        "dividend_yield": 3.0,
        "roe": 13.5,
        "market_cap": 15_000_000_000_000,
    },
    "9984.T": {  # SoftBank Group
        "pe_ratio": None,  # Often volatile/negative
        "pbr": 1.2,
        "dividend_yield": 0.8,
        "roe": 5.0,
        "market_cap": 12_000_000_000_000,
    },
    "4502.T": {  # Takeda
        "pe_ratio": 35.0,
        "pbr": 1.3,
        "dividend_yield": 4.5,
        "roe": 4.0,
        "market_cap": 7_000_000_000_000,
    },
    "9983.T": {  # Fast Retailing
        "pe_ratio": 42.0,
        "pbr": 7.5,
        "dividend_yield": 0.8,
        "roe": 18.0,
        "market_cap": 13_000_000_000_000,
    },
    "7974.T": {  # Nintendo
        "pe_ratio": 18.0,
        "pbr": 4.2,
        "dividend_yield": 2.5,
        "roe": 22.0,
        "market_cap": 10_000_000_000_000,
    },
    "4063.T": {  # Shin-Etsu Chemical
        "pe_ratio": 15.0,
        "pbr": 2.8,
        "dividend_yield": 2.0,
        "roe": 18.5,
        "market_cap": 9_000_000_000_000,
    },
    "6098.T": {  # Recruit
        "pe_ratio": 30.0,
        "pbr": 6.5,
        "dividend_yield": 0.7,
        "roe": 22.0,
        "market_cap": 9_500_000_000_000,
    },
}


class FundamentalFilter:
    """Filter stocks based on fundamental analysis"""

    def __init__(
        self,
        use_api: bool = False,
        cache_days: int = 7,
    ):
        """
        Initialize fundamental filter

        Args:
            use_api: Whether to fetch data from API (requires yfinance)
            cache_days: Number of days to cache fundamental data
        """
        self.use_api = use_api
        self.cache_days = cache_days
        self._cache: Dict[str, FundamentalData] = {}
        self._load_default_data()

    def _load_default_data(self) -> None:
        """Load default fundamental data"""
        for symbol, data in DEFAULT_FUNDAMENTALS.items():
            self._cache[symbol] = FundamentalData(
                symbol=symbol,
                pe_ratio=data.get("pe_ratio"),
                pbr=data.get("pbr"),
                dividend_yield=data.get("dividend_yield"),
                roe=data.get("roe"),
                market_cap=data.get("market_cap"),
            )

    def get_fundamentals(self, symbol: str) -> FundamentalData:
        """
        Get fundamental data for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            FundamentalData object
        """
        # Check cache
        if symbol in self._cache and self._cache[symbol].is_valid:
            return self._cache[symbol]

        # Try API if enabled
        if self.use_api:
            data = self._fetch_from_api(symbol)
            if data:
                self._cache[symbol] = data
                return data

        # Return cached or empty data
        if symbol in self._cache:
            return self._cache[symbol]

        return FundamentalData(symbol=symbol)

    def _fetch_from_api(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch fundamental data from Yahoo Finance API"""
        try:
            import yfinance as yf

            # Convert symbol format
            yf_symbol = symbol.replace(".T", ".T")  # Already correct format

            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            return FundamentalData(
                symbol=symbol,
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                pbr=info.get("priceToBook"),
                dividend_yield=info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
                roe=info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
                market_cap=info.get("marketCap"),
                profit_margin=info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
                debt_to_equity=info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else None,
            )

        except ImportError:
            logger.warning("yfinance not installed. Using default data.")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return None

    def filter_by_pe(
        self,
        symbols: List[str],
        min_pe: Optional[float] = None,
        max_pe: Optional[float] = None,
    ) -> List[str]:
        """
        Filter symbols by P/E ratio

        Args:
            symbols: List of stock symbols
            min_pe: Minimum P/E ratio (None to skip)
            max_pe: Maximum P/E ratio (None to skip)

        Returns:
            Filtered list of symbols
        """
        filtered = []
        for symbol in symbols:
            data = self.get_fundamentals(symbol)
            if data.pe_ratio is None:
                filtered.append(symbol)  # Include if no data
                continue

            if min_pe is not None and data.pe_ratio < min_pe:
                continue
            if max_pe is not None and data.pe_ratio > max_pe:
                continue

            filtered.append(symbol)

        return filtered

    def filter_by_dividend(
        self,
        symbols: List[str],
        min_yield: float = 0.0,
    ) -> List[str]:
        """
        Filter symbols by dividend yield

        Args:
            symbols: List of stock symbols
            min_yield: Minimum dividend yield percentage

        Returns:
            Filtered list of symbols
        """
        filtered = []
        for symbol in symbols:
            data = self.get_fundamentals(symbol)
            if data.dividend_yield is None or data.dividend_yield >= min_yield:
                filtered.append(symbol)

        return filtered

    def filter_by_quality(
        self,
        symbols: List[str],
        min_score: float = 50.0,
    ) -> List[str]:
        """
        Filter symbols by fundamental quality score

        Args:
            symbols: List of stock symbols
            min_score: Minimum quality score (0-100)

        Returns:
            Filtered list of symbols
        """
        filtered = []
        for symbol in symbols:
            data = self.get_fundamentals(symbol)
            if data.quality_score >= min_score:
                filtered.append(symbol)

        return filtered

    def get_quality_adjustment(
        self,
        symbol: str,
    ) -> float:
        """
        Get position size adjustment based on fundamental quality

        Args:
            symbol: Stock symbol

        Returns:
            Adjustment multiplier (0.5 to 1.5)
        """
        data = self.get_fundamentals(symbol)
        score = data.quality_score

        # Map score to adjustment
        # 0-30: 0.5-0.8x, 30-70: 0.8-1.2x, 70-100: 1.2-1.5x
        if score < 30:
            return 0.5 + (score / 30) * 0.3
        elif score < 70:
            return 0.8 + ((score - 30) / 40) * 0.4
        else:
            return 1.2 + ((score - 70) / 30) * 0.3

    def rank_by_value(
        self,
        symbols: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Rank symbols by value metrics

        Args:
            symbols: List of stock symbols

        Returns:
            List of (symbol, value_score) tuples, sorted by score
        """
        scores = []
        for symbol in symbols:
            data = self.get_fundamentals(symbol)
            value_score = 50  # Base score

            # Low P/E is better
            if data.pe_ratio is not None and data.pe_ratio > 0:
                pe_score = max(0, 25 - data.pe_ratio) / 25 * 25
                value_score += pe_score

            # Low PBR is better
            if data.pbr is not None and data.pbr > 0:
                pbr_score = max(0, 3 - data.pbr) / 3 * 15
                value_score += pbr_score

            # High dividend is better
            if data.dividend_yield is not None:
                div_score = min(data.dividend_yield, 5) / 5 * 10
                value_score += div_score

            scores.append((symbol, value_score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def rank_by_growth(
        self,
        symbols: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Rank symbols by growth metrics

        Args:
            symbols: List of stock symbols

        Returns:
            List of (symbol, growth_score) tuples, sorted by score
        """
        scores = []
        for symbol in symbols:
            data = self.get_fundamentals(symbol)
            growth_score = 50  # Base score

            # High ROE is better for growth
            if data.roe is not None:
                roe_score = min(data.roe, 25) / 25 * 30
                growth_score += roe_score

            # Lower debt for growth sustainability
            if data.debt_to_equity is not None:
                debt_score = max(0, 2 - data.debt_to_equity) / 2 * 10
                growth_score += debt_score

            # Profit margin indicates business quality
            if data.profit_margin is not None:
                margin_score = min(data.profit_margin, 20) / 20 * 10
                growth_score += margin_score

            scores.append((symbol, growth_score))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_composite_score(
        self,
        symbol: str,
        value_weight: float = 0.4,
        growth_weight: float = 0.3,
        quality_weight: float = 0.3,
    ) -> float:
        """
        Get composite fundamental score

        Args:
            symbol: Stock symbol
            value_weight: Weight for value metrics
            growth_weight: Weight for growth metrics
            quality_weight: Weight for quality metrics

        Returns:
            Composite score (0-100)
        """
        data = self.get_fundamentals(symbol)

        # Value score
        value_score = 50
        if data.pe_ratio is not None and data.pe_ratio > 0:
            value_score += max(0, 25 - data.pe_ratio) / 25 * 25
        if data.pbr is not None and data.pbr > 0:
            value_score += max(0, 3 - data.pbr) / 3 * 15
        if data.dividend_yield is not None:
            value_score += min(data.dividend_yield, 5) / 5 * 10

        # Growth score
        growth_score = 50
        if data.roe is not None:
            growth_score += min(data.roe, 25) / 25 * 30
        if data.profit_margin is not None:
            growth_score += min(data.profit_margin, 20) / 20 * 20

        # Quality score (from FundamentalData property)
        quality_score = data.quality_score

        # Weighted composite
        composite = (
            value_score * value_weight +
            growth_score * growth_weight +
            quality_score * quality_weight
        )

        return min(100, max(0, composite))

    def get_all_scores(
        self,
        symbols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Get all fundamental scores for multiple symbols

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbols to score dictionaries
        """
        results = {}
        for symbol in symbols:
            data = self.get_fundamentals(symbol)
            results[symbol] = {
                "quality_score": data.quality_score,
                "composite_score": self.get_composite_score(symbol),
                "pe_ratio": data.pe_ratio,
                "pbr": data.pbr,
                "dividend_yield": data.dividend_yield,
                "roe": data.roe,
                "adjustment": self.get_quality_adjustment(symbol),
            }
        return results
