"""
Technical indicator calculations for swing trading analysis
"""
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
except ImportError:
    # Fallback to manual calculations if pandas_ta not available
    ta = None

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import INDICATOR_PARAMS

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for stock data
    Designed for swing trading analysis (3-15 day holding periods)
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize with indicator parameters

        Args:
            params: Dictionary of indicator parameters (uses defaults if not provided)
        """
        self.params = params if params is not None else INDICATOR_PARAMS

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators

        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)

        Returns:
            DataFrame with all indicators added as new columns
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_all")
            return df

        df = df.copy()

        # Trend indicators
        df = self.add_moving_averages(df)
        df = self.add_macd(df)
        df = self.add_adx(df)  # ADX for trend strength

        # Momentum indicators
        df = self.add_rsi(df)
        df = self.add_stochastic(df)

        # Volatility indicators
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)

        # Volume indicators
        df = self.add_volume_indicators(df)
        df = self.add_obv(df)

        logger.debug(f"Calculated all indicators. DataFrame now has {len(df.columns)} columns")
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Simple Moving Averages (SMA)

        Adds columns:
        - SMA_5: 5-day short-term trend
        - SMA_25: 25-day medium-term trend
        - SMA_75: 75-day long-term trend
        """
        df = df.copy()

        df["SMA_5"] = df["Close"].rolling(window=self.params["sma_short"]).mean()
        df["SMA_25"] = df["Close"].rolling(window=self.params["sma_medium"]).mean()
        df["SMA_75"] = df["Close"].rolling(window=self.params["sma_long"]).mean()

        # Golden Cross / Death Cross signals
        df["Golden_Cross"] = (df["SMA_5"] > df["SMA_25"]) & (
            df["SMA_5"].shift(1) <= df["SMA_25"].shift(1)
        )
        df["Death_Cross"] = (df["SMA_5"] < df["SMA_25"]) & (
            df["SMA_5"].shift(1) >= df["SMA_25"].shift(1)
        )

        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence)

        Adds columns:
        - MACD: MACD line
        - MACD_Signal: Signal line
        - MACD_Histogram: Histogram (MACD - Signal)
        """
        df = df.copy()

        if ta is not None:
            macd = ta.macd(
                df["Close"],
                fast=self.params["macd_fast"],
                slow=self.params["macd_slow"],
                signal=self.params["macd_signal"],
            )
            df["MACD"] = macd[f"MACD_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"]
            df["MACD_Signal"] = macd[f"MACDs_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"]
            df["MACD_Histogram"] = macd[f"MACDh_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"]
        else:
            # Manual calculation
            exp1 = df["Close"].ewm(span=self.params["macd_fast"], adjust=False).mean()
            exp2 = df["Close"].ewm(span=self.params["macd_slow"], adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["MACD_Signal"] = df["MACD"].ewm(span=self.params["macd_signal"], adjust=False).mean()
            df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        return df

    def add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ADX (Average Directional Index) for trend strength measurement

        Adds columns:
        - ADX: 14-period ADX (0-100, trend strength)
        - Plus_DI: +DI line
        - Minus_DI: -DI line
        - ADX_Trend: 1 if strong trend (ADX > 25), 0 otherwise

        Academic basis:
        - ADX > 25 indicates a strong trend (Wilder, 1978)
        - ADX > 50 indicates very strong trend
        - ADX < 20 indicates weak/no trend
        """
        df = df.copy()
        period = self.params.get("adx_period", 14)

        if ta is not None:
            try:
                adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=period)
                df["ADX"] = adx_df[f"ADX_{period}"]
                df["Plus_DI"] = adx_df[f"DMP_{period}"]
                df["Minus_DI"] = adx_df[f"DMN_{period}"]
            except Exception as e:
                logger.warning(f"pandas_ta ADX calculation failed: {e}")
                df = self._calculate_adx_manual(df, period)
        else:
            df = self._calculate_adx_manual(df, period)

        # ADX trend strength indicator
        df["ADX_Trend"] = np.where(df["ADX"] > 25, 1, 0)

        return df

    def _calculate_adx_manual(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Manual ADX calculation"""
        # True Range
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        up_move = df["High"] - df["High"].shift()
        down_move = df["Low"].shift() - df["Low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values using Wilder's smoothing
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr

        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        df["ADX"] = adx
        df["Plus_DI"] = plus_di
        df["Minus_DI"] = minus_di

        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI (Relative Strength Index)

        Adds column:
        - RSI_14: 14-period RSI
        """
        df = df.copy()

        if ta is not None:
            df["RSI_14"] = ta.rsi(df["Close"], length=self.params["rsi_period"])
        else:
            # Manual calculation
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.params["rsi_period"]).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.params["rsi_period"]).mean()
            # Avoid division by zero
            rs = np.where(loss != 0, gain / loss, np.inf)
            df["RSI_14"] = np.where(
                loss == 0,
                100.0,  # If no losses, RSI = 100
                100 - (100 / (1 + rs))
            )

        return df

    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Stochastic Oscillator

        Adds columns:
        - Stoch_K: %K line
        - Stoch_D: %D line (signal)
        """
        df = df.copy()

        if ta is not None:
            stoch = ta.stoch(
                df["High"],
                df["Low"],
                df["Close"],
                k=self.params["stoch_k"],
                d=self.params["stoch_d"],
            )
            df["Stoch_K"] = stoch[f"STOCHk_{self.params['stoch_k']}_{self.params['stoch_d']}_3"]
            df["Stoch_D"] = stoch[f"STOCHd_{self.params['stoch_k']}_{self.params['stoch_d']}_3"]
        else:
            # Manual calculation
            low_min = df["Low"].rolling(window=self.params["stoch_k"]).min()
            high_max = df["High"].rolling(window=self.params["stoch_k"]).max()
            denominator = high_max - low_min
            # Avoid division by zero when high == low
            df["Stoch_K"] = np.where(
                denominator != 0,
                100 * (df["Close"] - low_min) / denominator,
                50.0  # Neutral value when range is zero
            )
            df["Stoch_D"] = df["Stoch_K"].rolling(window=self.params["stoch_d"]).mean()

        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Bands

        Adds columns:
        - BB_Middle: Middle band (20-day SMA)
        - BB_Upper: Upper band (Middle + 2*std)
        - BB_Lower: Lower band (Middle - 2*std)
        - BB_Width: Band width (Upper - Lower)
        - BB_Percent: Position within bands (0-100%)
        """
        df = df.copy()

        if ta is not None:
            bbands = ta.bbands(
                df["Close"],
                length=self.params["bb_period"],
                std=self.params["bb_std"],
            )
            df["BB_Upper"] = bbands[f"BBU_{self.params['bb_period']}_{self.params['bb_std']}"]
            df["BB_Middle"] = bbands[f"BBM_{self.params['bb_period']}_{self.params['bb_std']}"]
            df["BB_Lower"] = bbands[f"BBL_{self.params['bb_period']}_{self.params['bb_std']}"]
            df["BB_Width"] = bbands[f"BBB_{self.params['bb_period']}_{self.params['bb_std']}"]
            df["BB_Percent"] = bbands[f"BBP_{self.params['bb_period']}_{self.params['bb_std']}"]
        else:
            # Manual calculation
            df["BB_Middle"] = df["Close"].rolling(window=self.params["bb_period"]).mean()
            std = df["Close"].rolling(window=self.params["bb_period"]).std()
            df["BB_Upper"] = df["BB_Middle"] + (self.params["bb_std"] * std)
            df["BB_Lower"] = df["BB_Middle"] - (self.params["bb_std"] * std)
            df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
            # Avoid division by zero when BB_Width is zero
            bb_range = df["BB_Upper"] - df["BB_Lower"]
            df["BB_Percent"] = np.where(
                bb_range != 0,
                (df["Close"] - df["BB_Lower"]) / bb_range,
                0.5  # Neutral value when bands are equal
            )

        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ATR (Average True Range)

        Adds columns:
        - ATR: 14-period ATR
        - ATR_Percent: ATR as percentage of close price
        """
        df = df.copy()

        if ta is not None:
            df["ATR"] = ta.atr(
                df["High"],
                df["Low"],
                df["Close"],
                length=self.params["atr_period"],
            )
        else:
            # Manual calculation
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift())
            low_close = np.abs(df["Low"] - df["Close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["ATR"] = true_range.rolling(window=self.params["atr_period"]).mean()

        # ATR as percentage of price (useful for position sizing)
        df["ATR_Percent"] = (df["ATR"] / df["Close"]) * 100

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators

        Adds columns:
        - Volume_MA: 20-day volume moving average
        - Volume_Ratio: Current volume / average volume
        """
        df = df.copy()

        df["Volume_MA"] = df["Volume"].rolling(window=self.params["volume_ma_period"]).mean()
        # Avoid division by zero when Volume_MA is zero
        df["Volume_Ratio"] = np.where(
            df["Volume_MA"] != 0,
            df["Volume"] / df["Volume_MA"],
            1.0  # Neutral value when no average available
        )

        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add OBV (On-Balance Volume)

        Adds columns:
        - OBV: Cumulative OBV
        - OBV_MA: 10-day moving average of OBV
        - OBV_Trend: 1 if OBV rising, -1 if falling, 0 if flat
        """
        df = df.copy()

        if ta is not None:
            df["OBV"] = ta.obv(df["Close"], df["Volume"])
        else:
            # Manual calculation
            obv = [0]
            for i in range(1, len(df)):
                if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                    obv.append(obv[-1] + df["Volume"].iloc[i])
                elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                    obv.append(obv[-1] - df["Volume"].iloc[i])
                else:
                    obv.append(obv[-1])
            df["OBV"] = obv

        # OBV trend
        df["OBV_MA"] = df["OBV"].rolling(window=self.params["obv_ma_period"]).mean()
        df["OBV_Trend"] = np.where(df["OBV"] > df["OBV_MA"], 1,
                                    np.where(df["OBV"] < df["OBV_MA"], -1, 0))

        return df

    def get_latest_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Get the most recent indicator values as a dictionary

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary of latest indicator values
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        return {
            # Price
            "Close": latest["Close"],

            # Trend
            "SMA_5": latest.get("SMA_5"),
            "SMA_25": latest.get("SMA_25"),
            "SMA_75": latest.get("SMA_75"),
            "MACD": latest.get("MACD"),
            "MACD_Signal": latest.get("MACD_Signal"),
            "MACD_Histogram": latest.get("MACD_Histogram"),
            "ADX": latest.get("ADX"),
            "Plus_DI": latest.get("Plus_DI"),
            "Minus_DI": latest.get("Minus_DI"),
            "ADX_Trend": latest.get("ADX_Trend"),

            # Momentum
            "RSI_14": latest.get("RSI_14"),
            "Stoch_K": latest.get("Stoch_K"),
            "Stoch_D": latest.get("Stoch_D"),

            # Volatility
            "BB_Upper": latest.get("BB_Upper"),
            "BB_Middle": latest.get("BB_Middle"),
            "BB_Lower": latest.get("BB_Lower"),
            "BB_Width": latest.get("BB_Width"),
            "BB_Percent": latest.get("BB_Percent"),
            "ATR": latest.get("ATR"),
            "ATR_Percent": latest.get("ATR_Percent"),

            # Volume
            "Volume": latest["Volume"],
            "Volume_MA": latest.get("Volume_MA"),
            "Volume_Ratio": latest.get("Volume_Ratio"),
            "OBV": latest.get("OBV"),
            "OBV_Trend": latest.get("OBV_Trend"),
        }
