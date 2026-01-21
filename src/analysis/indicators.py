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

        # Momentum indicators
        df = self.add_rsi(df)
        df = self.add_stochastic(df)

        # Volatility indicators
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)

        # Volume indicators
        df = self.add_volume_indicators(df)
        df = self.add_obv(df)

        # Candlestick patterns
        df = self.add_candlestick_patterns(df)

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

    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick pattern recognition

        Adds columns:
        - Candle_Body: Body size (Close - Open)
        - Candle_Body_Pct: Body as percentage of range
        - Upper_Shadow: Upper shadow size
        - Lower_Shadow: Lower shadow size
        - Pattern_Doji: Doji pattern detected (1/-1/0)
        - Pattern_Hammer: Hammer/Hanging Man pattern (1/-1/0)
        - Pattern_Engulfing: Bullish/Bearish engulfing (1/-1/0)
        - Pattern_PinBar: Pin bar pattern (1/-1/0)
        - Pattern_ThreeSoldiers: Three white soldiers (1) or three black crows (-1)
        - Pattern_MorningStar: Morning/Evening star (1/-1/0)
        - Pattern_Score: Combined pattern score (-100 to 100)
        """
        df = df.copy()

        # Calculate basic candle components
        df["Candle_Body"] = df["Close"] - df["Open"]
        df["Candle_Range"] = df["High"] - df["Low"]
        df["Candle_Body_Pct"] = np.where(
            df["Candle_Range"] != 0,
            np.abs(df["Candle_Body"]) / df["Candle_Range"],
            0
        )
        df["Upper_Shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["Lower_Shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

        # Initialize pattern columns
        df["Pattern_Doji"] = 0
        df["Pattern_Hammer"] = 0
        df["Pattern_Engulfing"] = 0
        df["Pattern_PinBar"] = 0
        df["Pattern_ThreeSoldiers"] = 0
        df["Pattern_MorningStar"] = 0

        # Doji: Very small body (< 10% of range)
        doji_threshold = 0.10
        df["Pattern_Doji"] = np.where(
            df["Candle_Body_Pct"] < doji_threshold,
            np.where(df["Close"] > df["Open"], 1, -1),  # Slight bullish/bearish bias
            0
        )

        # Hammer/Hanging Man: Small body at top, long lower shadow
        # Hammer (bullish): After downtrend, body at top, lower shadow >= 2x body
        # Hanging Man (bearish): After uptrend, body at top, lower shadow >= 2x body
        body_abs = np.abs(df["Candle_Body"])
        is_small_body = df["Candle_Body_Pct"] < 0.35
        long_lower_shadow = df["Lower_Shadow"] >= 2 * body_abs
        small_upper_shadow = df["Upper_Shadow"] < body_abs * 0.5

        hammer_condition = is_small_body & long_lower_shadow & small_upper_shadow

        # Determine trend context using 5-day SMA
        prior_trend = df["Close"].rolling(5).mean().diff()
        is_downtrend = prior_trend < 0
        is_uptrend = prior_trend > 0

        df["Pattern_Hammer"] = np.where(
            hammer_condition & is_downtrend, 1,  # Bullish hammer
            np.where(hammer_condition & is_uptrend, -1, 0)  # Bearish hanging man
        )

        # Inverted Hammer / Shooting Star: Small body at bottom, long upper shadow
        long_upper_shadow = df["Upper_Shadow"] >= 2 * body_abs
        small_lower_shadow = df["Lower_Shadow"] < body_abs * 0.5
        inverted_condition = is_small_body & long_upper_shadow & small_lower_shadow

        # Add inverted patterns to hammer column
        df["Pattern_Hammer"] = np.where(
            inverted_condition & is_downtrend, 1,  # Inverted hammer (bullish)
            np.where(inverted_condition & is_uptrend, -1,  # Shooting star (bearish)
                     df["Pattern_Hammer"])
        )

        # Engulfing Pattern: Current candle body engulfs previous candle body
        prev_body = df["Candle_Body"].shift(1)
        prev_open = df["Open"].shift(1)
        prev_close = df["Close"].shift(1)

        # Bullish engulfing: Previous bearish, current bullish and engulfs
        bullish_engulfing = (
            (prev_body < 0) &  # Previous was bearish
            (df["Candle_Body"] > 0) &  # Current is bullish
            (df["Open"] < prev_close) &  # Current open below previous close
            (df["Close"] > prev_open)  # Current close above previous open
        )

        # Bearish engulfing: Previous bullish, current bearish and engulfs
        bearish_engulfing = (
            (prev_body > 0) &  # Previous was bullish
            (df["Candle_Body"] < 0) &  # Current is bearish
            (df["Open"] > prev_close) &  # Current open above previous close
            (df["Close"] < prev_open)  # Current close below previous open
        )

        df["Pattern_Engulfing"] = np.where(
            bullish_engulfing, 1,
            np.where(bearish_engulfing, -1, 0)
        )

        # Pin Bar: Long tail, small body at one end
        pin_bar_body_max = 0.25
        pin_bar_shadow_min = 2.5
        is_pin_body = df["Candle_Body_Pct"] < pin_bar_body_max

        bullish_pin = is_pin_body & (df["Lower_Shadow"] >= pin_bar_shadow_min * body_abs)
        bearish_pin = is_pin_body & (df["Upper_Shadow"] >= pin_bar_shadow_min * body_abs)

        df["Pattern_PinBar"] = np.where(
            bullish_pin & is_downtrend, 1,
            np.where(bearish_pin & is_uptrend, -1, 0)
        )

        # Three White Soldiers / Three Black Crows
        for i in range(2, len(df)):
            # Check for three consecutive bullish candles with higher closes
            if (df["Candle_Body"].iloc[i] > 0 and
                df["Candle_Body"].iloc[i-1] > 0 and
                df["Candle_Body"].iloc[i-2] > 0 and
                df["Close"].iloc[i] > df["Close"].iloc[i-1] > df["Close"].iloc[i-2] and
                df["Open"].iloc[i] > df["Open"].iloc[i-1] > df["Open"].iloc[i-2]):
                df.iloc[i, df.columns.get_loc("Pattern_ThreeSoldiers")] = 1

            # Check for three consecutive bearish candles with lower closes
            if (df["Candle_Body"].iloc[i] < 0 and
                df["Candle_Body"].iloc[i-1] < 0 and
                df["Candle_Body"].iloc[i-2] < 0 and
                df["Close"].iloc[i] < df["Close"].iloc[i-1] < df["Close"].iloc[i-2] and
                df["Open"].iloc[i] < df["Open"].iloc[i-1] < df["Open"].iloc[i-2]):
                df.iloc[i, df.columns.get_loc("Pattern_ThreeSoldiers")] = -1

        # Morning Star / Evening Star (3-candle reversal pattern)
        for i in range(2, len(df)):
            first_body = df["Candle_Body"].iloc[i-2]
            second_body_pct = df["Candle_Body_Pct"].iloc[i-1]
            third_body = df["Candle_Body"].iloc[i]

            # Morning Star: Large bearish, small body (star), large bullish
            if (first_body < 0 and
                abs(first_body) > df["Candle_Range"].iloc[i-2] * 0.5 and
                second_body_pct < 0.3 and
                third_body > 0 and
                third_body > df["Candle_Range"].iloc[i] * 0.5 and
                df["Close"].iloc[i] > (df["Open"].iloc[i-2] + df["Close"].iloc[i-2]) / 2):
                df.iloc[i, df.columns.get_loc("Pattern_MorningStar")] = 1

            # Evening Star: Large bullish, small body (star), large bearish
            if (first_body > 0 and
                first_body > df["Candle_Range"].iloc[i-2] * 0.5 and
                second_body_pct < 0.3 and
                third_body < 0 and
                abs(third_body) > df["Candle_Range"].iloc[i] * 0.5 and
                df["Close"].iloc[i] < (df["Open"].iloc[i-2] + df["Close"].iloc[i-2]) / 2):
                df.iloc[i, df.columns.get_loc("Pattern_MorningStar")] = -1

        # Combined Pattern Score (-100 to 100)
        # Weight different patterns by reliability
        pattern_weights = {
            "Engulfing": 30,
            "ThreeSoldiers": 25,
            "MorningStar": 25,
            "Hammer": 15,
            "PinBar": 10,
            "Doji": 5,
        }

        df["Pattern_Score"] = (
            df["Pattern_Engulfing"] * pattern_weights["Engulfing"] +
            df["Pattern_ThreeSoldiers"] * pattern_weights["ThreeSoldiers"] +
            df["Pattern_MorningStar"] * pattern_weights["MorningStar"] +
            df["Pattern_Hammer"] * pattern_weights["Hammer"] +
            df["Pattern_PinBar"] * pattern_weights["PinBar"] +
            df["Pattern_Doji"] * pattern_weights["Doji"]
        )

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

            # Candlestick Patterns
            "Pattern_Doji": latest.get("Pattern_Doji"),
            "Pattern_Hammer": latest.get("Pattern_Hammer"),
            "Pattern_Engulfing": latest.get("Pattern_Engulfing"),
            "Pattern_PinBar": latest.get("Pattern_PinBar"),
            "Pattern_ThreeSoldiers": latest.get("Pattern_ThreeSoldiers"),
            "Pattern_MorningStar": latest.get("Pattern_MorningStar"),
            "Pattern_Score": latest.get("Pattern_Score"),
        }
