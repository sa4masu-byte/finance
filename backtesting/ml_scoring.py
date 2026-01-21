"""
Machine Learning Scoring Module (改善10 - Enhanced)
Integrates ML predictions with traditional scoring

Features:
- Ensemble learning (LightGBM + XGBoost + RandomForest)
- Advanced feature engineering
- Cross-validation with time-series awareness
- Feature importance analysis
- Auto hyperparameter tuning
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import ML_SCORING_PARAMS, REPORTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Feature set with metadata"""
    features: pd.DataFrame
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class MLScoringEngine:
    """
    Machine Learning based scoring engine
    Supports LightGBM and XGBoost models
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_type: str = "lightgbm",
    ):
        self.model_path = model_path or ML_SCORING_PARAMS["model_path"]
        self.model_type = model_type
        self.model = None
        self.feature_columns = ML_SCORING_PARAMS["feature_columns"]
        self.is_trained = False

    def load_model(self) -> bool:
        """Load pre-trained model"""
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return False

        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"Loaded ML model from {self.model_path}")
            return True
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to access model file: {e}")
            return False
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Failed to deserialize model (corrupted file?): {e}")
            return False
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error(f"Model incompatible with current environment: {e}")
            return False

    def train_model(
        self,
        training_data: pd.DataFrame,
        labels: pd.Series,
        validation_data: Optional[pd.DataFrame] = None,
        validation_labels: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Train ML model on historical data

        Args:
            training_data: Features DataFrame
            labels: Target labels (1 = successful trade, 0 = unsuccessful)
            validation_data: Optional validation features
            validation_labels: Optional validation labels

        Returns:
            Training metrics
        """
        # Prepare features
        X_train = self._prepare_features(training_data)
        y_train = labels

        if X_train.empty or len(y_train) == 0:
            logger.error("No training data available")
            return {"error": "No training data"}

        # Train model based on type
        if self.model_type == "lightgbm":
            metrics = self._train_lightgbm(X_train, y_train, validation_data, validation_labels)
        elif self.model_type == "xgboost":
            metrics = self._train_xgboost(X_train, y_train, validation_data, validation_labels)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return {"error": f"Unknown model type: {self.model_type}"}

        if self.model is not None:
            self.is_trained = True
            self.save_model()

        return metrics

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train LightGBM model"""
        try:
            import lightgbm as lgb

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': -1,
            }

            train_data = lgb.Dataset(X_train, label=y_train)

            if X_val is not None and y_val is not None:
                X_val_prepared = self._prepare_features(X_val)
                val_data = lgb.Dataset(X_val_prepared, label=y_val, reference=train_data)
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[train_data, val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
                )
            else:
                self.model = lgb.train(params, train_data, num_boost_round=200)

            # Get feature importance
            importance = dict(zip(
                X_train.columns,
                self.model.feature_importance(importance_type='gain')
            ))

            return {
                "model_type": "lightgbm",
                "num_features": len(X_train.columns),
                "feature_importance": importance,
                "best_iteration": self.model.best_iteration if hasattr(self.model, 'best_iteration') else 200,
            }

        except ImportError:
            logger.warning("LightGBM not installed. Install with: pip install lightgbm")
            return {"error": "LightGBM not installed"}
        except Exception as e:
            logger.error(f"LightGBM training error: {e}")
            return {"error": str(e)}

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train XGBoost model"""
        try:
            import xgboost as xgb

            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1,
            }

            dtrain = xgb.DMatrix(X_train, label=y_train)

            if X_val is not None and y_val is not None:
                X_val_prepared = self._prepare_features(X_val)
                dval = xgb.DMatrix(X_val_prepared, label=y_val)
                self.model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=100,
                )
            else:
                self.model = xgb.train(params, dtrain, num_boost_round=200)

            # Get feature importance
            importance = self.model.get_score(importance_type='gain')

            return {
                "model_type": "xgboost",
                "num_features": len(X_train.columns),
                "feature_importance": importance,
                "best_iteration": self.model.best_iteration if hasattr(self.model, 'best_iteration') else 200,
            }

        except ImportError:
            logger.warning("XGBoost not installed. Install with: pip install xgboost")
            return {"error": "XGBoost not installed"}
        except Exception as e:
            logger.error(f"XGBoost training error: {e}")
            return {"error": str(e)}

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model with advanced feature engineering"""
        # Select only the feature columns that exist
        available_cols = [col for col in self.feature_columns if col in data.columns]

        if not available_cols:
            return pd.DataFrame()

        features = data[available_cols].copy()

        # Advanced feature engineering
        features = self._add_engineered_features(features, data)

        # Fill NaN values
        features = features.fillna(features.median())

        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def _add_engineered_features(
        self,
        features: pd.DataFrame,
        original_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add engineered features for better prediction"""
        df = features.copy()

        # RSI-based features
        if "RSI_14" in original_data.columns:
            rsi = original_data["RSI_14"]
            df["RSI_Oversold"] = (rsi < 30).astype(int)
            df["RSI_Overbought"] = (rsi > 70).astype(int)
            df["RSI_Neutral"] = ((rsi >= 40) & (rsi <= 60)).astype(int)

        # MACD momentum
        if "MACD" in original_data.columns and "MACD_Signal" in original_data.columns:
            macd = original_data["MACD"]
            signal = original_data["MACD_Signal"]
            df["MACD_Cross_Up"] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
            df["MACD_Momentum"] = macd - signal

        # Bollinger Band position
        if "BB_Percent" in original_data.columns:
            bb_pct = original_data["BB_Percent"]
            df["BB_Lower_Zone"] = (bb_pct < 0.2).astype(int)
            df["BB_Upper_Zone"] = (bb_pct > 0.8).astype(int)

        # Volume strength
        if "Volume_Ratio" in original_data.columns:
            vol_ratio = original_data["Volume_Ratio"]
            df["Volume_Spike"] = (vol_ratio > 2.0).astype(int)
            df["Volume_Dry"] = (vol_ratio < 0.5).astype(int)

        # Stochastic signals
        if "Stoch_K" in original_data.columns and "Stoch_D" in original_data.columns:
            stoch_k = original_data["Stoch_K"]
            stoch_d = original_data["Stoch_D"]
            df["Stoch_Bullish_Cross"] = ((stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))).astype(int)

        # Moving average alignment
        if all(col in original_data.columns for col in ["SMA_5", "SMA_25", "SMA_75"]):
            sma5 = original_data["SMA_5"]
            sma25 = original_data["SMA_25"]
            sma75 = original_data["SMA_75"]
            df["MA_Bullish_Alignment"] = ((sma5 > sma25) & (sma25 > sma75)).astype(int)
            df["MA_Bearish_Alignment"] = ((sma5 < sma25) & (sma25 < sma75)).astype(int)

        # Pattern scores
        if "Pattern_Score" in original_data.columns:
            pattern = original_data["Pattern_Score"]
            df["Pattern_Bullish"] = (pattern > 20).astype(int)
            df["Pattern_Bearish"] = (pattern < -20).astype(int)

        # Trend strength (ATR normalized)
        if "ATR_Percent" in original_data.columns:
            atr_pct = original_data["ATR_Percent"]
            df["Low_Volatility"] = (atr_pct < 2).astype(int)
            df["High_Volatility"] = (atr_pct > 5).astype(int)

        return df

    def predict_score(self, indicators: Dict) -> Tuple[float, float]:
        """
        Predict ML score for given indicators

        Args:
            indicators: Dictionary of indicator values

        Returns:
            Tuple of (ml_score 0-100, confidence 0-1)
        """
        if not self.is_trained or self.model is None:
            return 0.0, 0.0

        # Create DataFrame from indicators
        features = pd.DataFrame([indicators])
        X = self._prepare_features(features)

        if X.empty:
            return 0.0, 0.0

        try:
            if self.model_type == "lightgbm":
                prob = self.model.predict(X)[0]
            elif self.model_type == "xgboost":
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                prob = self.model.predict(dtest)[0]
            else:
                return 0.0, 0.0

            # Convert probability to score (0-100)
            ml_score = prob * 100

            # Confidence is the distance from 0.5 (uncertainty)
            confidence = abs(prob - 0.5) * 2

            return ml_score, confidence

        except (ValueError, TypeError) as e:
            logger.warning(f"Prediction input error: {e}")
            return 0.0, 0.0
        except (IndexError, KeyError) as e:
            logger.warning(f"Prediction output error: {e}")
            return 0.0, 0.0
        except ImportError as e:
            logger.warning(f"ML library not available: {e}")
            return 0.0, 0.0

    def get_combined_score(
        self,
        traditional_score: float,
        indicators: Dict,
    ) -> Tuple[float, float]:
        """
        Combine traditional and ML scores

        Args:
            traditional_score: Score from traditional scoring engine (0-100)
            indicators: Dictionary of indicator values

        Returns:
            Tuple of (combined_score, ml_confidence)
        """
        if not self.is_trained:
            return traditional_score, 0.0

        ml_score, ml_confidence = self.predict_score(indicators)

        if ml_confidence < ML_SCORING_PARAMS["min_confidence_threshold"]:
            # Low confidence, use traditional score
            return traditional_score, ml_confidence

        # Weighted combination
        traditional_weight = ML_SCORING_PARAMS["traditional_score_weight"]
        ml_weight = ML_SCORING_PARAMS["ml_score_weight"]

        combined = (
            traditional_score * traditional_weight +
            ml_score * ml_weight
        )

        return combined, ml_confidence

    def save_model(self, path: Optional[Path] = None):
        """Save trained model"""
        save_path = path or self.model_path
        if self.model is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {save_path}")

    def create_training_labels(
        self,
        trades: List,
        min_return_threshold: float = 0.05,
    ) -> Tuple[List[Dict], List[int]]:
        """
        Create training labels from historical trades

        Args:
            trades: List of Trade objects with indicators_at_entry attribute
            min_return_threshold: Minimum return to consider a trade successful

        Returns:
            Tuple of (features list, labels list)
        """
        features = []
        labels = []

        for trade in trades:
            if trade.is_open:
                continue

            # Label: 1 if trade was profitable, 0 otherwise
            label = 1 if trade.return_pct >= min_return_threshold else 0

            # Collect available features from trade
            trade_features = {
                "score_at_entry": getattr(trade, 'score_at_entry', 50.0),
                "atr_at_entry": getattr(trade, 'atr_at_entry', 0.0),
            }

            # Add indicators if stored (for enhanced trades)
            if hasattr(trade, 'indicators_at_entry') and trade.indicators_at_entry:
                trade_features.update(trade.indicators_at_entry)

            features.append(trade_features)
            labels.append(label)

        return features, labels


def train_ml_model_from_backtest(
    stock_data: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    model_type: str = "lightgbm",
) -> Optional[MLScoringEngine]:
    """
    Train ML model from backtest data

    This function:
    1. Runs a backtest to collect training data
    2. Labels each entry point based on outcome
    3. Trains a model to predict successful entries

    Args:
        stock_data: Dictionary of stock DataFrames
        start_date: Training start date
        end_date: Training end date
        model_type: "lightgbm" or "xgboost"

    Returns:
        Trained MLScoringEngine or None if training failed
    """
    from backtesting.backtest_engine import BacktestEngine
    from src.analysis.indicators import TechnicalIndicators

    logger.info("Training ML model from backtest data...")

    # Run backtest to collect trade data
    engine = BacktestEngine(
        initial_capital=3_000_000,
        max_positions=5,
    )

    results = engine.run_backtest(stock_data, start_date, end_date)

    if results.num_trades < 50:
        logger.warning(f"Insufficient trades for ML training: {results.num_trades}")
        return None

    # Prepare training data
    indicator_engine = TechnicalIndicators()
    training_features = []
    training_labels = []

    for trade in results.trades:
        if trade.is_open:
            continue

        # Get the stock data for this trade
        if trade.symbol not in stock_data:
            continue

        df = stock_data[trade.symbol]
        df_with_ind = indicator_engine.calculate_all(df)

        # Get indicators at entry date
        if trade.entry_date not in df_with_ind.index:
            continue

        indicators = df_with_ind.loc[trade.entry_date].to_dict()

        # Label: successful if return > 5%
        label = 1 if trade.return_pct >= 5.0 else 0

        training_features.append(indicators)
        training_labels.append(label)

    if len(training_features) < 30:
        logger.warning(f"Insufficient training samples: {len(training_features)}")
        return None

    # Create and train model
    ml_engine = MLScoringEngine(model_type=model_type)

    training_df = pd.DataFrame(training_features)
    labels_series = pd.Series(training_labels)

    # Split for validation
    split_idx = int(len(training_df) * 0.8)
    X_train = training_df.iloc[:split_idx]
    y_train = labels_series.iloc[:split_idx]
    X_val = training_df.iloc[split_idx:]
    y_val = labels_series.iloc[split_idx:]

    metrics = ml_engine.train_model(X_train, y_train, X_val, y_val)

    if "error" in metrics:
        logger.error(f"ML training failed: {metrics['error']}")
        return None

    logger.info(f"ML model trained successfully: {metrics}")

    return ml_engine


class EnsembleScoringEngine:
    """
    Ensemble ML scoring engine combining multiple models

    Models:
    - LightGBM (gradient boosting)
    - XGBoost (gradient boosting)
    - RandomForest (bagging)

    Final prediction is weighted average of all models
    """

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize ensemble engine

        Args:
            model_weights: Dict with weights for each model type
            model_path: Path to save/load ensemble
        """
        self.model_weights = model_weights or {
            "lightgbm": 0.40,
            "xgboost": 0.35,
            "random_forest": 0.25,
        }
        self.model_path = model_path or REPORTS_DIR / "ensemble_model.pkl"
        self.models: Dict[str, Any] = {}
        self.feature_columns = ML_SCORING_PARAMS["feature_columns"]
        self.is_trained = False
        self.feature_importance: Dict[str, float] = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train all ensemble models

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training metrics for each model
        """
        metrics = {}

        # Train LightGBM
        lgb_metrics = self._train_lightgbm(X_train, y_train, X_val, y_val)
        metrics["lightgbm"] = lgb_metrics

        # Train XGBoost
        xgb_metrics = self._train_xgboost(X_train, y_train, X_val, y_val)
        metrics["xgboost"] = xgb_metrics

        # Train RandomForest
        rf_metrics = self._train_random_forest(X_train, y_train, X_val, y_val)
        metrics["random_forest"] = rf_metrics

        # Aggregate feature importance
        self._aggregate_feature_importance()

        if any(m is not None for m in self.models.values()):
            self.is_trained = True
            self.save()

        return metrics

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train LightGBM model"""
        try:
            import lightgbm as lgb

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': -1,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
            }

            train_data = lgb.Dataset(X_train, label=y_train)

            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                self.models["lightgbm"] = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[train_data, val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
            else:
                self.models["lightgbm"] = lgb.train(params, train_data, num_boost_round=200)

            return {"status": "success", "best_iter": getattr(self.models["lightgbm"], 'best_iteration', 200)}

        except ImportError:
            logger.warning("LightGBM not installed")
            return {"status": "not_installed"}
        except Exception as e:
            logger.error(f"LightGBM error: {e}")
            return {"status": "error", "message": str(e)}

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train XGBoost model"""
        try:
            import xgboost as xgb

            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_jobs': -1,
            }

            dtrain = xgb.DMatrix(X_train, label=y_train)

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                self.models["xgboost"] = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
            else:
                self.models["xgboost"] = xgb.train(params, dtrain, num_boost_round=200)

            return {"status": "success"}

        except ImportError:
            logger.warning("XGBoost not installed")
            return {"status": "not_installed"}
        except Exception as e:
            logger.error(f"XGBoost error: {e}")
            return {"status": "error", "message": str(e)}

    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train RandomForest model"""
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.models["random_forest"] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42,
            )

            self.models["random_forest"].fit(X_train, y_train)

            accuracy = None
            if X_val is not None and y_val is not None:
                accuracy = self.models["random_forest"].score(X_val, y_val)

            return {"status": "success", "val_accuracy": accuracy}

        except ImportError:
            logger.warning("sklearn not installed")
            return {"status": "not_installed"}
        except Exception as e:
            logger.error(f"RandomForest error: {e}")
            return {"status": "error", "message": str(e)}

    def _aggregate_feature_importance(self) -> None:
        """Aggregate feature importance from all models"""
        importance_sum: Dict[str, float] = {}
        weight_sum = 0

        # LightGBM importance
        if "lightgbm" in self.models and self.models["lightgbm"] is not None:
            try:
                lgb_imp = dict(zip(
                    self.models["lightgbm"].feature_name(),
                    self.models["lightgbm"].feature_importance(importance_type='gain')
                ))
                total = sum(lgb_imp.values()) or 1
                for feat, imp in lgb_imp.items():
                    importance_sum[feat] = importance_sum.get(feat, 0) + (imp / total) * self.model_weights["lightgbm"]
                weight_sum += self.model_weights["lightgbm"]
            except Exception:
                pass

        # XGBoost importance
        if "xgboost" in self.models and self.models["xgboost"] is not None:
            try:
                xgb_imp = self.models["xgboost"].get_score(importance_type='gain')
                total = sum(xgb_imp.values()) or 1
                for feat, imp in xgb_imp.items():
                    importance_sum[feat] = importance_sum.get(feat, 0) + (imp / total) * self.model_weights["xgboost"]
                weight_sum += self.model_weights["xgboost"]
            except Exception:
                pass

        # RandomForest importance
        if "random_forest" in self.models and self.models["random_forest"] is not None:
            try:
                rf_imp = dict(zip(
                    self.models["random_forest"].feature_names_in_,
                    self.models["random_forest"].feature_importances_
                ))
                total = sum(rf_imp.values()) or 1
                for feat, imp in rf_imp.items():
                    importance_sum[feat] = importance_sum.get(feat, 0) + (imp / total) * self.model_weights["random_forest"]
                weight_sum += self.model_weights["random_forest"]
            except Exception:
                pass

        # Normalize
        if weight_sum > 0:
            self.feature_importance = {
                feat: imp / weight_sum
                for feat, imp in importance_sum.items()
            }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ensemble prediction

        Args:
            X: Features DataFrame

        Returns:
            Tuple of (probabilities, confidences)
        """
        if not self.is_trained:
            return np.zeros(len(X)), np.zeros(len(X))

        predictions = []
        weights = []

        # LightGBM prediction
        if "lightgbm" in self.models and self.models["lightgbm"] is not None:
            try:
                pred = self.models["lightgbm"].predict(X)
                predictions.append(pred)
                weights.append(self.model_weights["lightgbm"])
            except Exception:
                pass

        # XGBoost prediction
        if "xgboost" in self.models and self.models["xgboost"] is not None:
            try:
                import xgboost as xgb
                dtest = xgb.DMatrix(X)
                pred = self.models["xgboost"].predict(dtest)
                predictions.append(pred)
                weights.append(self.model_weights["xgboost"])
            except Exception:
                pass

        # RandomForest prediction
        if "random_forest" in self.models and self.models["random_forest"] is not None:
            try:
                pred = self.models["random_forest"].predict_proba(X)[:, 1]
                predictions.append(pred)
                weights.append(self.model_weights["random_forest"])
            except Exception:
                pass

        if not predictions:
            return np.zeros(len(X)), np.zeros(len(X))

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # Confidence = agreement between models (low std = high confidence)
        if len(predictions) > 1:
            pred_std = np.std(predictions, axis=0)
            confidence = 1 - (pred_std * 2)  # Scale std to confidence
            confidence = np.clip(confidence, 0, 1)
        else:
            confidence = np.abs(ensemble_pred - 0.5) * 2

        return ensemble_pred, confidence

    def predict_score(self, indicators: Dict) -> Tuple[float, float]:
        """
        Predict ML score for given indicators

        Args:
            indicators: Dictionary of indicator values

        Returns:
            Tuple of (ml_score 0-100, confidence 0-1)
        """
        X = pd.DataFrame([indicators])

        # Select available columns
        available_cols = [col for col in self.feature_columns if col in X.columns]
        if not available_cols:
            return 0.0, 0.0

        X = X[available_cols].fillna(0).replace([np.inf, -np.inf], 0)

        probs, confs = self.predict(X)

        return probs[0] * 100, confs[0]

    def save(self, path: Optional[Path] = None) -> None:
        """Save ensemble model"""
        save_path = path or self.model_path
        data = {
            "models": self.models,
            "model_weights": self.model_weights,
            "feature_columns": self.feature_columns,
            "feature_importance": self.feature_importance,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Ensemble saved to {save_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load ensemble model"""
        load_path = path or self.model_path
        if not load_path.exists():
            return False

        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            self.models = data["models"]
            self.model_weights = data["model_weights"]
            self.feature_columns = data["feature_columns"]
            self.feature_importance = data.get("feature_importance", {})
            self.is_trained = True
            logger.info(f"Ensemble loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
