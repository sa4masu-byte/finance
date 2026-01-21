"""
Machine Learning Scoring Module (改善10) - Enhanced with Ensemble Learning
Integrates ML predictions with traditional scoring using multiple models
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from abc import ABC, abstractmethod

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import ML_SCORING_PARAMS, REPORTS_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

ENSEMBLE_CONFIG = {
    "enabled": True,
    "models": ["lightgbm", "xgboost", "random_forest"],
    "voting_method": "soft",  # "hard" or "soft" (probability-weighted)
    "model_weights": {
        "lightgbm": 0.40,      # LightGBM gets highest weight (best on financial data)
        "xgboost": 0.35,       # XGBoost second
        "random_forest": 0.25,  # Random Forest for diversity
    },
    "min_models_for_prediction": 2,  # Minimum models that must agree
    "confidence_threshold": 0.60,     # Minimum average confidence
}


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
        """Prepare features for model"""
        # Select only the feature columns that exist
        available_cols = [col for col in self.feature_columns if col in data.columns]

        if not available_cols:
            return pd.DataFrame()

        features = data[available_cols].copy()

        # Fill NaN values
        features = features.fillna(features.median())

        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0)

        return features

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


class EnsembleMLScoringEngine:
    """
    Ensemble Machine Learning scoring engine
    Combines multiple models (LightGBM, XGBoost, RandomForest) for more robust predictions

    Benefits:
    - Reduced variance through model averaging
    - Better generalization on unseen data
    - More stable predictions
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        config: Optional[Dict] = None,
    ):
        self.models_dir = models_dir or REPORTS_DIR / "ensemble_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ENSEMBLE_CONFIG
        self.models: Dict[str, object] = {}
        self.feature_columns = ML_SCORING_PARAMS["feature_columns"]
        self.is_trained = False

    def train_ensemble(
        self,
        training_data: pd.DataFrame,
        labels: pd.Series,
        validation_data: Optional[pd.DataFrame] = None,
        validation_labels: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Train all ensemble models

        Args:
            training_data: Features DataFrame
            labels: Target labels (1 = successful trade, 0 = unsuccessful)
            validation_data: Optional validation features
            validation_labels: Optional validation labels

        Returns:
            Training metrics for all models
        """
        X_train = self._prepare_features(training_data)
        y_train = labels

        if X_train.empty or len(y_train) == 0:
            logger.error("No training data available for ensemble")
            return {"error": "No training data"}

        X_val = self._prepare_features(validation_data) if validation_data is not None else None
        y_val = validation_labels

        all_metrics = {}

        for model_name in self.config["models"]:
            logger.info(f"Training {model_name}...")
            metrics = self._train_single_model(
                model_name, X_train, y_train, X_val, y_val
            )
            all_metrics[model_name] = metrics

        # Check if we have enough trained models
        trained_models = [m for m in self.config["models"] if m in self.models]
        if len(trained_models) >= self.config["min_models_for_prediction"]:
            self.is_trained = True
            self.save_models()
            logger.info(f"Ensemble trained successfully with {len(trained_models)} models")
        else:
            logger.warning(f"Only {len(trained_models)} models trained, need at least {self.config['min_models_for_prediction']}")

        return all_metrics

    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train a single model in the ensemble"""
        if model_name == "lightgbm":
            return self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif model_name == "xgboost":
            return self._train_xgboost(X_train, y_train, X_val, y_val)
        elif model_name == "random_forest":
            return self._train_random_forest(X_train, y_train, X_val, y_val)
        else:
            return {"error": f"Unknown model type: {model_name}"}

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train LightGBM model for ensemble"""
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
                'seed': 42,
            }

            train_data = lgb.Dataset(X_train, label=y_train)

            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[train_data, val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
            else:
                model = lgb.train(params, train_data, num_boost_round=200)

            self.models["lightgbm"] = model
            return {"model_type": "lightgbm", "status": "trained"}

        except ImportError:
            logger.warning("LightGBM not installed")
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
        """Train XGBoost model for ensemble"""
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
                'seed': 42,
            }

            dtrain = xgb.DMatrix(X_train, label=y_train)

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
            else:
                model = xgb.train(params, dtrain, num_boost_round=200)

            self.models["xgboost"] = model
            return {"model_type": "xgboost", "status": "trained"}

        except ImportError:
            logger.warning("XGBoost not installed")
            return {"error": "XGBoost not installed"}
        except Exception as e:
            logger.error(f"XGBoost training error: {e}")
            return {"error": str(e)}

    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict:
        """Train Random Forest model for ensemble"""
        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced',
            )

            model.fit(X_train, y_train)
            self.models["random_forest"] = model
            return {"model_type": "random_forest", "status": "trained"}

        except ImportError:
            logger.warning("scikit-learn not installed")
            return {"error": "scikit-learn not installed"}
        except Exception as e:
            logger.error(f"Random Forest training error: {e}")
            return {"error": str(e)}

    def predict_ensemble(self, indicators: Dict) -> Tuple[float, float, Dict]:
        """
        Get ensemble prediction using all trained models

        Args:
            indicators: Dictionary of indicator values

        Returns:
            Tuple of (ensemble_score 0-100, confidence 0-1, model_predictions dict)
        """
        if not self.is_trained or not self.models:
            return 0.0, 0.0, {}

        features = pd.DataFrame([indicators])
        X = self._prepare_features(features)

        if X.empty:
            return 0.0, 0.0, {}

        model_predictions = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for model_name, model in self.models.items():
            try:
                prob = self._get_model_prediction(model_name, model, X)
                if prob is not None:
                    model_predictions[model_name] = prob
                    weight = self.config["model_weights"].get(model_name, 1.0)
                    weighted_sum += prob * weight
                    total_weight += weight
            except Exception as e:
                logger.warning(f"Prediction error for {model_name}: {e}")

        if total_weight == 0 or len(model_predictions) < self.config["min_models_for_prediction"]:
            return 0.0, 0.0, model_predictions

        # Calculate ensemble score
        if self.config["voting_method"] == "soft":
            ensemble_prob = weighted_sum / total_weight
        else:  # hard voting
            ensemble_prob = np.mean(list(model_predictions.values()))

        ensemble_score = ensemble_prob * 100

        # Calculate confidence based on model agreement
        probs = list(model_predictions.values())
        agreement = 1.0 - np.std(probs) * 2  # Lower std = higher agreement
        confidence = max(0.0, min(1.0, agreement))

        return ensemble_score, confidence, model_predictions

    def _get_model_prediction(self, model_name: str, model: object, X: pd.DataFrame) -> Optional[float]:
        """Get prediction from a single model"""
        if model_name == "lightgbm":
            return float(model.predict(X)[0])
        elif model_name == "xgboost":
            import xgboost as xgb
            dtest = xgb.DMatrix(X)
            return float(model.predict(dtest)[0])
        elif model_name == "random_forest":
            return float(model.predict_proba(X)[0, 1])
        return None

    def get_combined_score(
        self,
        traditional_score: float,
        indicators: Dict,
    ) -> Tuple[float, float, Dict]:
        """
        Combine traditional score with ensemble ML score

        Args:
            traditional_score: Score from traditional scoring engine (0-100)
            indicators: Dictionary of indicator values

        Returns:
            Tuple of (combined_score, ml_confidence, model_predictions)
        """
        if not self.is_trained:
            return traditional_score, 0.0, {}

        ml_score, ml_confidence, model_preds = self.predict_ensemble(indicators)

        if ml_confidence < self.config["confidence_threshold"]:
            return traditional_score, ml_confidence, model_preds

        # Weighted combination
        traditional_weight = ML_SCORING_PARAMS["traditional_score_weight"]
        ml_weight = ML_SCORING_PARAMS["ml_score_weight"]

        combined = traditional_score * traditional_weight + ml_score * ml_weight

        return combined, ml_confidence, model_preds

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model"""
        if data is None:
            return pd.DataFrame()

        available_cols = [col for col in self.feature_columns if col in data.columns]
        if not available_cols:
            return pd.DataFrame()

        features = data[available_cols].copy()
        features = features.fillna(features.median())
        features = features.replace([np.inf, -np.inf], 0)
        return features

    def save_models(self, path: Optional[Path] = None):
        """Save all trained models"""
        save_dir = path or self.models_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = save_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} model to {model_path}")

    def load_models(self, path: Optional[Path] = None) -> bool:
        """Load all trained models"""
        load_dir = path or self.models_dir

        loaded_count = 0
        for model_name in self.config["models"]:
            model_path = load_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    loaded_count += 1
                    logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")

        self.is_trained = loaded_count >= self.config["min_models_for_prediction"]
        return self.is_trained


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
