"""
Unit tests for ML Scoring Module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.append(str(Path(__file__).parent.parent))

from backtesting.ml_scoring import MLScoringEngine


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def ml_engine():
    """Create ML scoring engine"""
    return MLScoringEngine(model_type="lightgbm")


@pytest.fixture
def sample_training_data():
    """Generate sample training data"""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        "RSI_14": np.random.uniform(20, 80, n_samples),
        "MACD": np.random.uniform(-5, 5, n_samples),
        "MACD_Histogram": np.random.uniform(-2, 2, n_samples),
        "BB_Percent": np.random.uniform(0, 1, n_samples),
        "Stoch_K": np.random.uniform(0, 100, n_samples),
        "Stoch_D": np.random.uniform(0, 100, n_samples),
        "Volume_Ratio": np.random.uniform(0.5, 2.0, n_samples),
        "ATR_Percent": np.random.uniform(0.01, 0.05, n_samples),
        "SMA_5": np.random.uniform(900, 1100, n_samples),
        "SMA_25": np.random.uniform(900, 1100, n_samples),
        "SMA_75": np.random.uniform(900, 1100, n_samples),
        "OBV_Trend": np.random.choice([-1, 0, 1], n_samples),
    })

    # Labels: 1 = successful trade, 0 = unsuccessful
    labels = pd.Series(np.random.choice([0, 1], n_samples, p=[0.5, 0.5]))

    return data, labels


@pytest.fixture
def sample_indicators():
    """Generate sample indicator values"""
    return {
        "RSI_14": 55.0,
        "MACD": 0.5,
        "MACD_Histogram": 0.2,
        "BB_Percent": 0.45,
        "Stoch_K": 60.0,
        "Stoch_D": 55.0,
        "Volume_Ratio": 1.2,
        "ATR_Percent": 0.02,
        "SMA_5": 1050.0,
        "SMA_25": 1000.0,
        "SMA_75": 980.0,
        "OBV_Trend": 1,
    }


# =============================================================================
# Test MLScoringEngine Initialization
# =============================================================================

class TestMLScoringEngineInit:
    """Tests for MLScoringEngine initialization"""

    def test_initialization_lightgbm(self):
        """Test initialization with LightGBM"""
        engine = MLScoringEngine(model_type="lightgbm")
        assert engine.model_type == "lightgbm"
        assert engine.model is None
        assert engine.is_trained is False

    def test_initialization_xgboost(self):
        """Test initialization with XGBoost"""
        engine = MLScoringEngine(model_type="xgboost")
        assert engine.model_type == "xgboost"

    def test_initialization_custom_path(self):
        """Test initialization with custom model path"""
        custom_path = Path("/tmp/test_model.pkl")
        engine = MLScoringEngine(model_path=custom_path)
        assert engine.model_path == custom_path

    def test_feature_columns_set(self, ml_engine):
        """Test that feature columns are set"""
        assert len(ml_engine.feature_columns) > 0
        assert "RSI_14" in ml_engine.feature_columns


# =============================================================================
# Test Feature Preparation
# =============================================================================

class TestFeaturePreparation:
    """Tests for feature preparation"""

    def test_prepare_features_valid_data(self, ml_engine, sample_training_data):
        """Test feature preparation with valid data"""
        data, _ = sample_training_data
        features = ml_engine._prepare_features(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)
        # Check no NaN values
        assert features.isna().sum().sum() == 0

    def test_prepare_features_missing_columns(self, ml_engine):
        """Test feature preparation with missing columns"""
        data = pd.DataFrame({
            "RSI_14": [50, 60, 70],
            "MACD": [0.1, 0.2, 0.3],
            # Missing other columns
        })
        features = ml_engine._prepare_features(data)

        # Should only include available columns
        assert "RSI_14" in features.columns
        assert "MACD" in features.columns
        assert len(features.columns) == 2

    def test_prepare_features_empty_data(self, ml_engine):
        """Test feature preparation with empty data"""
        data = pd.DataFrame()
        features = ml_engine._prepare_features(data)
        assert features.empty

    def test_prepare_features_nan_handling(self, ml_engine):
        """Test that NaN values are handled"""
        data = pd.DataFrame({
            "RSI_14": [50, np.nan, 70],
            "MACD": [0.1, 0.2, np.nan],
        })
        features = ml_engine._prepare_features(data)

        # NaN values should be filled
        assert features.isna().sum().sum() == 0

    def test_prepare_features_inf_handling(self, ml_engine):
        """Test that infinite values are handled"""
        data = pd.DataFrame({
            "RSI_14": [50, np.inf, 70],
            "MACD": [0.1, -np.inf, 0.3],
        })
        features = ml_engine._prepare_features(data)

        # Infinite values should be replaced
        assert not np.isinf(features.values).any()


# =============================================================================
# Test Model Loading/Saving
# =============================================================================

class TestModelIO:
    """Tests for model loading and saving"""

    def test_load_model_not_found(self, ml_engine):
        """Test loading non-existent model"""
        ml_engine.model_path = Path("/nonexistent/path/model.pkl")
        result = ml_engine.load_model()
        assert result is False
        assert ml_engine.is_trained is False

    def test_save_and_load_model(self, ml_engine, sample_training_data):
        """Test saving and loading a model"""
        # Skip if lightgbm not installed
        pytest.importorskip("lightgbm")

        data, labels = sample_training_data

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            ml_engine.model_path = model_path

            # Train model
            ml_engine.train_model(data, labels)

            # Verify save
            assert model_path.exists()

            # Load in new engine
            new_engine = MLScoringEngine(model_path=model_path)
            result = new_engine.load_model()

            assert result is True
            assert new_engine.is_trained is True
            assert new_engine.model is not None


# =============================================================================
# Test Predictions
# =============================================================================

class TestPredictions:
    """Tests for ML predictions"""

    def test_predict_untrained_model(self, ml_engine, sample_indicators):
        """Test prediction with untrained model"""
        score, confidence = ml_engine.predict_score(sample_indicators)
        assert score == 0.0
        assert confidence == 0.0

    def test_predict_trained_model(self, ml_engine, sample_training_data, sample_indicators):
        """Test prediction with trained model"""
        pytest.importorskip("lightgbm")

        data, labels = sample_training_data
        ml_engine.train_model(data, labels)

        score, confidence = ml_engine.predict_score(sample_indicators)

        assert 0 <= score <= 100
        assert 0 <= confidence <= 1

    def test_get_combined_score_untrained(self, ml_engine, sample_indicators):
        """Test combined score with untrained model"""
        traditional_score = 75.0
        combined, confidence = ml_engine.get_combined_score(traditional_score, sample_indicators)

        # Should return traditional score when untrained
        assert combined == traditional_score
        assert confidence == 0.0

    def test_get_combined_score_trained(self, ml_engine, sample_training_data, sample_indicators):
        """Test combined score with trained model"""
        pytest.importorskip("lightgbm")

        data, labels = sample_training_data
        ml_engine.train_model(data, labels)

        traditional_score = 75.0
        combined, confidence = ml_engine.get_combined_score(traditional_score, sample_indicators)

        # Combined score should be a blend
        assert combined > 0
        assert combined <= 100


# =============================================================================
# Test Model Training
# =============================================================================

class TestModelTraining:
    """Tests for model training"""

    def test_train_lightgbm(self, ml_engine, sample_training_data):
        """Test LightGBM training"""
        pytest.importorskip("lightgbm")

        data, labels = sample_training_data
        metrics = ml_engine.train_model(data, labels)

        assert "error" not in metrics
        assert metrics["model_type"] == "lightgbm"
        assert ml_engine.is_trained is True

    def test_train_xgboost(self, sample_training_data):
        """Test XGBoost training"""
        pytest.importorskip("xgboost")

        engine = MLScoringEngine(model_type="xgboost")
        data, labels = sample_training_data
        metrics = engine.train_model(data, labels)

        assert "error" not in metrics
        assert metrics["model_type"] == "xgboost"
        assert engine.is_trained is True

    def test_train_with_validation(self, ml_engine, sample_training_data):
        """Test training with validation data"""
        pytest.importorskip("lightgbm")

        data, labels = sample_training_data

        # Split data
        split_idx = int(len(data) * 0.8)
        X_train = data.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_val = data.iloc[split_idx:]
        y_val = labels.iloc[split_idx:]

        metrics = ml_engine.train_model(X_train, y_train, X_val, y_val)

        assert "error" not in metrics
        assert ml_engine.is_trained is True

    def test_train_empty_data(self, ml_engine):
        """Test training with empty data"""
        data = pd.DataFrame()
        labels = pd.Series(dtype=int)

        metrics = ml_engine.train_model(data, labels)

        assert "error" in metrics

    def test_train_invalid_model_type(self):
        """Test training with invalid model type"""
        engine = MLScoringEngine(model_type="invalid")
        data = pd.DataFrame({"RSI_14": [50, 60, 70]})
        labels = pd.Series([0, 1, 0])

        metrics = engine.train_model(data, labels)

        assert "error" in metrics


# =============================================================================
# Test Training Labels Creation
# =============================================================================

class TestTrainingLabels:
    """Tests for training label creation"""

    def test_create_training_labels_basic(self, ml_engine):
        """Test basic training label creation"""
        from backtesting.enhanced_backtest_engine import EnhancedTrade
        from datetime import datetime

        trades = [
            EnhancedTrade(
                symbol="TEST1",
                entry_date=datetime(2023, 1, 1),
                entry_price=1000.0,
                exit_date=datetime(2023, 1, 10),
                exit_price=1100.0,  # +10% return
                shares=100,
                score_at_entry=75.0,
            ),
            EnhancedTrade(
                symbol="TEST2",
                entry_date=datetime(2023, 1, 15),
                entry_price=2000.0,
                exit_date=datetime(2023, 1, 20),
                exit_price=1900.0,  # -5% return
                shares=50,
                score_at_entry=60.0,
            ),
        ]

        features, labels = ml_engine.create_training_labels(trades, min_return_threshold=0.05)

        assert len(features) == 2
        assert len(labels) == 2
        assert labels[0] == 1  # +10% >= 5%
        assert labels[1] == 0  # -5% < 5%

    def test_create_training_labels_open_trades_ignored(self, ml_engine):
        """Test that open trades are ignored"""
        from backtesting.enhanced_backtest_engine import EnhancedTrade
        from datetime import datetime

        trades = [
            EnhancedTrade(
                symbol="TEST1",
                entry_date=datetime(2023, 1, 1),
                entry_price=1000.0,
                # No exit - open trade
                shares=100,
                score_at_entry=75.0,
            ),
        ]

        features, labels = ml_engine.create_training_labels(trades)

        assert len(features) == 0
        assert len(labels) == 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_predict_with_missing_features(self, ml_engine, sample_training_data):
        """Test prediction with missing features"""
        pytest.importorskip("lightgbm")

        data, labels = sample_training_data
        ml_engine.train_model(data, labels)

        # Indicators with only some features
        partial_indicators = {"RSI_14": 55.0}
        score, confidence = ml_engine.predict_score(partial_indicators)

        # Should still work with available features
        assert isinstance(score, float)
        assert isinstance(confidence, float)

    def test_predict_empty_indicators(self, ml_engine, sample_training_data):
        """Test prediction with empty indicators"""
        pytest.importorskip("lightgbm")

        data, labels = sample_training_data
        ml_engine.train_model(data, labels)

        empty_indicators = {}
        score, confidence = ml_engine.predict_score(empty_indicators)

        assert score == 0.0
        assert confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
