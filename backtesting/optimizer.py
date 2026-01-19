"""
Weight optimization engine using grid search and Bayesian optimization
"""
import logging
from typing import Dict, List, Tuple, Optional
from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import OPTIMIZATION_CONFIG, BACKTEST_CONFIG, RISK_PARAMS
from backtesting.backtest_engine import BacktestEngine, BacktestResults

logger = logging.getLogger(__name__)


class WeightOptimizer:
    """
    Optimize scoring weights using historical data
    """

    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        train_start: str,
        train_end: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
    ):
        """
        Initialize optimizer

        Args:
            stock_data: Dictionary mapping symbol to OHLCV DataFrame
            train_start: Training period start date
            train_end: Training period end date
            test_start: Test period start date (optional)
            test_end: Test period end date (optional)
        """
        self.stock_data = stock_data
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        self.results_history = []

    def grid_search(self, param_space: Optional[Dict] = None) -> List[Tuple[Dict, float]]:
        """
        Perform grid search over weight combinations

        Args:
            param_space: Dictionary of parameter ranges (uses default if None)

        Returns:
            List of (weights, score) tuples, sorted by score (descending)
        """
        param_space = param_space or OPTIMIZATION_CONFIG["param_space"]

        logger.info("Starting grid search...")
        logger.info(f"Parameter space: {param_space}")

        # Generate all valid combinations
        combinations = self._generate_valid_combinations(param_space)
        logger.info(f"Testing {len(combinations)} valid weight combinations")

        results = []

        # Test each combination
        for weights in tqdm(combinations, desc="Grid Search"):
            score = self._evaluate_weights(weights, self.train_start, self.train_end)
            results.append((weights, score))

            # Store in history
            self.results_history.append({
                "weights": weights,
                "score": score,
                "method": "grid_search"
            })

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Grid search complete. Best score: {results[0][1]:.4f}")
        logger.info(f"Best weights: {results[0][0]}")

        return results

    def bayesian_optimization(
        self,
        initial_results: List[Tuple[Dict, float]],
        n_iterations: int = OPTIMIZATION_CONFIG["n_iterations"],
        verbose: bool = False,
    ) -> Tuple[Dict, float]:
        """
        Perform Bayesian optimization starting from grid search results

        Args:
            initial_results: Results from grid search
            n_iterations: Number of optimization iterations
            verbose: Whether to print optimization progress

        Returns:
            (best_weights, best_score) tuple
        """
        logger.info("Starting Bayesian optimization...")

        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
        except ImportError:
            logger.warning("scikit-optimize not available, skipping Bayesian optimization")
            return initial_results[0]

        # Define search space
        space = [
            Real(0.20, 0.40, name="trend"),
            Real(0.15, 0.35, name="momentum"),
            Real(0.10, 0.30, name="volume"),
            Real(0.10, 0.25, name="volatility"),
        ]

        # Objective function
        @use_named_args(space)
        def objective(**params):
            # Calculate pattern weight
            pattern = 1.0 - sum(params.values())

            # Validate constraints
            if not (0.05 <= pattern <= 0.20):
                return 1e6  # Penalty for invalid combination

            if pattern < OPTIMIZATION_CONFIG["min_weight"] or pattern > OPTIMIZATION_CONFIG["max_weight"]:
                return 1e6

            weights = {
                "trend": params["trend"],
                "momentum": params["momentum"],
                "volume": params["volume"],
                "volatility": params["volatility"],
                "pattern": pattern,
            }

            # Evaluate (minimize negative score)
            score = self._evaluate_weights(weights, self.train_start, self.train_end)
            return -score  # Minimize negative score = maximize score

        # Initial points from grid search
        x0 = []
        y0 = []
        for weights, score in initial_results[:OPTIMIZATION_CONFIG["n_initial_points"]]:
            x0.append([
                weights["trend"],
                weights["momentum"],
                weights["volume"],
                weights["volatility"],
            ])
            y0.append(-score)  # Negative because we minimize

        # Run optimization
        result = gp_minimize(
            objective,
            space,
            x0=x0,
            y0=y0,
            n_calls=n_iterations,
            random_state=42,
            verbose=verbose,
        )

        # Extract best weights
        best_weights = {
            "trend": result.x[0],
            "momentum": result.x[1],
            "volume": result.x[2],
            "volatility": result.x[3],
            "pattern": 1.0 - sum(result.x),
        }
        best_score = -result.fun

        logger.info(f"Bayesian optimization complete. Best score: {best_score:.4f}")
        logger.info(f"Best weights: {best_weights}")

        return best_weights, best_score

    def walk_forward_analysis(
        self,
        weights: Dict,
        periods: List[Tuple[str, str, str, str]],
    ) -> List[BacktestResults]:
        """
        Perform walk-forward analysis

        Args:
            weights: Scoring weights to test
            periods: List of (train_start, train_end, test_start, test_end) tuples

        Returns:
            List of BacktestResults for each test period
        """
        logger.info("Starting walk-forward analysis...")

        results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"Period {i+1}/{len(periods)}: "
                        f"Train {train_start} to {train_end}, "
                        f"Test {test_start} to {test_end}")

            # Run backtest on test period
            engine = BacktestEngine(scoring_weights=weights)
            test_results = engine.run_backtest(
                self.stock_data,
                test_start,
                test_end,
            )

            results.append(test_results)

            logger.info(f"Period {i+1} results: "
                        f"Win rate={test_results.win_rate:.1%}, "
                        f"Return={test_results.total_return_pct:.2f}%, "
                        f"Sharpe={test_results.sharpe_ratio:.2f}")

        return results

    def _generate_valid_combinations(self, param_space: Dict) -> List[Dict]:
        """Generate all valid weight combinations from parameter space"""
        combinations = []

        # Generate all combinations
        for trend, momentum, volume, volatility in product(
            param_space["trend"],
            param_space["momentum"],
            param_space["volume"],
            param_space["volatility"],
        ):
            pattern = 1.0 - (trend + momentum + volume + volatility)

            # Check constraints
            if not (OPTIMIZATION_CONFIG["min_weight"] <= pattern <= OPTIMIZATION_CONFIG["max_weight"]):
                continue

            if not np.isclose(trend + momentum + volume + volatility + pattern, 1.0, atol=0.001):
                continue

            combinations.append({
                "trend": trend,
                "momentum": momentum,
                "volume": volume,
                "volatility": volatility,
                "pattern": pattern,
            })

        return combinations

    def _evaluate_weights(self, weights: Dict, start_date: str, end_date: str) -> float:
        """
        Evaluate a set of weights using composite score

        Args:
            weights: Scoring weights
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Composite optimization score (0.0 on error)
        """
        try:
            # Run backtest
            engine = BacktestEngine(scoring_weights=weights)
            results = engine.run_backtest(self.stock_data, start_date, end_date)

            # Calculate composite score
            score = self._calculate_optimization_score(results)

            return score
        except Exception as e:
            logger.warning(f"Error evaluating weights {weights}: {e}")
            return 0.0  # Return minimum score on error

    def _calculate_optimization_score(self, results: BacktestResults) -> float:
        """
        Calculate composite optimization score from backtest results

        Args:
            results: BacktestResults object

        Returns:
            Composite score (0-100), returns 0.0 for empty results
        """
        # Handle empty results
        if results.num_trades == 0:
            logger.debug("No trades in backtest results, returning score 0")
            return 0.0

        obj_weights = OPTIMIZATION_CONFIG["objective_weights"]

        # Normalize metrics to 0-100 scale
        win_rate_score = results.win_rate * 100

        # Profit factor (cap at 5.0 for normalization)
        pf_normalized = min(results.profit_factor, 5.0) / 5.0 * 100

        # Sharpe ratio (cap at 3.0 for normalization)
        sharpe_normalized = min(results.sharpe_ratio, 3.0) / 3.0 * 100

        # Max drawdown (lower is better, invert)
        dd_score = max(0, (1 - results.max_drawdown / 100)) * 100

        # Holding days (score based on proximity to ideal range)
        ideal_min = RISK_PARAMS["min_holding_days"]
        ideal_max = RISK_PARAMS["max_holding_days"]
        avg_days = results.avg_holding_days

        if ideal_min <= avg_days <= ideal_max:
            holding_score = 100
        else:
            # Penalty for being outside range
            if avg_days < ideal_min:
                holding_score = (avg_days / ideal_min) * 100
            else:
                holding_score = max(0, (1 - (avg_days - ideal_max) / ideal_max)) * 100

        # Weighted composite
        composite_score = (
            win_rate_score * obj_weights["win_rate"]
            + pf_normalized * obj_weights["profit_factor"]
            + sharpe_normalized * obj_weights["sharpe_ratio"]
            + dd_score * obj_weights["max_drawdown"]
            + holding_score * obj_weights["holding_days"]
        )

        return composite_score

    def save_results(self, filepath: Path, best_weights: Dict, all_results: List[Tuple[Dict, float]]):
        """Save optimization results to JSON"""
        output = {
            "best_weights": best_weights,
            "training_period": {
                "start": self.train_start,
                "end": self.train_end,
            },
            "top_results": [
                {"weights": w, "score": float(s)}
                for w, s in all_results[:20]
            ],
            "optimization_config": OPTIMIZATION_CONFIG,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to {filepath}")


# Convenience function
def optimize_weights(
    stock_data: Dict[str, pd.DataFrame],
    train_start: str,
    train_end: str,
    method: str = "both",
) -> Tuple[Dict, float]:
    """
    Optimize scoring weights

    Args:
        stock_data: Dictionary of stock DataFrames
        train_start: Training start date
        train_end: Training end date
        method: "grid", "bayesian", or "both"

    Returns:
        (best_weights, best_score) tuple
    """
    optimizer = WeightOptimizer(stock_data, train_start, train_end)

    if method in ["grid", "both"]:
        grid_results = optimizer.grid_search()

        if method == "grid":
            return grid_results[0]

        # Continue with Bayesian
        best_weights, best_score = optimizer.bayesian_optimization(grid_results)
        return best_weights, best_score

    else:
        raise ValueError(f"Unknown method: {method}")
