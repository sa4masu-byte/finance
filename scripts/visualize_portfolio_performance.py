#!/usr/bin/env python3
"""
ポートフォリオパフォーマンス可視化スクリプト

推奨銘柄を実際に購入していた場合の資金推移を可視化します。

機能:
- 過去の推奨履歴を読み込み
- 推奨銘柄を購入したシミュレーション
- 資金推移をグラフで可視化
- パフォーマンス統計を表示
"""
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    BACKTEST_CONFIG, RISK_PARAMS, REPORTS_DIR, DATA_DIR
)
from src.data.fetcher import StockDataFetcher
from src.analysis.indicators import TechnicalIndicators

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioTrade:
    """ポートフォリオ内の取引"""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    score: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def return_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def profit_loss(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares


@dataclass
class PortfolioSnapshot:
    """ポートフォリオのスナップショット"""
    date: datetime
    cash: float
    positions_value: float
    total_value: float
    trades: List[PortfolioTrade] = field(default_factory=list)


class PortfolioSimulator:
    """
    推奨銘柄ポートフォリオのシミュレーター
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.001,
        max_positions: int = 5,
    ):
        """
        初期化

        Args:
            initial_capital: 初期資金
            commission_rate: 手数料率
            max_positions: 最大同時保有数
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.max_positions = max_positions

        self.open_positions: List[PortfolioTrade] = []
        self.closed_trades: List[PortfolioTrade] = []
        self.snapshots: List[PortfolioSnapshot] = []

        self.fetcher = StockDataFetcher()
        self.indicator_engine = TechnicalIndicators()

    def simulate_recommendations(
        self,
        recommendations: List[Dict],
        start_date: datetime,
        end_date: datetime,
    ) -> List[PortfolioSnapshot]:
        """
        推奨履歴に基づいてポートフォリオをシミュレート

        Args:
            recommendations: 推奨銘柄のリスト
            start_date: シミュレーション開始日
            end_date: シミュレーション終了日

        Returns:
            ポートフォリオスナップショットのリスト
        """
        logger.info(f"シミュレーション開始: {start_date.date()} → {end_date.date()}")
        logger.info(f"初期資金: ¥{self.initial_capital:,.0f}")
        logger.info(f"推奨銘柄数: {len(recommendations)}")

        # 推奨を日付順にソート
        sorted_recs = sorted(recommendations, key=lambda x: x["date"])

        # 各推奨日にエントリー処理
        for rec in sorted_recs:
            rec_date = rec["date"]
            if rec_date < start_date or rec_date > end_date:
                continue

            # その日の推奨銘柄を処理
            self._process_recommendations_for_date(rec_date, rec["symbols"])

        # シミュレーション期間全体で日次処理
        current_date = start_date
        while current_date <= end_date:
            self._process_trading_day(current_date)
            current_date += timedelta(days=1)

        # 最終日に全ポジションをクローズ
        self._close_all_positions(end_date, "simulation_end")

        logger.info(f"シミュレーション完了")
        logger.info(f"最終資金: ¥{self.cash:,.0f}")
        logger.info(f"総リターン: {((self.cash - self.initial_capital) / self.initial_capital * 100):.2f}%")

        return self.snapshots

    def _process_recommendations_for_date(
        self,
        date: datetime,
        symbols: List[Dict],
    ):
        """特定日の推奨銘柄を処理"""
        # 利用可能なスロット数
        available_slots = self.max_positions - len(self.open_positions)

        if available_slots <= 0:
            return

        # スコアの高い順に並べ替え
        sorted_symbols = sorted(symbols, key=lambda x: x.get("score", 0), reverse=True)

        # 上位N銘柄をエントリー
        for symbol_info in sorted_symbols[:available_slots]:
            self._enter_position(date, symbol_info)

    def _enter_position(self, date: datetime, symbol_info: Dict):
        """ポジションをエントリー"""
        symbol = symbol_info["symbol"]
        score = symbol_info.get("score", 0)

        # すでにポジションを持っている場合はスキップ
        if any(pos.symbol == symbol for pos in self.open_positions):
            return

        # 株価データを取得
        stock_data = self._fetch_stock_data(
            symbol,
            date - timedelta(days=60),
            date + timedelta(days=30)
        )

        if stock_data is None or len(stock_data) == 0:
            logger.warning(f"データ取得失敗: {symbol}")
            return

        # エントリー価格を取得（翌営業日の始値）
        entry_price = self._get_next_trading_day_price(stock_data, date, "Open")
        if entry_price is None:
            logger.warning(f"エントリー価格取得失敗: {symbol}")
            return

        # ATRを計算（ストップロス用）
        df_with_indicators = self.indicator_engine.calculate_all(stock_data)
        atr = self._get_indicator_value(df_with_indicators, date, "ATR")
        if atr is None:
            atr = entry_price * 0.02  # デフォルト2%

        # ポジションサイズを計算
        position_value = self.cash * RISK_PARAMS["position_size_pct"]
        effective_price = entry_price * (1 + self.commission_rate)
        shares = int(position_value / effective_price)

        if shares == 0:
            return

        cost = shares * effective_price

        # 資金が不足している場合はスキップ
        if cost > self.cash:
            return

        # ストップロスと目標価格を設定
        stop_loss = entry_price - (atr * RISK_PARAMS["stop_loss_atr_multiplier"])
        target_price = entry_price * (1 + RISK_PARAMS["profit_target_max"])

        # トレードを作成
        trade = PortfolioTrade(
            symbol=symbol,
            entry_date=date,
            entry_price=effective_price,
            shares=shares,
            score=score,
            stop_loss=stop_loss,
            target_price=target_price,
        )

        self.open_positions.append(trade)
        self.cash -= cost

        logger.info(f"  エントリー: {symbol} @ ¥{effective_price:,.0f} x {shares}株 (スコア: {score:.1f})")

    def _process_trading_day(self, date: datetime):
        """日次処理"""
        # オープンポジションの評価とエグジット判定
        for position in self.open_positions[:]:
            self._check_exit(position, date)

        # ポートフォリオスナップショットを記録
        positions_value = sum(
            self._get_position_value(pos, date) for pos in self.open_positions
        )
        total_value = self.cash + positions_value

        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            trades=list(self.open_positions),
        )
        self.snapshots.append(snapshot)

    def _check_exit(self, position: PortfolioTrade, date: datetime):
        """エグジット条件をチェック"""
        # 株価データを取得
        stock_data = self._fetch_stock_data(
            position.symbol,
            position.entry_date,
            date + timedelta(days=1)
        )

        if stock_data is None:
            return

        # 当日の終値を取得
        current_price = self._get_price_at_date(stock_data, date, "Close")
        if current_price is None:
            return

        should_exit = False
        reason = ""

        # 1. ストップロス
        if current_price <= position.stop_loss:
            should_exit = True
            reason = "stop_loss"

        # 2. 目標価格
        elif current_price >= position.target_price:
            should_exit = True
            reason = "profit_target"

        # 3. 最大保有期間
        elif (date - position.entry_date).days >= RISK_PARAMS["max_holding_days"]:
            should_exit = True
            reason = "max_holding_period"

        if should_exit:
            self._close_position(position, date, current_price, reason)

    def _close_position(
        self,
        position: PortfolioTrade,
        date: datetime,
        price: float,
        reason: str
    ):
        """ポジションをクローズ"""
        effective_price = price * (1 - self.commission_rate)

        position.exit_date = date
        position.exit_price = effective_price
        position.exit_reason = reason

        proceeds = position.shares * effective_price
        self.cash += proceeds

        self.open_positions.remove(position)
        self.closed_trades.append(position)

        logger.info(
            f"  エグジット: {position.symbol} @ ¥{effective_price:,.0f} "
            f"({position.return_pct:+.2f}%, 理由: {reason})"
        )

    def _close_all_positions(self, date: datetime, reason: str):
        """全ポジションをクローズ"""
        for position in self.open_positions[:]:
            stock_data = self._fetch_stock_data(
                position.symbol,
                position.entry_date,
                date + timedelta(days=1)
            )

            if stock_data is None:
                continue

            current_price = self._get_price_at_date(stock_data, date, "Close")
            if current_price:
                self._close_position(position, date, current_price, reason)

    def _fetch_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """株価データを取得"""
        try:
            df = self.fetcher.fetch_stock_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            return df
        except Exception as e:
            logger.warning(f"データ取得エラー ({symbol}): {e}")
            return None

    def _get_next_trading_day_price(
        self,
        df: pd.DataFrame,
        date: datetime,
        column: str
    ) -> Optional[float]:
        """翌営業日の価格を取得"""
        try:
            future_dates = df.index[df.index > date]
            if len(future_dates) == 0:
                return None
            next_date = future_dates[0]
            return float(df.loc[next_date, column])
        except Exception:
            return None

    def _get_price_at_date(
        self,
        df: pd.DataFrame,
        date: datetime,
        column: str
    ) -> Optional[float]:
        """特定日の価格を取得"""
        try:
            if date not in df.index:
                # 最も近い過去の日付を探す
                past_dates = df.index[df.index <= date]
                if len(past_dates) == 0:
                    return None
                date = past_dates[-1]
            return float(df.loc[date, column])
        except Exception:
            return None

    def _get_indicator_value(
        self,
        df: pd.DataFrame,
        date: datetime,
        indicator: str
    ) -> Optional[float]:
        """指標値を取得"""
        try:
            if date not in df.index:
                past_dates = df.index[df.index <= date]
                if len(past_dates) == 0:
                    return None
                date = past_dates[-1]
            return float(df.loc[date, indicator])
        except Exception:
            return None

    def _get_position_value(self, position: PortfolioTrade, date: datetime) -> float:
        """ポジションの時価評価額を取得"""
        stock_data = self._fetch_stock_data(
            position.symbol,
            position.entry_date,
            date + timedelta(days=1)
        )

        if stock_data is None:
            return position.shares * position.entry_price

        current_price = self._get_price_at_date(stock_data, date, "Close")
        if current_price is None:
            return position.shares * position.entry_price

        return position.shares * current_price


class PortfolioVisualizer:
    """
    ポートフォリオパフォーマンスを可視化
    """

    def __init__(self, simulator: PortfolioSimulator):
        self.simulator = simulator

    def plot_equity_curve(self, output_path: Optional[Path] = None):
        """資金推移グラフを表示"""
        if len(self.simulator.snapshots) == 0:
            logger.warning("スナップショットがありません")
            return

        # データを抽出
        dates = [s.date for s in self.simulator.snapshots]
        total_values = [s.total_value for s in self.simulator.snapshots]
        cash_values = [s.cash for s in self.simulator.snapshots]

        # グラフを作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 資金推移グラフ
        ax1.plot(dates, total_values, label="総資産", linewidth=2, color='#2E86DE')
        ax1.axhline(
            y=self.simulator.initial_capital,
            color='gray',
            linestyle='--',
            label=f"初期資金 (¥{self.simulator.initial_capital:,.0f})"
        )
        ax1.set_title("ポートフォリオ資金推移", fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel("日付", fontsize=12)
        ax1.set_ylabel("資産額 (円)", fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x/10000:.0f}万'))

        # 現金とポジション価値の推移
        positions_values = [s.positions_value for s in self.simulator.snapshots]
        ax2.fill_between(dates, 0, cash_values, label="現金", alpha=0.6, color='#26DE81')
        ax2.fill_between(dates, cash_values, total_values, label="ポジション", alpha=0.6, color='#FD7272')
        ax2.set_title("資産構成", fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel("日付", fontsize=12)
        ax2.set_ylabel("資産額 (円)", fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x/10000:.0f}万'))

        # 日付フォーマット
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"グラフを保存しました: {output_path}")
        else:
            plt.show()

    def print_performance_summary(self):
        """パフォーマンスサマリーを表示"""
        print("\n" + "=" * 70)
        print("📊 ポートフォリオパフォーマンスサマリー")
        print("=" * 70)

        # 基本情報
        initial = self.simulator.initial_capital
        final = self.simulator.cash
        total_return = ((final - initial) / initial) * 100

        print(f"\n【資金推移】")
        print(f"  初期資金:     ¥{initial:>12,.0f}")
        print(f"  最終資金:     ¥{final:>12,.0f}")
        print(f"  総リターン:   {total_return:>11,.2f}%")

        # 取引統計
        total_trades = len(self.simulator.closed_trades)
        if total_trades > 0:
            winning_trades = [t for t in self.simulator.closed_trades if t.return_pct > 0]
            losing_trades = [t for t in self.simulator.closed_trades if t.return_pct <= 0]

            win_rate = len(winning_trades) / total_trades * 100
            avg_win = np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0

            print(f"\n【取引統計】")
            print(f"  総取引回数:   {total_trades:>12}")
            print(f"  勝ちトレード: {len(winning_trades):>12} ({win_rate:.1f}%)")
            print(f"  負けトレード: {len(losing_trades):>12}")
            print(f"  平均勝ち:     {avg_win:>11.2f}%")
            print(f"  平均負け:     {avg_loss:>11.2f}%")

            # プロフィットファクター
            total_wins = sum(t.profit_loss for t in winning_trades)
            total_losses = abs(sum(t.profit_loss for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

            print(f"  プロフィットファクター: {profit_factor:>6.2f}")

            # 最大ドローダウン
            equity_curve = [s.total_value for s in self.simulator.snapshots]
            max_dd = self._calculate_max_drawdown(equity_curve)
            print(f"  最大ドローダウン: {max_dd:>10.2f}%")

        # トップ5のトレード
        if total_trades > 0:
            print(f"\n【トップ5トレード】")
            top_trades = sorted(
                self.simulator.closed_trades,
                key=lambda t: t.return_pct,
                reverse=True
            )[:5]

            for i, trade in enumerate(top_trades, 1):
                print(
                    f"  {i}. {trade.symbol:<10} "
                    f"{trade.return_pct:>7.2f}%  "
                    f"(¥{trade.profit_loss:>10,.0f})"
                )

        print("\n" + "=" * 70 + "\n")

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """最大ドローダウンを計算"""
        if not equity_curve:
            return 0.0

        equity = pd.Series(equity_curve)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min()) * 100


def load_recommendation_history() -> List[Dict]:
    """
    推奨履歴を読み込む

    Returns:
        推奨履歴のリスト（日付、銘柄情報を含む）
    """
    recommendation_files = sorted(REPORTS_DIR.glob("recommendation_*.json"))

    if not recommendation_files:
        logger.warning("推奨履歴が見つかりません。サンプルデータを生成します。")
        return generate_sample_recommendations()

    history = []

    for file_path in recommendation_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            date_str = data.get("date", "")
            if not date_str:
                continue

            # 日付をパース
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

            # 推奨銘柄を抽出
            symbols = []
            for rec in data.get("recommendations", []):
                symbols.append({
                    "symbol": rec.get("symbol"),
                    "score": rec.get("total_score", 0),
                    "price": rec.get("price", 0),
                })

            history.append({
                "date": date,
                "symbols": symbols,
            })

        except Exception as e:
            logger.warning(f"ファイル読み込みエラー ({file_path}): {e}")

    logger.info(f"推奨履歴を読み込みました: {len(history)}日分")
    return history


def generate_sample_recommendations() -> List[Dict]:
    """
    サンプル推奨データを生成（デモ用）
    """
    # 過去3ヶ月分のサンプルデータ
    sample_symbols = [
        "7203.JP",  # トヨタ
        "6758.JP",  # ソニー
        "9984.JP",  # ソフトバンクG
        "6861.JP",  # キーエンス
        "8306.JP",  # 三菱UFJ
    ]

    history = []
    start_date = datetime.now() - timedelta(days=90)

    # 週に1回推奨
    for i in range(12):
        rec_date = start_date + timedelta(days=i * 7)

        symbols = []
        for symbol in sample_symbols[:3]:  # 上位3銘柄
            symbols.append({
                "symbol": symbol,
                "score": 70 + (i % 20),  # 70-90点
                "price": 0,
            })

        history.append({
            "date": rec_date,
            "symbols": symbols,
        })

    logger.info(f"サンプル推奨データを生成しました: {len(history)}日分")
    return history


def main():
    """メイン処理"""
    logger.info("=" * 70)
    logger.info("ポートフォリオパフォーマンス可視化ツール")
    logger.info("=" * 70)

    # 1. 推奨履歴を読み込む
    recommendations = load_recommendation_history()

    if not recommendations:
        logger.error("推奨履歴がありません")
        return 1

    # シミュレーション期間を決定
    start_date = min(rec["date"] for rec in recommendations)
    end_date = max(rec["date"] for rec in recommendations) + timedelta(days=30)

    # 2. シミュレーターを初期化
    simulator = PortfolioSimulator(
        initial_capital=1_000_000,
        commission_rate=0.001,
        max_positions=5,
    )

    # 3. シミュレーションを実行
    snapshots = simulator.simulate_recommendations(
        recommendations,
        start_date,
        end_date,
    )

    # 4. 可視化
    visualizer = PortfolioVisualizer(simulator)

    # パフォーマンスサマリーを表示
    visualizer.print_performance_summary()

    # グラフを保存
    output_path = REPORTS_DIR / f"portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    visualizer.plot_equity_curve(output_path)

    logger.info("可視化完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
