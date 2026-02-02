# 日本株スイングトレード推奨システム

チャートとテクニカル指標を駆使して、その日購入すべき日本株銘柄を推奨するシステムです。

## 🎯 特徴

- **データ駆動型**: 過去データから最適な重み付けを自動導出
- **テクニカル分析**: 移動平均、RSI、MACD、ボリンジャーバンド、ATR、出来高分析などを統合
- **スイングトレード最適化**: 3-15日間の保有期間に特化
- **リスク管理**: ATRベースのストップロス、ポジションサイズ計算
- **バックテスト検証**: ウォークフォワード分析で過学習を防止

## 📊 システム概要

### スコアリングロジック

各銘柄をテクニカル指標から総合スコア（0-100点）で評価:

| カテゴリ | 配点 | 主要指標 |
|---------|------|---------|
| **トレンド** | 30% | SMA(5/25/75), MACD |
| **モメンタム** | 25% | RSI, ストキャスティクス |
| **出来高** | 20% | 出来高比率, OBV |
| **ボラティリティ** | 15% | ボリンジャーバンド, ATR |
| **パターン** | 10% | サポート/レジスタンス |

※ 重み付けは過去データの最適化により決定

### テクニカル指標の根拠

- **SMA (5/25/75日)**: ゴールデンクロスで勝率58-62% (Brock et al. 1992)
- **MACD (12/26/9)**: 価格より先行してトレンド転換を検出
- **RSI (14日)**: 40-65が最適ゾーン、勝率58-63% (Brown 2014)
- **ボリンジャーバンド**: スクイーズ後3-10日で平均7-12%の動き
- **出来高**: 1.5倍以上で継続確率65-70% (Lee & Swaminathan 2000)

詳細は [docs/indicator_selection_rationale.md](docs/indicator_selection_rationale.md) を参照

## 🚀 セットアップ

### 必要要件

- Python 3.10以上
- インターネット接続（データ取得用）

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd finance

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 主要パッケージ

```
pandas-datareader  # Stooq経由でデータ取得
pandas            # データ処理
numpy             # 数値計算
scikit-optimize   # ベイズ最適化
scikit-learn      # 機械学習
scipy             # 科学計算
tqdm              # プログレスバー
pyarrow           # Parquet形式サポート
```

## 📈 使用方法

### 1. 重み付けの最適化

過去データから最適なスコアリング重み付けを導出:

```bash
python scripts/run_optimization.py
```

**処理内容:**
1. 15銘柄の過去4年分データを取得（2022-2025）
2. グリッドサーチで500-800通りの重み組み合わせを評価
3. ベイズ最適化で精密探索
4. ウォークフォワード分析で3期間検証
5. 最適重みを `data/reports/best_weights.json` に保存

**実行時間:** 約30分〜1時間（銘柄数とデータ量による）

**成功基準:**
- 勝率 > 60%
- プロフィットファクター > 2.0
- シャープレシオ > 1.5
- 最大ドローダウン < 15%

### 2. 最適化結果の確認

```bash
cat data/reports/best_weights.json
```

出力例:
```json
{
  "weights": {
    "trend": 0.325,
    "momentum": 0.270,
    "volume": 0.225,
    "volatility": 0.125,
    "pattern": 0.055
  },
  "validation_performance": {
    "avg_win_rate": 0.648,
    "avg_return": 32.5,
    "avg_sharpe": 1.82,
    "avg_profit_factor": 2.35
  }
}
```

### 3. ポートフォリオパフォーマンスの可視化

推奨銘柄を実際に購入していた場合の資金推移を可視化:

```bash
python scripts/visualize_portfolio_performance.py
```

**機能:**
- 過去の推奨履歴を自動読み込み
- 推奨銘柄ポートフォリオのシミュレーション
- 資金推移グラフの生成
- パフォーマンス統計の表示
  - 総リターン
  - 勝率とプロフィットファクター
  - 最大ドローダウン
  - トップ/ワーストトレード

**出力:**
- グラフ: `data/reports/portfolio_performance_YYYYMMDD_HHMMSS.png`
- パフォーマンスサマリー（コンソール出力）

## 🌐 Webダッシュボード

### ローカル起動

Streamlitベースのインタラクティブなダッシュボードを提供:

```bash
# Webダッシュボード起動
streamlit run web/app.py
```

ブラウザで http://localhost:8501 にアクセス

**ページ構成:**
- 🏠 **ホーム**: 今日の推奨銘柄と詳細スコア
- 📈 **パフォーマンス**: ポートフォリオ資金推移グラフ
- 📜 **履歴**: 過去の推奨銘柄一覧とフィルター機能
- ⚙️ **設定**: システム設定と手動実行

### 定期実行スケジューラー

APSchedulerで自動タスク実行:

```bash
# スケジューラー起動（別ターミナル）
python web/scheduler.py
```

**実行スケジュール:**
- 毎日 09:00 JST: 推奨銘柄生成
- 毎日 18:00 JST: パフォーマンス可視化
- 毎週月曜 10:00 JST: 重み最適化（オプション）

### Docker環境

```bash
# ビルド & 起動
docker-compose up -d

# ログ確認
docker-compose logs -f

# 停止
docker-compose down
```

**サービス構成:**
- `web`: Streamlitダッシュボード (ポート: 8501)
- `scheduler`: バックグラウンド定期実行

**詳細:** [docs/deployment_guide.md](docs/deployment_guide.md)

## 🗂️ プロジェクト構成

```
finance/
├── README.md                  # このファイル
├── requirements.txt           # 依存パッケージ
├── Dockerfile                 # Dockerイメージ定義
├── docker-compose.yml         # Docker Compose設定
│
├── config/
│   ├── settings.py           # 設定パラメータ
│   └── stock_universe.json   # 対象銘柄リスト
│
├── docs/
│   ├── indicator_selection_rationale.md      # 指標選定の根拠
│   ├── backtesting_optimization_design.md    # 最適化設計書
│   ├── web_service_architecture.md           # Webサービス設計書
│   └── deployment_guide.md                   # デプロイガイド
│
├── src/
│   ├── data/
│   │   ├── fetcher.py       # Stooqデータ取得
│   │   └── cache.py         # キャッシュ管理
│   │
│   └── analysis/
│       └── indicators.py    # テクニカル指標計算
│
├── backtesting/
│   ├── scoring_engine.py    # スコアリングロジック
│   ├── backtest_engine.py   # バックテストシミュレーション
│   └── optimizer.py         # 重み最適化
│
├── web/                      # Webダッシュボード
│   ├── app.py               # Streamlitメインアプリ
│   ├── data_manager.py      # データアクセス層
│   ├── scheduler.py         # 定期実行スケジューラー
│   ├── pages/               # ダッシュボードページ
│   │   ├── home.py          # ホームページ
│   │   ├── performance.py   # パフォーマンスページ
│   │   ├── history.py       # 履歴ページ
│   │   └── settings.py      # 設定ページ
│   └── .streamlit/
│       └── config.toml      # Streamlit設定
│
├── scripts/
│   ├── run_optimization.py               # 最適化実行
│   ├── run_daily_recommendation.py       # 日次推奨生成
│   ├── visualize_portfolio_performance.py # パフォーマンス可視化
│   ├── download_and_backtest.py          # データ取得&バックテスト
│   └── validate_backtest_simulation.py   # バックテスト検証
│
├── tests/                    # テストコード
│   └── ...
│
└── data/
    ├── stock_cache/          # 株価データキャッシュ (Parquet)
    └── reports/              # レポート & 推奨履歴
        ├── recommendation_*.json           # 日次推奨
        ├── portfolio_performance_*.png     # パフォーマンスグラフ
        └── best_weights.json               # 最適化重み
```

## ⚙️ 設定のカスタマイズ

### config/settings.py

主要な設定項目:

```python
# テクニカル指標パラメータ
INDICATOR_PARAMS = {
    "sma_short": 5,
    "sma_medium": 25,
    "sma_long": 75,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    # ...
}

# リスク管理
RISK_PARAMS = {
    "position_size_pct": 0.20,           # 1銘柄あたり20%
    "stop_loss_atr_multiplier": 2.0,     # ATR×2でストップロス
    "profit_target_max": 0.15,           # 利益目標15%
    "max_holding_days": 15,              # 最大保有15日
}

# バックテスト
BACKTEST_CONFIG = {
    "initial_capital": 1_000_000,        # 初期資金100万円
    "max_positions": 5,                  # 最大同時保有5銘柄
    "commission_rate": 0.001,            # 手数料0.1%
}
```

### config/stock_universe.json

監視する銘柄リストを編集可能。

## 📖 技術的詳細

### データソース

**Stooq** (pandas-datareader経由)
- 無料で利用可能
- 日本株データ完備（銘柄コード.JP形式）
- ローカルキャッシュで高速化

### テクニカル指標の計算

手動実装（pandas-ta不要）:
- 移動平均: `df["Close"].rolling(window).mean()`
- RSI: Wilderのアルゴリズム
- MACD: EMA差分
- ボリンジャーバンド: 標準偏差ベース
- ATR: True Rangeの移動平均

### 最適化アルゴリズム

1. **グリッドサーチ** (Phase 1)
   - 重みの組み合わせ: 500-800通り
   - 制約: 合計100%、各5-45%
   - 評価: 複合スコア（勝率35% + PF25% + Sharpe20% + DD15% + 保有期間5%）

2. **ベイズ最適化** (Phase 2)
   - ガウス過程で効率的探索
   - 上位20組から開始
   - 50イテレーションで収束

3. **ウォークフォワード分析** (検証)
   - 3期間で交差検証
   - 過学習検出

## 🎓 学術的根拠

本システムは以下の研究に基づいています:

1. **Brock, Lakonishok & LeBaron (1992)** - 移動平均クロス戦略の有効性
2. **Jegadeesh & Titman (1993)** - モメンタム戦略の年率8-12%リターン
3. **Lee & Swaminathan (2000)** - 出来高と価格パターンの関係
4. **Fama & French (2008)** - 出来高伴うモメンタムの高リターン
5. **Constance Brown (2014)** - RSI最適ゾーンの実証

詳細: [docs/indicator_selection_rationale.md](docs/indicator_selection_rationale.md)

## ⚠️ 免責事項

**本システムは教育・研究目的で作成されています。**

- 金融助言ではありません
- 投資判断は自己責任で行ってください
- 過去の成績は将来の成果を保証しません
- 実際の取引前に十分な検証を行ってください

## 📝 ライセンス

MIT License

---

**現在の実装状況:**
- ✅ データ取得
- ✅ テクニカル指標計算
- ✅ スコアリングエンジン
- ✅ バックテストエンジン
- ✅ 重み最適化
- ✅ 日次推奨システム
- ✅ ポートフォリオパフォーマンス可視化
- ✅ Webダッシュボード（Streamlit）
- ✅ 定期実行スケジューラー（APScheduler）
- ✅ Docker対応
