# Streamlit Cloud デプロイガイド

スマホからどこでもアクセスできる株式推奨システムを構築します。

## 📱 完成イメージ

```
GitHub Actions (自動実行)
  ├── 毎日 09:00 JST: 推奨銘柄生成
  ├── 毎日 09:00 JST: パフォーマンス可視化
  └── 毎週月曜 10:00 JST: 重み最適化
         ↓
    GitHubに自動commit
         ↓
Streamlit Cloud (自動デプロイ)
         ↓
スマホ・PCからアクセス
https://your-app.streamlit.app
```

---

## 🚀 デプロイ手順

### ステップ1: GitHubリポジトリの準備

#### 1-1. 変更をコミット&プッシュ

```bash
cd ~/finance
git add .
git commit -m "Add GitHub Actions and Streamlit Cloud config"
git push origin main
```

#### 1-2. リポジトリを公開（プライベートでもOK）

GitHub上で設定:
- リポジトリ → Settings → Visibility
- Public または Private（Streamlit CloudはPrivateもサポート）

---

### ステップ2: Streamlit Community Cloud にデプロイ

#### 2-1. Streamlit Cloudにアクセス

https://streamlit.io/cloud にアクセス

#### 2-2. サインアップ/ログイン

- GitHubアカウントで認証
- 無料プランでOK

#### 2-3. 新しいアプリをデプロイ

1. **"New app" ボタンをクリック**

2. **リポジトリ設定:**
   - Repository: `sa4masu-byte/finance`
   - Branch: `main`
   - Main file path: `web/app.py`

3. **Advanced settings（オプション）:**
   - Python version: `3.11`
   - その他はデフォルトでOK

4. **"Deploy!" ボタンをクリック**

#### 2-4. デプロイ完了を待つ

- 初回は5-10分かかります
- ログを確認しながら待機
- 完了すると自動的にURLが発行されます

例: `https://finance-recommendations.streamlit.app`

---

### ステップ3: GitHub Actions の有効化

#### 3-1. リポジトリの Actions を有効化

GitHub リポジトリ:
1. **Actions** タブをクリック
2. ワークフローを有効化

#### 3-2. 初回の手動実行

1. **Actions** タブ
2. **"Daily Stock Recommendations"** を選択
3. **"Run workflow"** ボタンをクリック
4. 実行完了を待つ（約5-10分）

#### 3-3. データ生成を確認

`data/reports/` に推奨ファイルが作成されたことを確認:
- `recommendation_YYYYMMDD.json`
- `portfolio_performance_YYYYMMDD.png`

---

### ステップ4: 初期データのバックフィル

過去のデータを生成して履歴を充実させます。

#### ローカルで実行（推奨）

```bash
# 仮想環境を有効化
source venv/bin/activate

# 過去30日分を生成
python scripts/backfill_recommendations.py --days 30

# コミット&プッシュ
git add data/reports/
git commit -m "Add historical recommendations data"
git push
```

#### GitHub Actions で実行

1. `.github/workflows/backfill.yml` を作成:

```yaml
name: Backfill Historical Data

on:
  workflow_dispatch:
    inputs:
      days:
        description: 'Number of days to backfill'
        required: true
        default: '30'

jobs:
  backfill:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python scripts/backfill_recommendations.py --days ${{ github.event.inputs.days }}
      - run: |
          git config --global user.name 'GitHub Actions Bot'
          git config --global user.email 'actions@github.com'
          git add data/reports/
          git commit -m "Backfill ${{ github.event.inputs.days }} days of recommendations"
          git push
```

2. Actions → "Backfill Historical Data" → Run workflow

---

## 📱 スマホからアクセス

### URLをホーム画面に追加

#### iOS (Safari)
1. Streamlit アプリのURLを開く
2. 共有ボタン（□↑）をタップ
3. 「ホーム画面に追加」を選択
4. 名前: 「株式推奨」など

#### Android (Chrome)
1. Streamlit アプリのURLを開く
2. メニュー（⋮）→「ホーム画面に追加」
3. 名前: 「株式推奨」など

→ **アプリのように使えます！**

---

## ⚙️ 運用設定

### 自動実行スケジュール

#### 現在の設定:
- **毎日 09:00 JST**: 推奨銘柄生成
- **毎週月曜 10:00 JST**: 重み最適化

#### 変更したい場合:

`.github/workflows/daily-recommendations.yml` を編集:

```yaml
on:
  schedule:
    - cron: '0 0 * * *'  # UTC時刻で指定
    # 日本時間 09:00 = UTC 00:00
    # 日本時間 18:00 = UTC 09:00
```

**Cron記法:**
```
┌───────────── 分 (0 - 59)
│ ┌───────────── 時 (0 - 23)
│ │ ┌───────────── 日 (1 - 31)
│ │ │ ┌───────────── 月 (1 - 12)
│ │ │ │ ┌───────────── 曜日 (0 - 6) (日曜=0)
│ │ │ │ │
* * * * *
```

例:
- `0 0 * * *` - 毎日 00:00 UTC (09:00 JST)
- `0 9 * * *` - 毎日 09:00 UTC (18:00 JST)
- `0 1 * * 1` - 毎週月曜 01:00 UTC (10:00 JST)

---

## 🔧 トラブルシューティング

### アプリが起動しない

**エラーログを確認:**
1. Streamlit Cloud ダッシュボード
2. アプリを選択
3. "Logs" タブでエラー確認

**よくある問題:**
- `requirements.txt` のパッケージエラー
  → バージョンを調整
- メモリ不足
  → データキャッシュを削減

### データが表示されない

**推奨データが生成されているか確認:**
1. GitHub リポジトリ
2. `data/reports/` フォルダ
3. `recommendation_*.json` ファイルの有無

**ない場合:**
```bash
# ローカルで生成
python scripts/run_daily_recommendation.py
git add data/reports/
git commit -m "Add initial recommendations"
git push
```

### GitHub Actions が動かない

**Actions が有効か確認:**
- リポジトリ → Settings → Actions → "Allow all actions"

**ワークフロー権限を確認:**
- Settings → Actions → General → Workflow permissions
- "Read and write permissions" を選択

---

## 🎯 日々の使い方

### 朝（通勤中など）
1. スマホでアプリを開く
2. ホームページで今日の推奨銘柄を確認
3. 気になる銘柄をメモ

### 日中
1. 証券アプリで詳細確認
2. 購入判断

### 夜
1. パフォーマンスページで結果確認
2. 過去の推奨のパフォーマンスをチェック

### 週末
1. 履歴ページで1週間を振り返り
2. CSVエクスポートで詳細分析

---

## 💰 コスト

### 完全無料で運用可能！

- **Streamlit Community Cloud**: 無料
  - 1アプリまで無料
  - 1GB RAM
  - 十分な性能

- **GitHub Actions**: 無料
  - Public リポジトリ: 無制限
  - Private リポジトリ: 月2,000分まで無料
  - 1日数分の実行なので十分

---

## 🔒 セキュリティ

### データの安全性

- **Streamlit Cloud**: HTTPS 暗号化
- **GitHub**: プライベートリポジトリ可能
- **認証**: 必要に応じてStreamlit認証を追加可能

### Basic認証を追加（オプション）

`web/app.py` の先頭に追加:

```python
import streamlit as st

# Basic認証
def check_password():
    def password_entered():
        if st.session_state["password"] == "your-password-here":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# 以下、既存のコード
```

---

## 📊 モニタリング

### デプロイ状況の確認

**Streamlit Cloud:**
- https://share.streamlit.io/ でダッシュボード確認
- アプリの稼働状況・ログ・リソース使用状況

**GitHub Actions:**
- リポジトリ → Actions タブ
- 実行履歴・成功/失敗状況

---

## 🚀 次のステップ

1. **通知機能追加**
   - 推奨銘柄をメール/Slackに通知
   - GitHub Actions → 通知スクリプト

2. **パフォーマンス改善**
   - データキャッシング最適化
   - 表示速度向上

3. **機能追加**
   - ポートフォリオ管理
   - アラート設定
   - カスタムスコアリング

---

準備完了です！デプロイを始めましょう 🎉
