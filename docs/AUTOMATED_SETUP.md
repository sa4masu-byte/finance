# 完全自動化セットアップガイド

PC不要・スマホだけで株式分析結果を確認できる環境を構築します。

## 概要

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   GitHub Actions          PythonAnywhere           スマホ       │
│   ┌───────────┐          ┌───────────┐          ┌─────────┐   │
│   │ 毎朝8時   │  ──────▶ │ダッシュ   │  ──────▶ │ 結果    │   │
│   │ 自動分析  │  アップ  │ボード     │  閲覧    │ 確認    │   │
│   └───────────┘          └───────────┘          └─────────┘   │
│     無料        ロード      無料                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 必要なもの

- GitHubアカウント（無料）
- PythonAnywhereアカウント（無料）
- スマートフォン

**PCは初期セットアップ時のみ必要。その後は不要。**

---

## Step 1: PythonAnywhere セットアップ（15分）

### 1.1 アカウント作成

1. https://www.pythonanywhere.com/ にアクセス
2. 「Start running Python online」→「Create a Beginner account」
3. ユーザー名・メール・パスワードを設定

### 1.2 プロジェクトをクローン

「Consoles」→「Bash」を開き、以下を実行:

```bash
cd ~
git clone https://github.com/YOUR_GITHUB_USERNAME/finance.git
cd finance

# 依存関係インストール
pip install --user -e .
pip install --user flask
```

### 1.3 データディレクトリ作成

```bash
mkdir -p ~/finance/data/reports
```

### 1.4 Web アプリ設定

1. 「Web」タブ → 「Add a new web app」
2. 「Next」→「Flask」→「Python 3.10」
3. 設定を以下に変更:

**Source code:**
```
/home/YOUR_PA_USERNAME/finance
```

**WSGI configuration file** をクリックして内容を以下に置き換え:

```python
import sys
from pathlib import Path

project_home = '/home/YOUR_PA_USERNAME/finance'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from dashboard.app import app as application
```

4. 「Reload」ボタンをクリック

### 1.5 API トークン取得

1. 「Account」→ 下にスクロール →「API token」
2. 「Create a new API token」をクリック
3. トークンをコピーして保存（後で使用）

### 1.6 動作確認

ブラウザで `https://YOUR_PA_USERNAME.pythonanywhere.com` にアクセス。
ダッシュボードが表示されればOK（データはまだ空）。

---

## Step 2: GitHub Actions セットアップ（10分）

### 2.1 リポジトリをフォーク

1. GitHubで元のリポジトリを開く
2. 「Fork」ボタンをクリック
3. 自分のアカウントにフォーク

### 2.2 シークレット設定

1. フォークしたリポジトリの「Settings」タブ
2. 左メニュー「Secrets and variables」→「Actions」
3. 「New repository secret」で以下を追加:

| Name | Value |
|------|-------|
| `PA_USERNAME` | PythonAnywhereのユーザー名 |
| `PA_API_TOKEN` | Step 1.5で取得したAPIトークン |

### 2.3 Actions 有効化

1. 「Actions」タブを開く
2. 「I understand my workflows, go ahead and enable them」をクリック

### 2.4 手動で初回実行

1. 「Actions」タブ
2. 左メニュー「Daily Stock Analysis」
3. 「Run workflow」→「Run workflow」

数分待つと完了。PythonAnywhereのダッシュボードにデータが表示されます。

---

## Step 3: スマホでブックマーク

スマホのブラウザで以下にアクセス:

```
https://YOUR_PA_USERNAME.pythonanywhere.com
```

ホーム画面に追加しておくと便利です。

---

## 完成！

### 自動実行スケジュール

- **毎朝 8:00 (JST)** に自動で分析が実行されます
- 結果はPythonAnywhereに自動アップロード
- スマホで確認するだけでOK

### 手動実行したい場合

1. GitHubの「Actions」タブを開く（スマホからも可能）
2. 「Daily Stock Analysis」→「Run workflow」

---

## トラブルシューティング

### Actions が失敗する

1. 「Actions」タブでエラーログを確認
2. よくある原因:
   - シークレットの設定ミス
   - PythonAnywhereのユーザー名が間違っている

### ダッシュボードが表示されない

1. PythonAnywhereの「Web」タブで「Error log」を確認
2. よくある原因:
   - WSGIファイルのパスが間違っている
   - `pip install --user flask` が未実行

### データが更新されない

1. GitHub Actionsの実行履歴を確認
2. PythonAnywhereの `~/finance/data/reports/latest_report.json` を確認

---

## 無料枠の使用量

| サービス | 無料枠 | 使用量（月） | 残り |
|----------|--------|-------------|------|
| GitHub Actions | 2000分 | 約90分 | 95%残 |
| PythonAnywhere | 512MB | 約10MB | 98%残 |

**完全無料で運用可能です。**

---

## セキュリティ

- APIトークンはGitHubシークレットで安全に管理
- ダッシュボードは公開URLですが、個人的な取引情報は含まれません
- 必要に応じてBasic認証を追加可能（PythonAnywhereの設定で）

---

## ファイル構成

```
finance/
├── .github/
│   └── workflows/
│       └── daily-analysis.yml    # 自動実行設定
├── dashboard/
│   ├── app.py                    # Flaskアプリ
│   ├── wsgi.py                   # PythonAnywhere用
│   └── templates/
│       ├── dashboard.html        # ダッシュボード画面
│       └── run.html              # 手動実行画面
├── scripts/
│   ├── generate_daily_report.py  # 分析スクリプト
│   └── upload_to_pythonanywhere.py # アップロード
└── data/
    └── reports/
        └── latest_report.json    # 分析結果
```
