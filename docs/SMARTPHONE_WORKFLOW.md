# スマホ完結ワークフロー

Anaconda環境で、スマホから分析実行〜結果確認まで完結する方法です。

## 構成

```
┌─────────────────────────────────────────────────────────────┐
│                      あなたのPC                              │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Anaconda環境     │    │ Dashboard       │                │
│  │ (finance)       │───▶│ localhost:5000  │                │
│  └─────────────────┘    └────────┬────────┘                │
│                                   │                         │
└───────────────────────────────────┼─────────────────────────┘
                                    │ ngrok
                                    ▼
                          https://xxxx.ngrok.io
                                    │
                                    ▼
                              ┌──────────┐
                              │  スマホ   │
                              └──────────┘
```

## 初期セットアップ（PCで1回だけ）

### 1. Anaconda環境構築

```bash
# 仮想環境作成
conda create -n finance python=3.11 -y
conda activate finance

# リポジトリクローン
git clone <repository-url>
cd finance

# 依存関係インストール
pip install -e ".[dashboard]"
pip install yfinance
```

### 2. ngrokインストール

```bash
# Windows (PowerShell)
choco install ngrok
# または https://ngrok.com/download からダウンロード

# Mac
brew install ngrok

# Linux
snap install ngrok
```

### 3. ngrokアカウント設定（無料）

1. https://ngrok.com/ でアカウント作成
2. https://dashboard.ngrok.com/get-started/your-authtoken でトークン取得
3. 設定:
```bash
ngrok config add-authtoken YOUR_TOKEN
```

## 毎日の使い方（スマホから）

### 方法1: PCを常時起動しておく場合

#### PCで起動（1回だけ）

```bash
# ターミナル1: Anaconda環境でダッシュボード起動
conda activate finance
cd finance
python -m dashboard.app

# ターミナル2: ngrokでトンネル作成
ngrok http 5000
```

ngrokが表示するURL（例: `https://abc123.ngrok.io`）をスマホでブックマーク。

#### スマホで操作

1. ブックマークしたURLにアクセス
2. 「Run Analysis」ボタンをタップ
3. 数分待つ
4. 「Dashboard」で結果確認

### 方法2: PCをリモート起動する場合

#### Wake on LAN設定（オプション）

1. PCのBIOSでWake on LANを有効化
2. スマホアプリ「Wake On Lan」等をインストール
3. PCのMACアドレスを登録

#### 起動スクリプト作成（PCで）

`start_dashboard.bat`（Windows）:
```batch
@echo off
call conda activate finance
cd /d C:\path\to\finance
start /B python -m dashboard.app
timeout /t 5
ngrok http 5000
```

`start_dashboard.sh`（Mac/Linux）:
```bash
#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate finance
cd ~/finance
python -m dashboard.app &
sleep 5
ngrok http 5000
```

### 方法3: 常時稼働サーバー（Raspberry Pi等）

Raspberry Piなどの小型PCにセットアップすれば、常時アクセス可能。

## スマホ画面の使い方

### ダッシュボード画面

```
┌─────────────────────────┐
│   Stock Recommender     │
│      2024-01-19         │
│    [Bull Market]        │  ← 市場状況
├─────────────────────────┤
│ Performance             │
│  +5.2%  │  52.1%        │  ← 成績
│  Return │  Win Rate     │
├─────────────────────────┤
│ Today's Picks           │
│ ┌─────────────────────┐ │
│ │ 7203.T Toyota   78  │ │  ← 推奨銘柄
│ │ ████████████░  85%  │ │
│ └─────────────────────┘ │
├─────────────────────────┤
│     [ Refresh ]         │  ← データ更新
│   [ Run Analysis ]      │  ← 分析実行
└─────────────────────────┘
```

### 分析実行画面

```
┌─────────────────────────┐
│    Stock Analysis       │
│                         │
│      ┌───────┐          │
│      │ RUN   │          │  ← タップで実行
│      │ANALYSIS│          │
│      └───────┘          │
│                         │
│  Status: Running...     │
│                         │
│     [ Dashboard ]       │
└─────────────────────────┘
```

## 注意事項

### セキュリティ

- ngrokのURLは毎回変わります（無料版）
- URLを他人に共有しないでください
- 有料版では固定URLも可能

### ネットワーク

- PCがインターネットに接続されている必要があります
- スマホはWiFi/モバイルどちらでもOK

### 制限事項

| 項目 | 内容 |
|------|------|
| ngrok無料版 | URLが毎回変わる、1接続のみ |
| データ取得 | 市場が閉まっている時間は前日データ |
| 応答時間 | 分析に1-2分かかることがあります |

## トラブルシューティング

### 「Connection refused」エラー

→ PCでダッシュボードが起動していません
```bash
conda activate finance
python -m dashboard.app
```

### 「ngrok tunnel not found」

→ ngrokを再起動してください
```bash
ngrok http 5000
```

### 分析結果が更新されない

→ 「Run Analysis」を再実行してください

### データ取得エラー

→ Yahoo Financeのレート制限の可能性
→ 数分待ってから再実行
