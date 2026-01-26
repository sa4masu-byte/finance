# Mac セットアップガイド

## 概要

このドキュメントでは、日本株スイングトレード推奨システムをMacで運用する方法を説明します。

## クイックスタート

```bash
# 1. リポジトリをクローン
git clone https://github.com/sa4masu-byte/finance.git
cd finance

# 2. セットアップスクリプトを実行
chmod +x setup_mac.sh
./setup_mac.sh

# 3. 手動実行する場合
source .venv/bin/activate
python scripts/run_daily_recommendation.py
```

## システム要件

- macOS 10.15 (Catalina) 以降
- Python 3.9以上（セットアップスクリプトで自動インストール）
- インターネット接続（株価データ取得用）

## ディレクトリ構成

```
finance/
├── .venv/                  # Python仮想環境
├── data/
│   ├── cache/              # 一時キャッシュ（3日で自動削除）
│   ├── reports/            # 推奨結果（3日で自動削除）
│   └── watchlist.json      # 監視銘柄リスト
├── logs/                   # ログファイル
├── scripts/
│   └── run_daily_recommendation.py
├── setup_mac.sh            # セットアップスクリプト
├── run_daily.sh            # 日次実行スクリプト
└── requirements-minimal.txt # 最小限の依存パッケージ
```

## 日次運用

### 手動実行

```bash
cd /path/to/finance
source .venv/bin/activate
python scripts/run_daily_recommendation.py
```

### 自動実行（launchd）

セットアップスクリプトで設定するか、手動で設定：

```bash
# 設定ファイルを作成
cat > ~/Library/LaunchAgents/com.finance.daily.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.finance.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/finance/run_daily.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>7</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/path/to/finance/logs/daily.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/finance/logs/daily-error.log</string>
</dict>
</plist>
EOF

# 有効化
launchctl load ~/Library/LaunchAgents/com.finance.daily.plist
```

### 自動実行の管理

```bash
# 状態確認
launchctl list | grep finance

# 無効化
launchctl unload ~/Library/LaunchAgents/com.finance.daily.plist

# 再有効化
launchctl load ~/Library/LaunchAgents/com.finance.daily.plist

# 即時実行（テスト用）
launchctl start com.finance.daily
```

## 監視銘柄のカスタマイズ

`data/watchlist.json` を編集：

```json
{
  "symbols": [
    "7203.JP",
    "6758.JP",
    "追加したい銘柄.JP"
  ]
}
```

## 通知設定（オプション）

`.env` ファイルを編集：

```bash
# Slack通知
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx

# LINE通知（別途設定が必要）
LINE_NOTIFY_TOKEN=your_token_here
```

## データ使用量の目安

- **日次データ取得**: 約 1-2 MB
- **キャッシュ**: 最大 10 MB（3日で自動削除）
- **ログ**: 約 1 MB/月

## トラブルシューティング

### データが取得できない場合

```bash
# ネットワーク確認
curl -I https://stooq.com

# キャッシュをクリア
rm -rf data/cache/*
```

### 仮想環境の問題

```bash
# 仮想環境を再作成
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
```

### launchdが動作しない場合

```bash
# ログを確認
cat ~/Library/Logs/com.finance.daily.log
cat logs/launchd-*.log

# 権限を確認
chmod +x run_daily.sh
```

## アップデート

```bash
cd /path/to/finance
git pull
source .venv/bin/activate
pip install -r requirements-minimal.txt --upgrade
```

## サポート

問題がある場合は Issue を作成してください:
https://github.com/sa4masu-byte/finance/issues
