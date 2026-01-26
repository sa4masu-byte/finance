#!/bin/bash
# =============================================================================
# Mac Setup Script for Japanese Stock Swing Trading System
# =============================================================================
#
# Usage:
#   chmod +x setup_mac.sh
#   ./setup_mac.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=============================================="
echo "  日本株スイングトレード推奨システム"
echo "  Mac セットアップスクリプト"
echo "=============================================="
echo -e "${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# =============================================================================
# 1. Check Prerequisites
# =============================================================================
echo -e "${YELLOW}[1/6] 前提条件をチェック中...${NC}"

# Check macOS version
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: このスクリプトはMac専用です${NC}"
    exit 1
fi

echo "  ✓ macOS detected: $(sw_vers -productVersion)"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}  Homebrewがインストールされていません。インストールしますか? (y/n)${NC}"
    read -r install_brew
    if [[ "$install_brew" == "y" ]]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        echo -e "${RED}Homebrewが必要です。インストールしてから再実行してください。${NC}"
        exit 1
    fi
fi
echo "  ✓ Homebrew: $(brew --version | head -1)"

# =============================================================================
# 2. Install Python
# =============================================================================
echo -e "${YELLOW}[2/6] Pythonをチェック中...${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 9 ]]; then
    echo -e "${YELLOW}  Python 3.9以上が必要です。インストールします...${NC}"
    brew install python@3.11

    # Update PATH
    export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
    echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
fi

echo "  ✓ Python: $(python3 --version)"

# =============================================================================
# 3. Create Virtual Environment
# =============================================================================
echo -e "${YELLOW}[3/6] 仮想環境を作成中...${NC}"

VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "  既存の仮想環境が見つかりました。再作成しますか? (y/n)"
    read -r recreate_venv
    if [[ "$recreate_venv" == "y" ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
else
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "  ✓ 仮想環境: $VENV_DIR"

# =============================================================================
# 4. Install Dependencies
# =============================================================================
echo -e "${YELLOW}[4/6] 依存パッケージをインストール中...${NC}"

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install optional ML packages
echo -e "${YELLOW}  オプション: 機械学習パッケージをインストールしますか? (y/n)${NC}"
read -r install_ml
if [[ "$install_ml" == "y" ]]; then
    pip install lightgbm xgboost
    echo "  ✓ LightGBM, XGBoost インストール完了"
fi

echo "  ✓ 依存パッケージインストール完了"

# =============================================================================
# 5. Create Directories and Configuration
# =============================================================================
echo -e "${YELLOW}[5/6] ディレクトリと設定ファイルを作成中...${NC}"

# Create necessary directories
mkdir -p "$SCRIPT_DIR/data/cache"
mkdir -p "$SCRIPT_DIR/data/db"
mkdir -p "$SCRIPT_DIR/data/reports"
mkdir -p "$SCRIPT_DIR/logs"

# Create .env file if not exists
ENV_FILE="$SCRIPT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'EOF'
# Finance System Environment Configuration
# =========================================

# Data Settings
DATA_SOURCE=stooq
CACHE_EXPIRY_HOURS=24

# Notification Settings (Optional)
# Slack通知を使用する場合は以下を設定
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx

# LINE通知を使用する場合は以下を設定
# LINE_NOTIFY_TOKEN=your_token_here

# Email通知を使用する場合は以下を設定
# EMAIL_SMTP_SERVER=smtp.gmail.com
# EMAIL_SMTP_PORT=587
# EMAIL_FROM=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password
# EMAIL_TO=recipient@example.com

# Logging
LOG_LEVEL=INFO
EOF
    echo "  ✓ .env ファイルを作成しました"
else
    echo "  ✓ .env ファイルは既に存在します"
fi

echo "  ✓ ディレクトリ構成完了"

# =============================================================================
# 6. Setup Launchd (Auto-run on schedule)
# =============================================================================
echo -e "${YELLOW}[6/6] 自動実行の設定...${NC}"

echo -e "${YELLOW}  毎日の自動実行を設定しますか? (y/n)${NC}"
read -r setup_launchd

if [[ "$setup_launchd" == "y" ]]; then
    echo "  実行時刻を選択してください:"
    echo "    1) 朝 7:00 (市場開始前)"
    echo "    2) 夜 18:00 (市場終了後)"
    echo "    3) 両方"
    echo "    4) カスタム時刻"
    read -r schedule_choice

    PLIST_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$PLIST_DIR"

    create_launchd_plist() {
        local hour=$1
        local minute=$2
        local name=$3

        cat > "$PLIST_DIR/com.finance.daily-$name.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.finance.daily-$name</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCRIPT_DIR/run_daily.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>$hour</integer>
        <key>Minute</key>
        <integer>$minute</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$SCRIPT_DIR/logs/launchd-$name.log</string>
    <key>StandardErrorPath</key>
    <string>$SCRIPT_DIR/logs/launchd-$name-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
        launchctl unload "$PLIST_DIR/com.finance.daily-$name.plist" 2>/dev/null || true
        launchctl load "$PLIST_DIR/com.finance.daily-$name.plist"
        echo "  ✓ $name スケジュール設定完了 ($hour:$minute)"
    }

    case $schedule_choice in
        1)
            create_launchd_plist 7 0 "morning"
            ;;
        2)
            create_launchd_plist 18 0 "evening"
            ;;
        3)
            create_launchd_plist 7 0 "morning"
            create_launchd_plist 18 0 "evening"
            ;;
        4)
            echo "  時刻を入力 (HH:MM形式):"
            read -r custom_time
            IFS=':' read -r hour minute <<< "$custom_time"
            create_launchd_plist "$hour" "$minute" "custom"
            ;;
    esac
fi

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo -e "${GREEN}=============================================="
echo "  セットアップ完了!"
echo "==============================================${NC}"
echo ""
echo "使用方法:"
echo ""
echo "  1. 仮想環境を有効化:"
echo -e "     ${BLUE}source .venv/bin/activate${NC}"
echo ""
echo "  2. リアルデータを取得:"
echo -e "     ${BLUE}python scripts/fetch_real_data.py${NC}"
echo ""
echo "  3. バックテストを実行:"
echo -e "     ${BLUE}python scripts/validate_with_real_data.py${NC}"
echo ""
echo "  4. 日次推奨を取得:"
echo -e "     ${BLUE}python scripts/run_daily_recommendation.py${NC}"
echo ""
echo "  5. 重み最適化 (オプション):"
echo -e "     ${BLUE}python scripts/run_optimization.py${NC}"
echo ""
echo "ログファイル: $SCRIPT_DIR/logs/"
echo "データ: $SCRIPT_DIR/data/"
echo ""

# Deactivate virtual environment
deactivate
