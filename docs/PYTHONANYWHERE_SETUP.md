# PythonAnywhere デプロイ手順

## 1. アカウント作成

1. https://www.pythonanywhere.com/ にアクセス
2. 「Start running Python online in less than a minute!」をクリック
3. 「Create a Beginner account」を選択（無料）
4. メールアドレス、ユーザー名、パスワードを入力

## 2. ファイルのアップロード

### 方法A: GitHubからクローン（推奨）

1. PythonAnywhereのダッシュボードで「Consoles」→「Bash」を開く
2. 以下を実行:

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/finance.git
cd finance
pip install --user -e .
pip install --user flask yfinance
```

### 方法B: 手動アップロード

1. 「Files」タブを開く
2. 「Upload a file」でZIPファイルをアップロード
3. Bashコンソールで解凍:

```bash
cd ~
unzip finance.zip
cd finance
pip install --user -e .
pip install --user flask yfinance
```

## 3. WSGIファイルの設定

1. `dashboard/wsgi.py` を開く
2. `YOUR_USERNAME` を自分のユーザー名に変更:

```python
project_home = '/home/YOUR_USERNAME/finance'
```

## 4. Webアプリの作成

1. 「Web」タブを開く
2. 「Add a new web app」をクリック
3. 「Next」→「Flask」→「Python 3.10」を選択
4. 以下を設定:

### Source code:
```
/home/YOUR_USERNAME/finance
```

### WSGI configuration file:
「WSGI configuration file」のリンクをクリックして、内容を以下に置き換え:

```python
import sys
from pathlib import Path

project_home = '/home/YOUR_USERNAME/finance'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from dashboard.app import app as application
```

## 5. 静的ファイルの設定（オプション）

「Static files」セクションで追加:

| URL | Directory |
|-----|-----------|
| /static | /home/YOUR_USERNAME/finance/dashboard/static |

## 6. リロード

1. 「Web」タブに戻る
2. 緑の「Reload」ボタンをクリック

## 7. アクセス確認

ブラウザで以下にアクセス:
```
https://YOUR_USERNAME.pythonanywhere.com
```

## 8. 日次レポートの更新

### 手動更新

Bashコンソールで:
```bash
cd ~/finance
python scripts/generate_daily_report.py
```

### 自動更新（Scheduled Tasks）

1. 「Tasks」タブを開く
2. 「Scheduled tasks」で時刻を設定（例: 08:00 UTC = 17:00 JST）
3. コマンドを入力:

```bash
cd ~/finance && /home/YOUR_USERNAME/.local/bin/python scripts/generate_daily_report.py
```

## トラブルシューティング

### エラーログの確認

「Web」タブ → 「Log files」セクション:
- Error log: エラー詳細
- Server log: アクセスログ

### よくある問題

1. **ModuleNotFoundError**
   - `pip install --user <module>` で再インストール

2. **Permission denied**
   - ファイルパスを確認

3. **外部API接続エラー**
   - 無料プランは外部接続に制限あり
   - `generate_daily_report.py` はローカルで実行し、レポートのみアップロード

## 無料プランの制限

- CPU時間: 100秒/日
- ストレージ: 512MB
- 外部接続: ホワイトリストのみ（※株価API制限あり）
- カスタムドメイン: 不可

## レポートのアップロード（外部API制限回避）

ローカルPCで生成したレポートをアップロード:

```bash
# ローカルで実行
python scripts/generate_daily_report.py

# PythonAnywhereにアップロード（scpまたはFilesタブ）
scp data/reports/latest_report.json YOUR_USERNAME@ssh.pythonanywhere.com:~/finance/data/reports/
```
