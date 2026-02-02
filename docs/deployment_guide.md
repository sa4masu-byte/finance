# デプロイガイド

株式推奨システムのWebサービスをデプロイする手順

## 目次

1. [ローカル開発環境](#ローカル開発環境)
2. [Docker環境](#docker環境)
3. [本番環境デプロイ](#本番環境デプロイ)
4. [トラブルシューティング](#トラブルシューティング)

---

## ローカル開発環境

### 前提条件

- Python 3.9以上
- pip
- 2GB以上のメモリ

### セットアップ

```bash
# 1. リポジトリクローン
git clone <repository-url>
cd finance

# 2. 依存パッケージインストール
pip install -r requirements.txt
pip install APScheduler

# 3. データディレクトリ作成
mkdir -p data/reports data/stock_cache

# 4. 初期データ生成（オプション）
python scripts/run_daily_recommendation.py
python scripts/visualize_portfolio_performance.py
```

### 起動

#### Webダッシュボードのみ起動

```bash
streamlit run web/app.py
```

ブラウザで http://localhost:8501 にアクセス

#### スケジューラーも起動（別ターミナル）

```bash
# バックグラウンドで実行
python web/scheduler.py &

# またはnohupで実行
nohup python web/scheduler.py > scheduler.log 2>&1 &

# ログ確認
tail -f scheduler.log
```

---

## Docker環境

### 前提条件

- Docker 20.10以上
- Docker Compose 2.0以上

### クイックスタート

```bash
# 1. ビルド & 起動
docker-compose up -d

# 2. ログ確認
docker-compose logs -f

# 3. 停止
docker-compose down
```

### 個別サービス管理

```bash
# Webダッシュボードのみ起動
docker-compose up -d web

# スケジューラーのみ起動
docker-compose up -d scheduler

# 特定サービスのログ確認
docker-compose logs -f web
docker-compose logs -f scheduler

# コンテナ内でコマンド実行
docker-compose exec web python scripts/run_daily_recommendation.py
```

### データ永続化

データは以下のディレクトリにマウントされます:

```
./data → /app/data (コンテナ内)
./config → /app/config (コンテナ内)
```

### ポート設定

- Webダッシュボード: http://localhost:8501

ポート変更する場合は `docker-compose.yml` を編集:

```yaml
services:
  web:
    ports:
      - "8080:8501"  # ホスト:コンテナ
```

---

## 本番環境デプロイ

### オプション1: VPS / クラウドサーバー

#### AWS EC2 / GCP Compute Engine / Azure VM

```bash
# 1. サーバーにSSH接続
ssh user@your-server-ip

# 2. Dockerインストール
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Docker Composeインストール
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. リポジトリクローン
git clone <repository-url>
cd finance

# 5. 起動
docker-compose up -d

# 6. 自動起動設定
sudo systemctl enable docker
```

#### ファイアウォール設定

```bash
# ポート8501を開放
sudo ufw allow 8501/tcp
sudo ufw enable
```

#### 独自ドメイン設定（オプション）

Nginx リバースプロキシ設定:

```nginx
# /etc/nginx/sites-available/finance
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Nginx有効化
sudo ln -s /etc/nginx/sites-available/finance /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Let's Encrypt SSL証明書（オプション）
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### オプション2: Heroku

```bash
# 1. Heroku CLIインストール
# https://devcenter.heroku.com/articles/heroku-cli

# 2. ログイン
heroku login

# 3. アプリ作成
heroku create your-app-name

# 4. コンテナスタック設定
heroku stack:set container

# 5. デプロイ
git push heroku main

# 6. スケール設定
heroku ps:scale web=1 scheduler=1

# 7. ログ確認
heroku logs --tail
```

### オプション3: Railway / Render

#### Railway

```bash
# 1. Railway CLI インストール
npm install -g @railway/cli

# 2. ログイン
railway login

# 3. プロジェクト初期化
railway init

# 4. デプロイ
railway up
```

#### Render

1. Render.com にアクセス
2. "New Web Service" をクリック
3. GitHubリポジトリを接続
4. Docker設定を選択
5. デプロイ

---

## 環境変数設定

### 本番環境用の環境変数

`.env` ファイルを作成:

```bash
# タイムゾーン
TZ=Asia/Tokyo

# ログレベル
LOG_LEVEL=INFO

# ポート設定（オプション）
WEB_PORT=8501

# その他の設定
PYTHONUNBUFFERED=1
```

Docker Composeで読み込む:

```yaml
services:
  web:
    env_file:
      - .env
```

---

## モニタリング & 管理

### ログ管理

```bash
# Dockerログ確認
docker-compose logs --tail=100 web
docker-compose logs --tail=100 scheduler

# リアルタイムログ
docker-compose logs -f

# スケジューラーログ（ローカル）
tail -f scheduler.log
```

### ヘルスチェック

```bash
# コンテナステータス確認
docker-compose ps

# リソース使用状況
docker stats

# ディスク使用状況
du -sh data/
```

### バックアップ

```bash
# データディレクトリをバックアップ
tar -czf finance-backup-$(date +%Y%m%d).tar.gz data/

# 定期バックアップ（cron）
0 2 * * * cd /path/to/finance && tar -czf backup-$(date +\%Y\%m\%d).tar.gz data/
```

### アップデート

```bash
# 1. 最新コードを取得
git pull origin main

# 2. コンテナ再ビルド
docker-compose build

# 3. 再起動
docker-compose down
docker-compose up -d

# 4. 古いイメージ削除
docker image prune -a
```

---

## トラブルシューティング

### よくある問題

#### 1. ポートが既に使用されている

```bash
# ポート使用状況確認
sudo lsof -i :8501

# プロセス終了
kill <PID>
```

#### 2. コンテナが起動しない

```bash
# ログ確認
docker-compose logs web
docker-compose logs scheduler

# コンテナ再起動
docker-compose restart

# 完全再ビルド
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

#### 3. データが表示されない

```bash
# 推奨銘柄を手動生成
docker-compose exec web python scripts/run_daily_recommendation.py

# パフォーマンスグラフを手動生成
docker-compose exec web python scripts/visualize_portfolio_performance.py

# データディレクトリ確認
ls -la data/reports/
```

#### 4. メモリ不足

Docker Composeでメモリ制限を設定:

```yaml
services:
  web:
    mem_limit: 1g
  scheduler:
    mem_limit: 2g
```

#### 5. スケジューラーが動作しない

```bash
# スケジューラーログ確認
docker-compose logs -f scheduler

# 手動でジョブ実行
docker-compose exec scheduler python scripts/run_daily_recommendation.py
```

### デバッグモード

開発用のdebug設定:

```yaml
# docker-compose.debug.yml
services:
  web:
    environment:
      - LOG_LEVEL=DEBUG
    command: streamlit run web/app.py --logger.level=debug
```

```bash
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
```

---

## セキュリティ

### Basic認証追加

Streamlitでは標準でBasic認証がサポートされていないため、Nginxでプロキシ設定:

```nginx
location / {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8501;
}
```

```bash
# パスワード生成
sudo htpasswd -c /etc/nginx/.htpasswd username
```

### ファイアウォール

```bash
# 必要なポートのみ開放
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8501/tcp   # Streamlitへの直接アクセスを拒否
sudo ufw enable
```

---

## パフォーマンスチューニング

### Streamlit設定

`web/.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

# キャッシュ設定
[runner]
magicEnabled = false
```

### Docker最適化

マルチステージビルドで軽量化:

```dockerfile
# ビルドステージ
FROM python:3.11 as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 実行ステージ
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["streamlit", "run", "web/app.py"]
```

---

## サポート

問題が発生した場合:

1. [GitHub Issues](repository-url/issues) で検索
2. ログファイルを確認
3. 新しいIssueを作成（ログを添付）
