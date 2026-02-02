FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# システムパッケージ更新
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt をコピーして依存関係インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# APSchedulerをインストール（スケジューラー用）
RUN pip install --no-cache-dir APScheduler

# アプリケーションファイルをコピー
COPY . .

# Streamlit設定ディレクトリ作成
RUN mkdir -p ~/.streamlit

# Streamlit設定をコピー
COPY web/.streamlit/config.toml ~/.streamlit/

# データディレクトリ作成
RUN mkdir -p /app/data/reports /app/data/stock_cache

# ポート公開
EXPOSE 8501

# デフォルトコマンド（Streamlit起動）
CMD ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
