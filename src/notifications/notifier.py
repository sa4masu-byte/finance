"""
通知機能

対応:
- LINE Notify
- Slack Webhook
- Discord Webhook
"""
import os
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime


class LineNotifier:
    """LINE Messaging API通知"""

    def __init__(self, channel_token: str = None, user_id: str = None):
        self.channel_token = channel_token or os.environ.get("LINE_CHANNEL_TOKEN")
        self.user_id = user_id or os.environ.get("LINE_USER_ID")
        self.api_url = "https://api.line.me/v2/bot/message/push"

    def send(self, message: str) -> bool:
        if not self.channel_token:
            print("⚠️ LINE_CHANNEL_TOKEN が設定されていません")
            return False
        if not self.user_id:
            print("⚠️ LINE_USER_ID が設定されていません")
            return False

        headers = {
            "Authorization": f"Bearer {self.channel_token}",
            "Content-Type": "application/json"
        }
        data = {
            "to": self.user_id,
            "messages": [{"type": "text", "text": message}]
        }

        try:
            r = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            return r.status_code == 200
        except Exception as e:
            print(f"LINE送信エラー: {e}")
            return False

    @property
    def token(self):
        """後方互換性のため"""
        return self.channel_token


class SlackNotifier:
    """Slack Webhook通知"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def send(self, message: str, blocks: List[Dict] = None) -> bool:
        if not self.webhook_url:
            print("⚠️ SLACK_WEBHOOK_URL が設定されていません")
            return False

        payload = {"text": message}
        if blocks:
            payload["blocks"] = blocks

        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            return r.status_code == 200
        except Exception as e:
            print(f"Slack送信エラー: {e}")
            return False


class DiscordNotifier:
    """Discord Webhook通知"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")

    def send(self, message: str) -> bool:
        if not self.webhook_url:
            print("⚠️ DISCORD_WEBHOOK_URL が設定されていません")
            return False

        payload = {"content": message}

        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            return r.status_code in [200, 204]
        except Exception as e:
            print(f"Discord送信エラー: {e}")
            return False


class Notifier:
    """統合通知クラス"""

    def __init__(self):
        self.line = LineNotifier()
        self.slack = SlackNotifier()
        self.discord = DiscordNotifier()

    def format_signals(self, signals: List[Dict]) -> str:
        """シグナルをメッセージ形式に変換"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        if not signals:
            return f"""
📊 株式推奨レポート
━━━━━━━━━━━━━━━
{now}

⚠️ 本日の推奨銘柄はありません
"""

        msg = f"""
📊 株式推奨レポート
━━━━━━━━━━━━━━━
{now}

🎯 推奨銘柄: {len(signals)}件
"""
        for i, s in enumerate(signals, 1):
            msg += f"""
【{i}】{s['symbol']}
  現在値: ¥{s['price']:,.0f}
  損切り: ¥{s['stop_loss']:,.0f}
  利確:   ¥{s['target']:,.0f}
  RSI: {s['rsi']:.1f}
"""

        msg += """
━━━━━━━━━━━━━━━
📋 ルール:
• 推奨価格付近で買い
• 損切りライン厳守
• 最大10日保有
"""
        return msg

    def notify_all(self, signals: List[Dict]) -> Dict[str, bool]:
        """全チャンネルに通知"""
        message = self.format_signals(signals)
        results = {}

        # LINE
        if self.line.channel_token and self.line.user_id:
            results["line"] = self.line.send(message)

        # Slack
        if self.slack.webhook_url:
            results["slack"] = self.slack.send(message)

        # Discord
        if self.discord.webhook_url:
            results["discord"] = self.discord.send(message)

        return results

    def notify(self, signals: List[Dict], channels: List[str] = None) -> Dict[str, bool]:
        """指定チャンネルに通知"""
        message = self.format_signals(signals)
        results = {}

        channels = channels or ["line", "slack", "discord"]

        if "line" in channels and self.line.channel_token and self.line.user_id:
            results["line"] = self.line.send(message)

        if "slack" in channels and self.slack.webhook_url:
            results["slack"] = self.slack.send(message)

        if "discord" in channels and self.discord.webhook_url:
            results["discord"] = self.discord.send(message)

        return results
