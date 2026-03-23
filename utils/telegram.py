import os
import logging
import httpx

log = logging.getLogger("telegram")


class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.client = httpx.AsyncClient(timeout=10.0)

    async def send(self, text: str):
        if not self.token or not self.chat_id:
            log.debug("[TG] No token/chat_id, skipping")
            return
        try:
            await self.client.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
            )
        except Exception as e:
            log.warning(f"[TG] Send failed: {e}")

    async def close(self):
        await self.client.aclose()
