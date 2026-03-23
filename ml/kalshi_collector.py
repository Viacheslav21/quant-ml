"""Kalshi data collector — fetches settled markets from Kalshi API."""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta

import httpx

log = logging.getLogger("kalshi_collector")

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

THEME_KEYWORDS = {
    "crypto": ["bitcoin", "btc", "crypto", "ethereum", "blockchain"],
    "oil": ["oil", "crude", "brent", "wti", "petroleum"],
    "gold": ["gold", "xau"],
    "fed": ["federal reserve", "fed funds", "interest rate", "inflation", "cpi"],
    "election": ["election", "vote", "president", "congress", "senate"],
    "trump": ["trump", "executive order", "tariff"],
    "war": ["war", "attack", "strike", "invasion", "missile"],
    "china": ["china", "taiwan", "beijing"],
    "ukraine": ["ukraine", "zelensky"],
    "russia": ["russia", "putin"],
    "israel": ["israel", "hamas", "gaza"],
    "iran": ["iran", "iranian", "tehran"],
    "sp500": ["s&p 500", "sp500", "spx"],
}


def detect_theme(question: str) -> str:
    lower = question.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return theme
    return "other"


class KalshiCollector:
    def __init__(self, db):
        self.db = db
        self.client = httpx.AsyncClient(timeout=15.0)

    async def collect(self, max_markets: int = 40000) -> int:
        log.info(f"[KALSHI] Starting collection, max={max_markets}")
        markets = await self._fetch_settled(max_markets)
        log.info(f"[KALSHI] Fetched {len(markets)} settled markets")

        samples = []
        for m in markets:
            s = self._to_sample(m)
            if s:
                samples.append(s)

        if samples:
            inserted = await self.db.save_training_batch(samples)
            log.info(f"[KALSHI] Saved {inserted} samples")
            return inserted
        return 0

    async def _fetch_settled(self, limit: int) -> list:
        markets = []
        cursor = None
        while len(markets) < limit:
            params = {"limit": 200, "status": "settled"}
            if cursor:
                params["cursor"] = cursor
            try:
                r = await self.client.get(f"{KALSHI_API}/markets", params=params)
                data = r.json()
                batch = data.get("markets", [])
                if not batch:
                    break
                markets.extend(batch)
                cursor = data.get("cursor")
                if not cursor:
                    break
                if len(markets) % 2000 == 0:
                    log.info(f"[KALSHI] Fetched {len(markets)} markets...")
            except Exception as e:
                log.warning(f"[KALSHI] Fetch error: {e}")
                break
        return markets[:limit]

    def _to_sample(self, m: dict) -> dict | None:
        result = m.get("result")
        if result not in ("yes", "no"):
            return None

        title = m.get("title") or m.get("yes_sub_title") or ""
        if not title or len(title) < 10:
            return None

        outcome = 1 if result == "yes" else 0
        last_price = float(m.get("last_price_dollars") or 0)
        if last_price <= 0.01 or last_price >= 0.99:
            return None  # already resolved at extreme

        open_interest = float(m.get("open_interest_fp") or 0)
        if open_interest < 10:
            return None  # too thin

        # Parse dates
        close_time = m.get("close_time") or m.get("expiration_time")
        created_time = m.get("created_time")
        if not close_time or not created_time:
            return None

        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            created_dt = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
            market_age_days = (close_dt - created_dt).total_seconds() / 86400
        except Exception:
            return None

        if market_age_days < 0.01:
            return None

        theme = detect_theme(title)
        has_numbers = any(c.isdigit() for c in title)
        price_distance_50 = round(abs(last_price - 0.5), 4)

        return {
            "market_id": f"kalshi_{m.get('ticker', '')}",
            "question": title[:200],
            "theme": theme,
            "outcome": outcome,
            "days_before_expiry": 0,  # last_price is at close
            "yes_price": round(last_price, 4),
            "volume": open_interest,  # use open_interest as volume proxy
            "neg_risk": False,
            "market_age_days": round(market_age_days, 1),
            "price_momentum_7d": None,
            "price_momentum_1d": None,
            "price_volatility_7d": None,
            "volume_per_day": round(open_interest / max(market_age_days, 1), 2),
            "price_distance_50": price_distance_50,
            "question_length": len(title),
            "has_numbers": has_numbers,
            "spread": float(m.get("spread", 0) if m.get("spread") else 0),
            "mispricing": round(outcome - last_price, 4),
        }

    async def close(self):
        await self.client.aclose()
