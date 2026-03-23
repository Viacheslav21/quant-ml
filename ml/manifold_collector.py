"""Manifold Markets data collector — fetches resolved markets from Manifold API."""

import asyncio
import logging
from datetime import datetime, timezone

import httpx

log = logging.getLogger("manifold_collector")

MANIFOLD_API = "https://api.manifold.markets/v0"

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
    "ai": ["ai", "artificial intelligence", "gpt", "openai", "anthropic"],
}


def detect_theme(question: str) -> str:
    lower = question.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return theme
    return "other"


class ManifoldCollector:
    def __init__(self, db):
        self.db = db
        self.client = httpx.AsyncClient(timeout=15.0)

    async def collect(self, max_markets: int = 20000) -> int:
        log.info(f"[MANIFOLD] Starting collection, max={max_markets}")
        markets = await self._fetch_resolved(max_markets)
        log.info(f"[MANIFOLD] Fetched {len(markets)} resolved markets")

        samples = []
        for m in markets:
            s = self._to_sample(m)
            if s:
                samples.append(s)

        if samples:
            inserted = await self.db.save_training_batch(samples)
            log.info(f"[MANIFOLD] Saved {inserted} samples")
            return inserted
        return 0

    async def _fetch_resolved(self, limit: int) -> list:
        markets = []
        before = None
        while len(markets) < limit:
            params = {"limit": 500, "filter": "resolved", "sort": "last-updated"}
            if before:
                params["before"] = before
            try:
                r = await self.client.get(f"{MANIFOLD_API}/search-markets", params=params)
                batch = r.json()
                if not isinstance(batch, list) or not batch:
                    break
                markets.extend(batch)
                # Use last item's ID as cursor
                before = batch[-1].get("id")
                if len(markets) % 5000 == 0:
                    log.info(f"[MANIFOLD] Fetched {len(markets)} markets...")
                await asyncio.sleep(0.15)  # rate limit: 500 req/min
            except Exception as e:
                log.warning(f"[MANIFOLD] Fetch error: {e}")
                break
        return markets[:limit]

    def _to_sample(self, m: dict) -> dict | None:
        if not m.get("isResolved"):
            return None
        if m.get("outcomeType") != "BINARY":
            return None

        resolution = m.get("resolution")
        if resolution == "YES":
            outcome = 1
        elif resolution == "NO":
            outcome = 0
        else:
            return None

        probability = m.get("probability")
        if probability is None or probability <= 0.01 or probability >= 0.99:
            return None

        question = m.get("question", "")
        if not question or len(question) < 10:
            return None

        volume = float(m.get("volume") or 0)
        bettors = m.get("uniqueBettorCount") or 0
        if bettors < 3:
            return None  # too few participants

        # Parse dates
        created_ts = m.get("createdTime")
        close_ts = m.get("closeTime") or m.get("resolutionTime")
        if not created_ts or not close_ts:
            return None

        try:
            created_dt = datetime.fromtimestamp(created_ts / 1000, tz=timezone.utc)
            close_dt = datetime.fromtimestamp(close_ts / 1000, tz=timezone.utc)
            market_age_days = (close_dt - created_dt).total_seconds() / 86400
        except Exception:
            return None

        if market_age_days < 0.01:
            return None

        theme = detect_theme(question)
        has_numbers = any(c.isdigit() for c in question)

        return {
            "market_id": f"manifold_{m.get('id', '')}",
            "question": question[:200],
            "theme": theme,
            "outcome": outcome,
            "days_before_expiry": 0,
            "yes_price": round(probability, 4),
            "volume": volume,
            "neg_risk": False,
            "market_age_days": round(market_age_days, 1),
            "price_momentum_7d": None,
            "price_momentum_1d": None,
            "price_volatility_7d": None,
            "volume_per_day": round(volume / max(market_age_days, 1), 2),
            "price_distance_50": round(abs(probability - 0.5), 4),
            "question_length": len(question),
            "has_numbers": has_numbers,
            "spread": None,
            "mispricing": round(outcome - probability, 4),
        }

    async def close(self):
        await self.client.aclose()
