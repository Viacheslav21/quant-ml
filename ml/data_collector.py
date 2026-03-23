"""
ML Data Collector — fetches historical closed markets from Polymarket
and builds training dataset with price features at different timepoints.
"""

import asyncio
import json
import logging
from bisect import bisect_left
from datetime import datetime, timezone, timedelta

import httpx
import numpy as np

log = logging.getLogger("ml_collector")

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DAYS_BEFORE = [14, 7, 3, 1, 0]

THEME_KEYWORDS = {
    "iran":     ["iran","iranian","tehran","nuclear iran","iaea"],
    "oil":      ["oil","opec","crude","brent","wti","petroleum"],
    "war":      ["war","attack","strike","invasion","missile","nuclear"],
    "peace":    ["ceasefire","peace","deal","agreement","surrender"],
    "ukraine":  ["ukraine","zelensky","donbas","crimea"],
    "russia":   ["russia","putin","kremlin","moscow"],
    "crypto":   ["bitcoin","btc","crypto","ethereum","blockchain"],
    "fed":      ["federal reserve","powell","rate","inflation","cpi"],
    "china":    ["china","taiwan","beijing","xi jinping"],
    "trump":    ["trump","executive order","tariff","maga"],
    "gold":     ["gold","xau","precious metal"],
    "election": ["election","vote","president","congress","senate"],
    "israel":   ["israel","hamas","gaza","hezbollah","netanyahu"],
}


def detect_theme(question: str) -> str:
    lower = question.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return theme
    return "other"


class MLDataCollector:
    def __init__(self, db):
        self.db = db
        self.gamma = httpx.AsyncClient(timeout=15.0)
        self.clob = httpx.AsyncClient(timeout=15.0)
        self._sem = asyncio.Semaphore(30)

    async def collect(self, max_markets: int = 5000) -> int:
        existing = await self.db.get_training_count()
        log.info(f"[ML] Starting collection. Existing samples: {existing}")

        markets = await self._fetch_closed_markets(max_markets)
        log.info(f"[ML] Fetched {len(markets)} closed markets from Gamma API")

        total = 0
        batch_size = 50
        for i in range(0, len(markets), batch_size):
            batch = markets[i:i + batch_size]
            tasks = [self._process_market(m) for m in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_samples = []
            for r in results:
                if isinstance(r, list):
                    batch_samples.extend(r)
            if batch_samples:
                inserted = await self.db.save_training_batch(batch_samples)
                total += inserted
            log.info(f"[ML] Progress: {min(i + batch_size, len(markets))}/{len(markets)} markets, {total} samples saved")
            await asyncio.sleep(0.5)

        final_count = await self.db.get_training_count()
        log.info(f"[ML] Done. Total samples in DB: {final_count} (+{total} new)")
        return total

    async def _fetch_closed_markets(self, limit: int) -> list:
        markets = []
        offset = 0
        while len(markets) < limit:
            try:
                r = await self.gamma.get(f"{GAMMA_API}/markets", params={
                    "closed": "true", "order": "volume", "ascending": "false",
                    "limit": 100, "offset": offset,
                })
                batch = r.json() or []
                if not batch:
                    break
                for m in batch:
                    prices = m.get("outcomePrices", ["0.5", "0.5"])
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    yes_final = float(prices[0])
                    if yes_final >= 0.95:
                        outcome = 1
                    elif yes_final <= 0.05:
                        outcome = 0
                    else:
                        continue

                    token_ids = m.get("clobTokenIds", [])
                    if isinstance(token_ids, str):
                        token_ids = json.loads(token_ids)
                    if not token_ids:
                        continue

                    end_date = m.get("endDate")
                    if not end_date:
                        continue

                    vol = float(m.get("volume") or 0)
                    if vol < 100:
                        continue

                    question = m.get("question", "")
                    markets.append({
                        "id": m["id"],
                        "question": question,
                        "token_id": token_ids[0],
                        "outcome": outcome,
                        "volume": vol,
                        "neg_risk": bool(m.get("negRisk")),
                        "end_date": end_date,
                        "created_at": m.get("createdAt"),
                        "theme": detect_theme(question),
                        "spread": float(m.get("spread") or 0),
                        "question_length": len(question),
                        "has_numbers": any(c.isdigit() for c in question),
                    })
                offset += 100
                if len(batch) < 100:
                    break
            except Exception as e:
                log.warning(f"[ML] Gamma fetch error at offset {offset}: {e}")
                break
        return markets[:limit]

    async def _fetch_price_history(self, token_id: str) -> list:
        async with self._sem:
            try:
                r = await self.clob.get(f"{CLOB_API}/prices-history", params={
                    "market": token_id, "interval": "all", "fidelity": "60",
                })
                data = r.json()
                return data.get("history", [])
            except Exception as e:
                log.debug(f"[ML] CLOB error for {token_id[:16]}: {e}")
                return []

    async def _process_market(self, market: dict) -> list:
        try:
            history = await self._fetch_price_history(market["token_id"])
            if len(history) < 10:
                return []

            timestamps = [h["t"] for h in history]
            prices = [float(h["p"]) for h in history]

            end_dt = datetime.fromisoformat(market["end_date"].replace("Z", "+00:00"))
            created_dt = None
            if market.get("created_at"):
                try:
                    created_dt = datetime.fromisoformat(market["created_at"].replace("Z", "+00:00"))
                except Exception:
                    pass

            samples = []
            for days_before in DAYS_BEFORE:
                target_dt = end_dt - timedelta(days=days_before)
                target_ts = target_dt.timestamp()

                price = self._find_closest_price(timestamps, prices, target_ts)
                if price is None or price >= 0.95 or price <= 0.05:
                    continue

                features = self._compute_features(timestamps, prices, target_ts, market, created_dt)
                # mispricing = how wrong was the market? (outcome - price)
                mispricing = round(market["outcome"] - price, 4)

                features.update({
                    "market_id": market["id"],
                    "question": market["question"][:200],
                    "theme": market["theme"],
                    "outcome": market["outcome"],
                    "days_before_expiry": days_before,
                    "yes_price": round(price, 4),
                    "volume": market["volume"],
                    "neg_risk": market["neg_risk"],
                    "question_length": market.get("question_length", 0),
                    "has_numbers": market.get("has_numbers", False),
                    "spread": market.get("spread", 0),
                    "mispricing": mispricing,
                })
                samples.append(features)
            return samples
        except Exception as e:
            log.debug(f"[ML] Process error for {market['id']}: {e}")
            return []

    def _find_closest_price(self, timestamps, prices, target_ts, tolerance_hours=12):
        if not timestamps:
            return None
        idx = bisect_left(timestamps, target_ts)
        candidates = []
        if idx > 0:
            candidates.append((abs(timestamps[idx - 1] - target_ts), prices[idx - 1]))
        if idx < len(timestamps):
            candidates.append((abs(timestamps[idx] - target_ts), prices[idx]))
        if not candidates:
            return None
        best_diff, best_price = min(candidates, key=lambda x: x[0])
        if best_diff > tolerance_hours * 3600:
            return None
        return best_price

    def _compute_features(self, timestamps, prices, target_ts, market, created_dt):
        market_age_days = None
        if created_dt:
            market_age_days = (datetime.fromtimestamp(target_ts, tz=timezone.utc) - created_dt).total_seconds() / 86400

        current = self._find_closest_price(timestamps, prices, target_ts)

        # Momentum: use actual price differences from history
        # These match what scanner provides at inference time (price_change_1wk/1mo)
        mom_7d = self._find_closest_price(timestamps, prices, target_ts - 7 * 86400)
        mom_1d = self._find_closest_price(timestamps, prices, target_ts - 86400)

        price_momentum_7d = round(current - mom_7d, 4) if current is not None and mom_7d is not None else None
        price_momentum_1d = round(current - mom_1d, 4) if current is not None and mom_1d is not None else None

        # Volatility: std of prices in window before T
        # At inference: computable from price_snapshots table (we store snapshots every scan)
        window_start = target_ts - 7 * 86400
        window_prices = [p for t, p in zip(timestamps, prices) if window_start <= t <= target_ts]
        price_volatility_7d = round(float(np.std(window_prices)), 4) if len(window_prices) >= 3 else None

        volume_per_day = round(market["volume"] / market_age_days, 2) if market_age_days and market_age_days > 0 else None
        price_distance_50 = round(abs(current - 0.5), 4) if current is not None else None

        return {
            "market_age_days": round(market_age_days, 1) if market_age_days else None,
            "price_momentum_7d": price_momentum_7d,
            "price_momentum_1d": price_momentum_1d,
            "price_volatility_7d": price_volatility_7d,
            "volume_per_day": volume_per_day,
            "price_distance_50": price_distance_50,
        }

    async def close(self):
        await self.gamma.aclose()
        await self.clob.aclose()
