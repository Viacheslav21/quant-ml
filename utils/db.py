import logging
import os
import asyncpg

log = logging.getLogger("db")


class Database:
    def __init__(self):
        self.pool = None

    async def init(self):
        self.pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"), min_size=2, max_size=5)
        await self._create_schema()
        await self._migrate()
        log.info("[DB] ML database connected")

    async def _migrate(self):
        """Add new columns to existing tables."""
        migrations = [
            "ALTER TABLE ml_training_data ADD COLUMN IF NOT EXISTS question_length INTEGER",
            "ALTER TABLE ml_training_data ADD COLUMN IF NOT EXISTS has_numbers BOOLEAN DEFAULT FALSE",
            "ALTER TABLE ml_training_data ADD COLUMN IF NOT EXISTS spread REAL",
            "ALTER TABLE ml_training_data ADD COLUMN IF NOT EXISTS mispricing REAL",
        ]
        async with self.pool.acquire() as conn:
            for sql in migrations:
                try:
                    await conn.execute(sql)
                except Exception:
                    pass

    async def _create_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_training_data (
                    id BIGSERIAL PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    question TEXT,
                    theme TEXT DEFAULT 'other',
                    outcome INTEGER NOT NULL,
                    days_before_expiry INTEGER NOT NULL,
                    yes_price REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    neg_risk BOOLEAN DEFAULT FALSE,
                    market_age_days REAL,
                    price_momentum_7d REAL,
                    price_momentum_1d REAL,
                    price_volatility_7d REAL,
                    volume_per_day REAL,
                    price_distance_50 REAL,
                    collected_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(market_id, days_before_expiry)
                );
                CREATE TABLE IF NOT EXISTS ml_models (
                    id TEXT PRIMARY KEY DEFAULT 'main',
                    model_data BYTEA,
                    metrics JSONB,
                    trained_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_ml_training_outcome ON ml_training_data(outcome);
                CREATE INDEX IF NOT EXISTS idx_ml_training_theme ON ml_training_data(theme);
            """)
            # Migrations for new columns
            cols = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name='ml_training_data'"
            )
            col_names = {r["column_name"] for r in cols}
            for col, typ in [
                ("question_length", "INTEGER"),
                ("has_numbers", "BOOLEAN DEFAULT FALSE"),
                ("spread", "REAL"),
                ("mispricing", "REAL"),
            ]:
                if col.split()[0] not in col_names:
                    await conn.execute(f"ALTER TABLE ml_training_data ADD COLUMN {col} {typ}")
                    log.info(f"[DB] Added {col} to ml_training_data")

    async def save_training_batch(self, samples: list) -> int:
        if not samples:
            return 0
        inserted = 0
        async with self.pool.acquire() as conn:
            for s in samples:
                try:
                    await conn.execute("""
                        INSERT INTO ml_training_data
                            (market_id, question, theme, outcome, days_before_expiry,
                             yes_price, volume, neg_risk, market_age_days,
                             price_momentum_7d, price_momentum_1d, price_volatility_7d,
                             volume_per_day, price_distance_50,
                             question_length, has_numbers, spread, mispricing)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
                        ON CONFLICT (market_id, days_before_expiry) DO NOTHING
                    """, s["market_id"], s.get("question"), s.get("theme", "other"),
                        s["outcome"], s["days_before_expiry"],
                        s["yes_price"], s.get("volume", 0), s.get("neg_risk", False),
                        s.get("market_age_days"), s.get("price_momentum_7d"),
                        s.get("price_momentum_1d"), s.get("price_volatility_7d"),
                        s.get("volume_per_day"), s.get("price_distance_50"),
                        s.get("question_length"), s.get("has_numbers", False),
                        s.get("spread"), s.get("mispricing"))
                    inserted += 1
                except Exception:
                    pass
        return inserted

    async def get_training_data(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM ml_training_data ORDER BY id")
            return [dict(r) for r in rows]

    async def get_training_count(self) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM ml_training_data")

    async def save_model(self, model_bytes: bytes, metrics: dict):
        """Save trained model to DB."""
        import json

        def _convert(obj):
            if hasattr(obj, "item"):
                return obj.item()  # numpy scalar → python scalar
            raise TypeError(f"Not serializable: {type(obj)}")

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ml_models (id, model_data, metrics, trained_at)
                VALUES ('main', $1, $2, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    model_data = EXCLUDED.model_data,
                    metrics = EXCLUDED.metrics,
                    trained_at = NOW()
            """, model_bytes, json.dumps(metrics, default=_convert))

    async def load_model(self) -> tuple:
        """Load model from DB. Returns (model_bytes, metrics_dict) or (None, None)."""
        import json
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT model_data, metrics FROM ml_models WHERE id = 'main'")
            if row and row["model_data"]:
                return bytes(row["model_data"]), json.loads(row["metrics"]) if row["metrics"] else {}
            return None, None

    async def close(self):
        if self.pool:
            await self.pool.close()
