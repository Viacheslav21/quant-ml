#!/usr/bin/env python3
"""
QUANT ML — Data collection, model training, and inference service.

Modes:
  python main.py collect   — Fetch historical data from Polymarket
  python main.py train     — Train XGBoost model on collected data
  python main.py serve     — Start inference API (FastAPI)
  python main.py all       — Collect + Train + Serve
"""

import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("quant-ml")


async def run_collect() -> dict:
    from utils.db import Database
    from ml.data_collector import MLDataCollector
    from ml.kalshi_collector import KalshiCollector
    from ml.manifold_collector import ManifoldCollector

    db = Database()
    await db.init()
    stats = {"polymarket": 0, "kalshi": 0, "manifold": 0}
    try:
        log.info("=" * 40 + " POLYMARKET " + "=" * 40)
        poly = MLDataCollector(db)
        stats["polymarket"] = await poly.collect(max_markets=40000)
        await poly.close()

        log.info("=" * 40 + " KALSHI " + "=" * 40)
        kalshi = KalshiCollector(db)
        stats["kalshi"] = await kalshi.collect(max_markets=40000)
        await kalshi.close()

        log.info("=" * 40 + " MANIFOLD " + "=" * 40)
        manifold = ManifoldCollector(db)
        stats["manifold"] = await manifold.collect(max_markets=20000)
        await manifold.close()

        stats["total_in_db"] = await db.get_training_count()
        stats["new_total"] = sum(v for k, v in stats.items() if k not in ("total_in_db",))
        log.info(f"Collection complete: {stats['new_total']} new, {stats['total_in_db']} total in DB")
    finally:
        await db.close()
    return stats


async def run_train() -> dict:
    from utils.db import Database
    from ml.signal_model import SignalModel

    db = Database()
    await db.init()
    try:
        samples = await db.get_training_data()
        if not samples:
            log.error("No training data. Run 'python main.py collect' first.")
            return {"error": "no data"}

        model = SignalModel()
        metrics = model.train(samples)

        if "error" in metrics:
            log.error(f"Training failed: {metrics['error']}")
            return metrics

        model_bytes = model.save_bytes()
        await db.save_model(model_bytes, metrics)
        log.info(f"Model saved to DB ({len(model_bytes)} bytes)")
        model.save_file("model.json")

        log.info("=" * 50)
        log.info(f"  Samples:     {metrics['n_total']} (train: {metrics['n_train']}, test: {metrics['n_test']})")
        log.info(f"  YES rate:    {metrics['yes_rate']:.1%}")
        log.info(f"  Test Brier:  {metrics['test_brier']:.4f}")
        log.info(f"  Market Brier:{metrics.get('market_brier', 'N/A')}")
        log.info(f"  Improvement: {metrics.get('brier_improvement', 'N/A')}")
        log.info(f"  Accuracy:    {metrics['test_accuracy']:.1%}")
        log.info(f"  Mispricing:  Acc={metrics.get('mis_accuracy', 'N/A')} Rate={metrics.get('mis_rate', 'N/A')}")
        if "filtered_accuracy" in metrics:
            log.info(f"  Filtered:    {metrics['filtered_count']} trades → Acc={metrics['filtered_accuracy']:.1%}")
        log.info("=" * 50)
        return metrics
    finally:
        await db.close()


async def run_serve():
    import uvicorn
    from fastapi import FastAPI, Query
    from utils.db import Database
    from ml.signal_model import SignalModel

    app = FastAPI(title="Quant ML", docs_url="/")
    model = SignalModel()
    db = Database()

    @app.on_event("startup")
    async def startup():
        await db.init()
        model_bytes, metrics = await db.load_model()
        if model_bytes:
            model.load_bytes(model_bytes)
            log.info(f"[SERVE] Model loaded. Metrics: Brier={metrics.get('test_brier')}, Acc={metrics.get('test_accuracy')}")
        else:
            log.warning("[SERVE] No model in DB — predictions will return defaults")

    @app.on_event("shutdown")
    async def shutdown():
        await db.close()

    @app.get("/predict")
    async def predict(
        yes_price: float = Query(...),
        theme: str = Query("other"),
        volume: float = Query(0),
        days_to_expiry: int = Query(7),
        market_age_days: float = Query(None),
        price_momentum_7d: float = Query(None),
        price_momentum_1d: float = Query(None),
        price_volatility_7d: float = Query(None),
        volume_per_day: float = Query(None),
        neg_risk: bool = Query(False),
        question_length: int = Query(50),
        has_numbers: bool = Query(False),
        spread: float = Query(None),
        # New v2 features from engine
        hurst: float = Query(None),
        book_imbalance: float = Query(None),
        contrarian_conf: float = Query(None),
        n_evidence: int = Query(None),
        volume_ratio: float = Query(None),
    ):
        features = {
            "yes_price": yes_price,
            "theme": theme,
            "volume": volume,
            "days_before_expiry": days_to_expiry,
            "market_age_days": market_age_days,
            "price_momentum_7d": price_momentum_7d,
            "price_momentum_1d": price_momentum_1d,
            "price_volatility_7d": price_volatility_7d,
            "volume_per_day": volume_per_day,
            "neg_risk": neg_risk,
            "question_length": question_length,
            "has_numbers": has_numbers,
            "spread": spread,
            "hurst": hurst,
            "book_imbalance": book_imbalance,
            "contrarian_conf": contrarian_conf,
            "n_evidence": n_evidence,
            "volume_ratio": volume_ratio,
        }
        result = model.predict(features)
        return result

    @app.get("/health")
    async def health():
        has_model = model.model is not None
        has_mispricing = model.mispricing_model is not None
        count = await db.get_training_count()
        return {
            "status": "ok" if has_model else "no_model",
            "model_loaded": has_model,
            "mispricing_loaded": has_mispricing,
            "training_samples": count,
        }

    port = int(os.getenv("PORT", "8080"))
    log.info(f"[SERVE] Starting on port {port}")
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"
    log.info(f"QUANT ML — mode: {mode}")

    if mode == "collect":
        await run_collect()
    elif mode == "train":
        await run_train()
    elif mode == "serve":
        await run_serve()
    elif mode == "all":
        from utils.telegram import TelegramBot
        tg = TelegramBot()

        collect_stats = await run_collect()
        train_metrics = await run_train()

        # Send summary to Telegram
        cs = collect_stats or {}
        tm = train_metrics or {}
        summary = (
            f"🤖 <b>Quant ML — Training Complete</b>\n\n"
            f"📊 <b>Data Collection</b>\n"
            f"  Polymarket: {cs.get('polymarket', 0)} new\n"
            f"  Kalshi: {cs.get('kalshi', 0)} new\n"
            f"  Manifold: {cs.get('manifold', 0)} new\n"
            f"  Total in DB: <b>{cs.get('total_in_db', '?')}</b>\n\n"
        )
        if "error" not in tm:
            improvement = tm.get('brier_improvement', 0)
            imp_emoji = "✅" if improvement > 0 else "⚠️"
            summary += (
                f"🧠 <b>Model Results</b>\n"
                f"  Samples: {tm.get('n_total', '?')} (train: {tm.get('n_train', '?')}, test: {tm.get('n_test', '?')})\n"
                f"  Model Brier: <b>{tm.get('test_brier', '?')}</b>\n"
                f"  Market Brier: {tm.get('market_brier', '?')}\n"
                f"  {imp_emoji} Improvement: <b>{improvement:+.4f}</b>\n"
                f"  Accuracy: <b>{tm.get('test_accuracy', 0):.1%}</b>\n\n"
                f"🔍 <b>Mispricing Model</b>\n"
                f"  Accuracy: {tm.get('mis_accuracy', '?')}\n"
                f"  Mispricing rate: {tm.get('mis_rate', '?')}\n"
            )
            if "filtered_accuracy" in tm:
                summary += f"  Filtered: {tm['filtered_count']} trades → <b>{tm['filtered_accuracy']:.1%}</b> acc\n"
            top_feats = list(tm.get("feature_importance", {}).items())[:3]
            if top_feats:
                summary += "\n📈 Top features: " + ", ".join(f"{k}={v:.2f}" for k, v in top_feats)
        else:
            summary += f"❌ Training failed: {tm.get('error', 'unknown')}"

        await tg.send(summary)
        await tg.close()
        log.info("[ML] Telegram summary sent")

        await run_serve()
    else:
        log.error(f"Unknown mode: {mode}. Use: collect, train, serve, all")


if __name__ == "__main__":
    asyncio.run(main())
