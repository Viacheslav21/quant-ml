#!/usr/bin/env python3
"""
QUANT ML — XGBoost inference server with collect/train API.
Always runs as HTTP server. Training triggered via /api/train endpoint (from dashboard).
"""

import asyncio
import logging
import os
import json as _json
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("quant-ml")

from utils.db import Database
from ml.signal_model import SignalModel

# ── Global state ──
app = FastAPI(title="Quant ML", docs_url="/")
model = SignalModel()
db = Database()

_training_status = {
    "running": False,
    "phase": "idle",       # idle, collecting_polymarket, collecting_kalshi, collecting_manifold, training, done, error
    "progress": "",
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


# ── Startup / Shutdown ──

@app.on_event("startup")
async def startup():
    await db.init()
    model_bytes, metrics = await db.load_model()
    if model_bytes:
        model.load_bytes(model_bytes)
        log.info(f"[SERVE] Model loaded. Brier={metrics.get('test_brier')}, Acc={metrics.get('test_accuracy')}")
    else:
        log.warning("[SERVE] No model in DB")


@app.on_event("shutdown")
async def shutdown():
    await db.close()


# ── Predict endpoint ──

@app.get("/predict")
async def predict(
    yes_price: float = Query(...), theme: str = Query("other"),
    volume: float = Query(0), days_to_expiry: int = Query(7),
    market_age_days: float = Query(None), price_momentum_7d: float = Query(None),
    price_momentum_1d: float = Query(None), price_volatility_7d: float = Query(None),
    volume_per_day: float = Query(None), neg_risk: bool = Query(False),
    question_length: int = Query(50), has_numbers: bool = Query(False),
    spread: float = Query(None), hurst: float = Query(None),
    book_imbalance: float = Query(None), contrarian_conf: float = Query(None),
    n_evidence: int = Query(None), volume_ratio: float = Query(None),
):
    features = {
        "yes_price": yes_price, "theme": theme, "volume": volume,
        "days_before_expiry": days_to_expiry, "market_age_days": market_age_days,
        "price_momentum_7d": price_momentum_7d, "price_momentum_1d": price_momentum_1d,
        "price_volatility_7d": price_volatility_7d, "volume_per_day": volume_per_day,
        "neg_risk": neg_risk, "question_length": question_length,
        "has_numbers": has_numbers, "spread": spread, "hurst": hurst,
        "book_imbalance": book_imbalance, "contrarian_conf": contrarian_conf,
        "n_evidence": n_evidence, "volume_ratio": volume_ratio,
    }
    return model.predict(features)


# ── Health / Status ──

@app.get("/health")
async def health():
    count = await db.get_training_count()
    return {
        "status": "training" if _training_status["running"] else "ok",
        "model_loaded": model.model is not None,
        "mispricing_loaded": model.mispricing_model is not None,
        "training_samples": count,
    }


@app.get("/api/training-status")
async def training_status():
    return _training_status


# ── Train API (called from dashboard) ──

async def _run_collect_and_train():
    """Background task: collect data + train model."""
    global _training_status
    _training_status["running"] = True
    _training_status["started_at"] = datetime.now(timezone.utc).isoformat()
    _training_status["error"] = None
    _training_status["result"] = None

    stats = {"polymarket": 0, "kalshi": 0, "manifold": 0}

    try:
        # 1. Collect Polymarket
        _training_status["phase"] = "collecting_polymarket"
        _training_status["progress"] = "Starting Polymarket..."
        from ml.data_collector import MLDataCollector
        poly = MLDataCollector(db)
        stats["polymarket"] = await poly.collect(max_markets=40000)
        await poly.close()
        _training_status["progress"] = f"Polymarket: {stats['polymarket']} new samples"

        # 2. Collect Kalshi
        _training_status["phase"] = "collecting_kalshi"
        _training_status["progress"] = "Starting Kalshi..."
        from ml.kalshi_collector import KalshiCollector
        kalshi = KalshiCollector(db)
        stats["kalshi"] = await kalshi.collect(max_markets=40000)
        await kalshi.close()
        _training_status["progress"] = f"Kalshi: {stats['kalshi']} new samples"

        # 3. Collect Manifold
        _training_status["phase"] = "collecting_manifold"
        _training_status["progress"] = "Starting Manifold..."
        from ml.manifold_collector import ManifoldCollector
        manifold = ManifoldCollector(db)
        stats["manifold"] = await manifold.collect(max_markets=20000)
        await manifold.close()
        _training_status["progress"] = f"Manifold: {stats['manifold']} new samples"

        # 4. Train
        _training_status["phase"] = "training"
        _training_status["progress"] = "Loading training data..."
        samples = await db.get_training_data()
        _training_status["progress"] = f"Training on {len(samples)} samples..."

        train_model = SignalModel()
        metrics = train_model.train(samples)

        if "error" in metrics:
            raise Exception(metrics["error"])

        # Save to DB
        model_bytes = train_model.save_bytes()
        await db.save_model(model_bytes, metrics)

        # Hot-reload into serving model
        model.load_bytes(model_bytes)

        total_in_db = await db.get_training_count()

        _training_status["phase"] = "done"
        _training_status["result"] = {
            **metrics,
            "collection": stats,
            "total_in_db": total_in_db,
        }
        _training_status["progress"] = f"Done! Brier={metrics['test_brier']:.4f}, Acc={metrics['test_accuracy']:.1%}"

        # Telegram notification
        try:
            from utils.telegram import TelegramBot
            tg = TelegramBot()
            improvement = metrics.get('brier_improvement', 0)
            await tg.send(
                f"🤖 <b>ML Training Complete</b>\n\n"
                f"📊 Data: {total_in_db} samples\n"
                f"  +Poly: {stats['polymarket']}, +Kalshi: {stats['kalshi']}, +Manifold: {stats['manifold']}\n\n"
                f"🧠 Brier: <b>{metrics['test_brier']:.4f}</b> (market: {metrics.get('market_brier', '?')})\n"
                f"{'✅' if improvement > 0 else '⚠️'} Improvement: <b>{improvement:+.4f}</b>\n"
                f"Accuracy: <b>{metrics['test_accuracy']:.1%}</b>\n"
                f"Mispricing: {metrics.get('mis_accuracy', '?')} acc, {metrics.get('mis_rate', '?')} rate"
            )
            await tg.close()
        except Exception as e:
            log.warning(f"[TRAIN] Telegram failed: {e}")

    except Exception as e:
        _training_status["phase"] = "error"
        _training_status["error"] = str(e)
        _training_status["progress"] = f"Error: {e}"
        log.error(f"[TRAIN] Failed: {e}", exc_info=True)
    finally:
        _training_status["running"] = False
        _training_status["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks):
    """Start collect + train in background. Returns immediately."""
    if _training_status["running"]:
        return JSONResponse({"error": "Training already running", "status": _training_status}, status_code=409)
    background_tasks.add_task(_run_collect_and_train)
    return {"message": "Training started", "status": "collecting_polymarket"}


@app.post("/api/train-only")
async def start_train_only(background_tasks: BackgroundTasks):
    """Train on existing data without collecting new. Fast."""
    if _training_status["running"]:
        return JSONResponse({"error": "Training already running"}, status_code=409)

    async def _train_only():
        global _training_status
        _training_status["running"] = True
        _training_status["phase"] = "training"
        _training_status["started_at"] = datetime.now(timezone.utc).isoformat()
        _training_status["error"] = None
        try:
            samples = await db.get_training_data()
            _training_status["progress"] = f"Training on {len(samples)} samples..."
            train_model = SignalModel()
            metrics = train_model.train(samples)
            if "error" in metrics:
                raise Exception(metrics["error"])
            model_bytes = train_model.save_bytes()
            await db.save_model(model_bytes, metrics)
            model.load_bytes(model_bytes)
            _training_status["phase"] = "done"
            _training_status["result"] = metrics
            _training_status["progress"] = f"Done! Brier={metrics['test_brier']:.4f}"
        except Exception as e:
            _training_status["phase"] = "error"
            _training_status["error"] = str(e)
            _training_status["progress"] = f"Error: {e}"
        finally:
            _training_status["running"] = False
            _training_status["finished_at"] = datetime.now(timezone.utc).isoformat()

    background_tasks.add_task(_train_only)
    return {"message": "Training started (existing data)", "status": "training"}


# ── Run ──

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    log.info(f"[ML] Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
