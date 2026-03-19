# ============================================================
#  Urdu Fake News Detection System — FastAPI Backend
#  Author : Salar Ahmed | FYP 2025-2026
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import datetime

from model.predict import predict_news
from scraper.rss_scraper import scrape_rss
from utils.database import init_db, save_history, get_history, get_stats

app = FastAPI(
    title="Urdu Fake News Detection API",
    description="Real-time AI-powered fact verification for Pakistani social media",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Startup ──────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    init_db()
    print("✅ Database initialised")


# ── Schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    source: Optional[str] = "manual"


# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Urdu Fake News Detection API",
        "version": "1.0.0",
        "author": "Salar Ahmed",
        "endpoints": ["/search/{keyword}", "/predict", "/history", "/stats", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}


@app.get("/search/{keyword}")
def search(keyword: str, limit: int = 10):
    """Scrape live news matching keyword and classify each article."""
    try:
        articles = scrape_rss(keyword, limit=limit)
        results = []
        for article in articles:
            prediction = predict_news(article["title"] + " " + article.get("summary", ""))
            entry = {**article, **prediction}
            save_history(entry)
            results.append(entry)
        return {"keyword": keyword, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """Classify a manually pasted news article."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        prediction = predict_news(req.text)
        entry = {
            "title": req.text[:120],
            "source": req.source,
            "url": "",
            "published": datetime.datetime.utcnow().isoformat(),
            **prediction,
        }
        save_history(entry)
        return entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def history(limit: int = 50):
    """Return recent prediction history."""
    return {"history": get_history(limit)}


@app.get("/stats")
def stats():
    """Aggregate statistics across all predictions."""
    return get_stats()
