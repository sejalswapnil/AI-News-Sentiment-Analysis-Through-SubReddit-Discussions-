"""
SillyBill AI — FastAPI Backend
Reddit Sentiment & News Intelligence Engine
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import json
from datetime import datetime
from typing import Optional

from api.reddit import RedditClient
from analytics.sentiment import SentimentEngine
from analytics.toxicity import ToxicityEngine
from analytics.emotion import EmotionEngine
from analytics.opinion import OpinionScorer
from analytics.summarizer import AISummarizer
from database.db import Database
from workers.cache import CacheLayer

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SillyBill AI",
    description="Reddit Sentiment & News Intelligence Engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ─────────────────────────────────────────────────────────────────

reddit     = RedditClient()
sentiment  = SentimentEngine()
toxicity   = ToxicityEngine()
emotion    = EmotionEngine()
opinion    = OpinionScorer()
summarizer = AISummarizer()
db         = Database()
cache      = CacheLayer()

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "online", "engine": "SillyBill AI v2.0", "time": datetime.utcnow().isoformat()}


@app.get("/analyze")
async def analyze(
    topic: str = Query(..., description="Search topic or keyword"),
    sort:  str = Query("hot", description="hot | top | new | rising"),
    limit: int = Query(25, ge=1, le=100),
):
    """
    Full analysis pipeline:
    Fetch Reddit posts → Sentiment → Toxicity → Emotion → Opinion Score
    """
    cache_key = f"analyze:{topic}:{sort}:{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    # 1. Fetch posts
    posts = await reddit.search(topic, sort=sort, limit=limit)
    if not posts:
        raise HTTPException(status_code=404, detail="No posts found for this topic.")

    # 2. Run NLP pipeline on each post
    analyzed = []
    for p in posts:
        text = f"{p['title']} {p.get('selftext', '')}"
        s = sentiment.analyze(text)
        t = toxicity.score(text)
        e = emotion.classify(text)
        o = opinion.compute(s, p.get("ups", 0), t, p.get("num_comments", 0))
        analyzed.append({
            **p,
            "sentiment":     s,
            "toxicity":      round(t, 4),
            "emotion":       e,
            "opinion_score": round(o, 4),
        })

    # 3. Aggregate stats
    result = {
        "topic":     topic,
        "sort":      sort,
        "count":     len(analyzed),
        "timestamp": datetime.utcnow().isoformat(),
        "posts":     analyzed,
        "aggregate": _aggregate(analyzed),
    }

    # 4. Persist + cache
    await db.save_search(topic, sort, result["aggregate"])
    await cache.set(cache_key, result, ttl=60)
    return result


@app.get("/trending")
async def trending(limit: int = Query(20, ge=1, le=50)):
    """Fetch trending posts from r/popular with inline sentiment."""
    cached = await cache.get("trending")
    if cached:
        return cached

    posts = await reddit.trending(limit=limit)
    result = []
    for p in posts:
        text = f"{p['title']} {p.get('selftext', '')}"
        s = sentiment.analyze(text)
        result.append({**p, "sentiment": s, "emotion": emotion.classify(text)})

    await cache.set("trending", {"posts": result, "timestamp": datetime.utcnow().isoformat()}, ttl=120)
    return {"posts": result, "timestamp": datetime.utcnow().isoformat()}


@app.get("/sentiment-history")
async def sentiment_history(topic: Optional[str] = None, limit: int = 50):
    """Return historical sentiment timeline from DB."""
    rows = await db.get_sentiment_history(topic=topic, limit=limit)
    return {"history": rows}


@app.get("/subreddit-analysis")
async def subreddit_analysis(topic: str = Query(...)):
    """Breakdown sentiment and toxicity by subreddit for a topic."""
    posts = await reddit.search(topic, sort="hot", limit=50)
    subs = {}
    for p in posts:
        sub = p["subreddit"]
        text = f"{p['title']} {p.get('selftext', '')}"
        s = sentiment.analyze(text)
        t = toxicity.score(text)
        if sub not in subs:
            subs[sub] = {"posts": 0, "upvotes": 0, "sentiment_sum": 0.0, "toxicity_sum": 0.0}
        subs[sub]["posts"]         += 1
        subs[sub]["upvotes"]       += p.get("ups", 0)
        subs[sub]["sentiment_sum"] += s["compound"]
        subs[sub]["toxicity_sum"]  += t

    result = [
        {
            "subreddit":    sub,
            "posts":        d["posts"],
            "upvotes":      d["upvotes"],
            "avg_sentiment": round(d["sentiment_sum"] / d["posts"], 4),
            "avg_toxicity":  round(d["toxicity_sum"] / d["posts"], 4),
            "verdict": "BULLISH" if d["sentiment_sum"]/d["posts"] > 0.2
                       else "BEARISH" if d["sentiment_sum"]/d["posts"] < -0.2
                       else "NEUTRAL",
        }
        for sub, d in sorted(subs.items(), key=lambda x: -x[1]["posts"])
    ]
    return {"topic": topic, "subreddits": result}


@app.get("/toxicity-report")
async def toxicity_report(topic: str = Query(...)):
    """Deep toxicity analysis for a topic."""
    posts = await reddit.search(topic, sort="hot", limit=50)
    flagged = []
    for p in posts:
        text = f"{p['title']} {p.get('selftext', '')}"
        t = toxicity.score(text)
        flags = toxicity.get_flags(text)
        if t > 0.2:
            flagged.append({
                "id":            p["id"],
                "title":         p["title"],
                "subreddit":     p["subreddit"],
                "toxicity":      round(t, 4),
                "flagged_words": flags,
                "risk":          "HIGH" if t > 0.6 else "MODERATE" if t > 0.3 else "LOW",
                "url":           f"https://reddit.com{p.get('permalink', '')}",
            })
    flagged.sort(key=lambda x: -x["toxicity"])
    return {
        "topic":         topic,
        "total_analyzed": len(posts),
        "flagged_count": len(flagged),
        "flagged_posts": flagged,
    }


@app.get("/ai-summary")
async def ai_summary(topic: str = Query(...), sort: str = Query("hot")):
    """Generate AI narrative summary via Claude."""
    posts = await reddit.search(topic, sort=sort, limit=15)
    analyzed = []
    for p in posts:
        text = f"{p['title']} {p.get('selftext', '')}"
        s = sentiment.analyze(text)
        e = emotion.classify(text)
        analyzed.append({**p, "sentiment": s, "emotion": e})

    agg = _aggregate(analyzed)
    summary = await summarizer.generate(topic, analyzed, agg)
    return {"topic": topic, "summary": summary, "aggregate": agg}


@app.get("/live-stream")
async def live_stream(topic: str = Query(...), interval: int = Query(30)):
    """
    SSE stream — pushes fresh Reddit analysis every `interval` seconds.
    Frontend connects with EventSource('/live-stream?topic=AI&interval=30')
    """
    async def event_generator():
        while True:
            posts = await reddit.search(topic, sort="hot", limit=25)
            analyzed = []
            for p in posts:
                text = f"{p['title']} {p.get('selftext', '')}"
                s = sentiment.analyze(text)
                t = toxicity.score(text)
                e = emotion.classify(text)
                o = opinion.compute(s, p.get("ups", 0), t, p.get("num_comments", 0))
                analyzed.append({**p, "sentiment": s, "toxicity": round(t,4), "emotion": e, "opinion_score": round(o,4)})

            payload = {
                "topic":     topic,
                "timestamp": datetime.utcnow().isoformat(),
                "posts":     analyzed[:10],
                "aggregate": _aggregate(analyzed),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _aggregate(posts: list) -> dict:
    if not posts:
        return {}
    n = len(posts)
    avg_compound  = sum(p["sentiment"]["compound"] for p in posts) / n
    avg_pos       = sum(p["sentiment"]["pos"]      for p in posts) / n
    avg_neg       = sum(p["sentiment"]["neg"]      for p in posts) / n
    avg_neu       = sum(p["sentiment"]["neu"]      for p in posts) / n
    avg_toxicity  = sum(p["toxicity"]              for p in posts) / n
    avg_opinion   = sum(p["opinion_score"]         for p in posts) / n
    total_ups     = sum(p.get("ups", 0)            for p in posts)
    total_comments= sum(p.get("num_comments", 0)   for p in posts)
    emotion_counts= {}
    for p in posts:
        emotion_counts[p["emotion"]] = emotion_counts.get(p["emotion"], 0) + 1
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"

    label = "POSITIVE" if avg_compound > 0.1 else "NEGATIVE" if avg_compound < -0.1 else "NEUTRAL"
    return {
        "avg_compound":    round(avg_compound, 4),
        "avg_positive":    round(avg_pos, 4),
        "avg_negative":    round(avg_neg, 4),
        "avg_neutral":     round(avg_neu, 4),
        "avg_toxicity":    round(avg_toxicity, 4),
        "avg_opinion":     round(avg_opinion, 4),
        "total_upvotes":   total_ups,
        "total_comments":  total_comments,
        "dominant_emotion":dominant_emotion,
        "emotion_counts":  emotion_counts,
        "label":           label,
        "post_count":      n,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)