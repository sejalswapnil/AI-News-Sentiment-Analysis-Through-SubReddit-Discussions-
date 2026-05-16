"""
analytics/summarizer.py
AI Narrative Summarizer — Claude API integration
Generates intelligence briefings from Reddit post data
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are SillyBill AI, an elite social media intelligence analyst.
Your job is to analyze Reddit discussions and deliver sharp, data-driven public opinion briefings.
Be specific, reference actual numbers, highlight key narratives and emerging patterns.
Keep responses to 3-5 sentences. Use professional intelligence analyst tone.
Never use markdown formatting — plain text only."""


class AISummarizer:

    async def generate(self, topic: str, posts: list[dict], aggregate: dict) -> str:
        """Generate an AI intelligence briefing for a topic."""
        if not ANTHROPIC_API_KEY:
            return "AI analysis unavailable — set ANTHROPIC_API_KEY in .env"

        top_posts = posts[:10]
        post_lines = "\n".join([
            f'- "{p["title"][:100]}" '
            f'(r/{p["subreddit"]}, {p.get("ups", 0)} upvotes, '
            f'sentiment: {p["sentiment"]["compound"]:.2f}, '
            f'emotion: {p.get("emotion", "neutral")})'
            for p in top_posts
        ])

        user_prompt = f"""Topic: "{topic}"
Overall Sentiment: {aggregate.get("avg_compound", 0):.3f} ({aggregate.get("label", "NEUTRAL")})
Avg Toxicity: {aggregate.get("avg_toxicity", 0) * 100:.1f}%
Dominant Emotion: {aggregate.get("dominant_emotion", "neutral")}
Public Opinion Score: {aggregate.get("avg_opinion", 0) * 100:.0f}%
Total Upvotes: {aggregate.get("total_upvotes", 0):,}
Total Comments: {aggregate.get("total_comments", 0):,}
Post Count: {aggregate.get("post_count", 0)}

Top Posts:
{post_lines}

Provide a sharp intelligence briefing on current Reddit public opinion around this topic."""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key":         ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      CLAUDE_MODEL,
                        "max_tokens": 500,
                        "system":     SYSTEM_PROMPT,
                        "messages":   [{"role": "user", "content": user_prompt}],
                    },
                )
                data = response.json()
                return data["content"][0]["text"]
        except Exception as e:
            return f"AI analysis error: {str(e)}"

    async def generate_narrative(self, posts: list[dict]) -> dict:
        """Extract key arguments for/against from posts."""
        if not ANTHROPIC_API_KEY:
            return {"for": [], "against": [], "summary": ""}

        titles = "\n".join([f'- {p["title"]}' for p in posts[:15]])
        prompt = f"""From these Reddit post titles, extract:
1. Top 3 arguments FOR the topic (positive perspectives)
2. Top 3 arguments AGAINST the topic (negative perspectives)
3. One sentence summary of the overall narrative

Posts:
{titles}

Respond in JSON: {{"for": ["...", "...", "..."], "against": ["...", "...", "..."], "summary": "..."}}"""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key":         ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      CLAUDE_MODEL,
                        "max_tokens": 400,
                        "messages":   [{"role": "user", "content": prompt}],
                    },
                )
                import json
                text = response.json()["content"][0]["text"]
                text = text.strip().lstrip("```json").rstrip("```").strip()
                return json.loads(text)
        except Exception as e:
            return {"for": [], "against": [], "summary": str(e)}