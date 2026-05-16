"""
api/reddit.py
Reddit API client — PRAW + async wrapper
Handles OAuth2, post fetching, comments, trending
"""

import asyncio
import praw
from typing import Optional
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id     = os.getenv("REDDIT_CLIENT_ID",     "_l5awOBbKvD57-w5kiASRA"),
            client_secret = os.getenv("REDDIT_CLIENT_SECRET", "vpXa_4k3drR5rHE5wbEaIVyogEKUoA"),
            user_agent    = os.getenv("REDDIT_USER_AGENT",    "SillyBillAI/2.0 by CarefulParsley6496"),
        )

    # ── Public Methods ──────────────────────────────────────────────────────────

    async def search(self, query: str, sort: str = "hot", limit: int = 25) -> list[dict]:
        """Search Reddit for posts matching query."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._search_sync, query, sort, limit
        )

    async def trending(self, limit: int = 20) -> list[dict]:
        """Fetch hot posts from r/popular."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._trending_sync, limit
        )

    async def get_comments(self, post_id: str, limit: int = 50) -> list[dict]:
        """Fetch top comments for a post."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._comments_sync, post_id, limit
        )

    async def subreddit_posts(self, subreddit: str, sort: str = "hot", limit: int = 25) -> list[dict]:
        """Fetch posts from a specific subreddit."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._subreddit_sync, subreddit, sort, limit
        )

    # ── Sync Internals ──────────────────────────────────────────────────────────

    def _search_sync(self, query: str, sort: str, limit: int) -> list[dict]:
        try:
            results = self.reddit.subreddit("all").search(
                query, sort=sort, limit=limit, time_filter="week"
            )
            return [self._serialize(p) for p in results]
        except Exception as e:
            print(f"[Reddit] search error: {e}")
            return []

    def _trending_sync(self, limit: int) -> list[dict]:
        try:
            posts = self.reddit.subreddit("popular").hot(limit=limit)
            return [self._serialize(p) for p in posts]
        except Exception as e:
            print(f"[Reddit] trending error: {e}")
            return []

    def _comments_sync(self, post_id: str, limit: int) -> list[dict]:
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            comments = []
            for c in submission.comments.list()[:limit]:
                comments.append({
                    "id":        c.id,
                    "body":      c.body,
                    "score":     c.score,
                    "author":    str(c.author) if c.author else "[deleted]",
                    "created":   datetime.utcfromtimestamp(c.created_utc).isoformat(),
                })
            return comments
        except Exception as e:
            print(f"[Reddit] comments error: {e}")
            return []

    def _subreddit_sync(self, subreddit: str, sort: str, limit: int) -> list[dict]:
        try:
            sub = self.reddit.subreddit(subreddit)
            if sort == "hot":      posts = sub.hot(limit=limit)
            elif sort == "top":    posts = sub.top(limit=limit)
            elif sort == "new":    posts = sub.new(limit=limit)
            elif sort == "rising": posts = sub.rising(limit=limit)
            else:                  posts = sub.hot(limit=limit)
            return [self._serialize(p) for p in posts]
        except Exception as e:
            print(f"[Reddit] subreddit error: {e}")
            return []

    def _serialize(self, post) -> dict:
        return {
            "id":           post.id,
            "title":        post.title,
            "selftext":     post.selftext[:1000] if post.selftext else "",
            "url":          post.url,
            "permalink":    post.permalink,
            "subreddit":    post.subreddit.display_name,
            "author":       str(post.author) if post.author else "[deleted]",
            "ups":          post.ups,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "score":        post.score,
            "created_utc":  datetime.utcfromtimestamp(post.created_utc).isoformat(),
            "is_video":     post.is_video,
            "over_18":      post.over_18,
            "flair":        post.link_flair_text or "",
        }