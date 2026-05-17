"""
api/reddit.py
Reddit API client — PRAW + async wrapper
Handles OAuth2, post fetching, comments, trending
"""

import asyncio
import os

from datetime import datetime
from typing import Optional

import praw
from dotenv import load_dotenv

# ── Environment Variables ─────────────────────────────────────────────────────

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "SillyBillAI/2.0"
)

# ── Reddit Client ─────────────────────────────────────────────────────────────

class RedditClient:

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):

        self.client_id = client_id or REDDIT_CLIENT_ID
        self.client_secret = client_secret or REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or REDDIT_USER_AGENT

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Missing Reddit API credentials. "
                "Please set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET in your .env file."
            )

        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            check_for_async=False,
        )

    # ── Public Async Methods ──────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        sort: str = "hot",
        limit: int = 25,
    ) -> list[dict]:
        """
        Search Reddit for posts matching a query.
        """

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._search_sync,
            query,
            sort,
            limit,
        )

    async def trending(
        self,
        limit: int = 20,
    ) -> list[dict]:
        """
        Fetch trending posts from r/popular.
        """

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._trending_sync,
            limit,
        )

    async def get_comments(
        self,
        post_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch top comments for a post.
        """

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._comments_sync,
            post_id,
            limit,
        )

    async def subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
    ) -> list[dict]:
        """
        Fetch posts from a specific subreddit.
        """

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._subreddit_sync,
            subreddit,
            sort,
            limit,
        )

    # ── Sync Internal Methods ────────────────────────────────────────────────

    def _search_sync(
        self,
        query: str,
        sort: str,
        limit: int,
    ) -> list[dict]:

        try:

            results = self.reddit.subreddit("all").search(
                query=query,
                sort=sort,
                limit=limit,
                time_filter="week",
            )

            return [
                self._serialize(post)
                for post in results
            ]

        except Exception as e:

            print(f"[Reddit] Search Error: {e}")

            return []

    def _trending_sync(
        self,
        limit: int,
    ) -> list[dict]:

        try:

            posts = self.reddit.subreddit("popular").hot(
                limit=limit
            )

            return [
                self._serialize(post)
                for post in posts
            ]

        except Exception as e:

            print(f"[Reddit] Trending Error: {e}")

            return []

    def _comments_sync(
        self,
        post_id: str,
        limit: int,
    ) -> list[dict]:

        try:

            submission = self.reddit.submission(
                id=post_id
            )

            submission.comments.replace_more(
                limit=0
            )

            comments = []

            for comment in submission.comments.list()[:limit]:

                comments.append({
                    "id": comment.id,
                    "body": comment.body,
                    "score": comment.score,
                    "author": (
                        str(comment.author)
                        if comment.author
                        else "[deleted]"
                    ),
                    "created": datetime.utcfromtimestamp(
                        comment.created_utc
                    ).isoformat(),
                })

            return comments

        except Exception as e:

            print(f"[Reddit] Comments Error: {e}")

            return []

    def _subreddit_sync(
        self,
        subreddit: str,
        sort: str,
        limit: int,
    ) -> list[dict]:

        try:

            sub = self.reddit.subreddit(
                subreddit
            )

            if sort == "hot":
                posts = sub.hot(limit=limit)

            elif sort == "top":
                posts = sub.top(limit=limit)

            elif sort == "new":
                posts = sub.new(limit=limit)

            elif sort == "rising":
                posts = sub.rising(limit=limit)

            else:
                posts = sub.hot(limit=limit)

            return [
                self._serialize(post)
                for post in posts
            ]

        except Exception as e:

            print(f"[Reddit] Subreddit Error: {e}")

            return []

    # ── Serialization ─────────────────────────────────────────────────────────

    def _serialize(
        self,
        post,
    ) -> dict:

        return {
            "id": post.id,

            "title": post.title,

            "selftext": (
                post.selftext[:1000]
                if post.selftext
                else ""
            ),

            "url": post.url,

            "permalink": post.permalink,

            "subreddit": (
                post.subreddit.display_name
            ),

            "author": (
                str(post.author)
                if post.author
                else "[deleted]"
            ),

            "ups": post.ups,

            "upvote_ratio": (
                post.upvote_ratio
            ),

            "num_comments": (
                post.num_comments
            ),

            "score": post.score,

            "created_utc": (
                datetime.utcfromtimestamp(
                    post.created_utc
                ).isoformat()
            ),

            "is_video": post.is_video,

            "over_18": post.over_18,

            "flair": (
                post.link_flair_text or ""
            ),
        }