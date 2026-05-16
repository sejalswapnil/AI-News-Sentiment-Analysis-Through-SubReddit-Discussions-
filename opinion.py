"""
analytics/opinion.py
Public Opinion Score Engine
Weighted formula combining sentiment, engagement, toxicity
"""

import math


class OpinionScorer:
    """
    Final Opinion Score =
        (0.5 × Sentiment Polarity)
      + (0.3 × Engagement Score)
      - (0.2 × Toxicity Penalty)

    All inputs normalized to [0, 1].
    Output range: [0, 1] where:
        > 0.65  →  POSITIVE opinion
        0.4–0.65 → NEUTRAL
        < 0.40  →  NEGATIVE opinion
    """

    def compute(
        self,
        sentiment: dict,
        upvotes: int,
        toxicity: float,
        num_comments: int,
        max_engagement: int = 10_000,
    ) -> float:
        # Sentiment component: map compound [-1,1] → [0,1]
        sentiment_component = (sentiment.get("compound", 0.0) + 1.0) / 2.0

        # Engagement component: log-normalized
        raw_eng = upvotes + num_comments
        if raw_eng <= 0:
            engagement_component = 0.0
        else:
            engagement_component = min(1.0, math.log1p(raw_eng) / math.log1p(max_engagement))

        # Toxicity penalty: already [0,1]
        toxicity_component = min(1.0, max(0.0, toxicity))

        score = (
            0.5 * sentiment_component
          + 0.3 * engagement_component
          - 0.2 * toxicity_component
        )

        return round(max(0.0, min(1.0, score)), 4)

    def label(self, score: float) -> str:
        if score >= 0.65: return "POSITIVE"
        if score >= 0.40: return "NEUTRAL"
        return "NEGATIVE"

    def interpret(self, score: float) -> str:
        pct = round(score * 100)
        label = self.label(score)
        if label == "POSITIVE":
            return f"Public opinion is {pct}% positive with strong engagement signals."
        elif label == "NEGATIVE":
            return f"Public opinion is {100 - pct}% negative — significant toxicity or low sentiment detected."
        else:
            return f"Public opinion is mixed at {pct}% — no dominant positive or negative signal."