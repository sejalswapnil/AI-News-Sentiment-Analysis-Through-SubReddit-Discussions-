"""
analytics/emotion.py
Emotion Classification Engine
Classifies text into 7 Ekman basic emotions
"""

import re
from collections import defaultdict


EMOTION_LEXICONS = {
    "fear": {
        "fear", "scared", "worried", "anxious", "terrified", "panic",
        "dread", "afraid", "nervous", "stress", "threat", "danger",
        "risk", "uncertain", "unstable", "alarming", "frightening",
        "concerning", "ominous", "warning", "suspicious",
    },
    "anger": {
        "angry", "furious", "outraged", "mad", "rage", "infuriated",
        "livid", "frustrated", "pissed", "hate", "despise", "loathe",
        "disgusted", "betrayed", "lied", "cheated", "unfair", "corrupt",
        "injustice", "wrong", "unacceptable", "ridiculous",
    },
    "joy": {
        "happy", "joy", "excited", "thrilled", "elated", "wonderful",
        "amazing", "love", "great", "fantastic", "excellent", "brilliant",
        "awesome", "delighted", "pleased", "glad", "overjoyed", "celebrate",
        "congratulations", "win", "success", "achievement",
    },
    "sadness": {
        "sad", "depressed", "miserable", "heartbroken", "sorrow", "grief",
        "unhappy", "upset", "cry", "tears", "mourning", "loss", "lonely",
        "hopeless", "helpless", "devastated", "tragic", "terrible",
        "unfortunate", "regret", "miss",
    },
    "surprise": {
        "surprise", "shocked", "unexpected", "unbelievable", "wow",
        "omg", "incredible", "astonish", "stunning", "remarkable",
        "extraordinary", "bizarre", "strange", "weird", "odd",
        "suddenly", "suddenly", "overnight", "breaking",
    },
    "disgust": {
        "disgusting", "horrible", "awful", "repulsive", "gross", "nasty",
        "vile", "revolting", "offensive", "appalling", "abhorrent",
        "shameful", "pathetic", "pitiful", "despicable", "heinous",
    },
    "anticipation": {
        "hope", "expect", "waiting", "looking", "forward", "soon",
        "upcoming", "planning", "predict", "forecast", "anticipate",
        "future", "potential", "possible", "maybe", "could", "might",
    },
}

# Priority order — earlier = higher priority when tied
PRIORITY = ["fear", "anger", "disgust", "sadness", "joy", "surprise", "anticipation"]


class EmotionEngine:

    def classify(self, text: str) -> str:
        """Return dominant emotion label."""
        scores = self.score_all(text)
        if not scores or max(scores.values()) == 0:
            return "neutral"
        return max(scores, key=lambda k: (scores[k], -PRIORITY.index(k) if k in PRIORITY else -99))

    def score_all(self, text: str) -> dict:
        """Return scores for all emotions."""
        if not text:
            return {e: 0 for e in EMOTION_LEXICONS}
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        return {
            emotion: len(tokens & words)
            for emotion, words in EMOTION_LEXICONS.items()
        }

    def distribution(self, texts: list[str]) -> dict:
        """Emotion distribution across a list of texts."""
        counts = defaultdict(int)
        for text in texts:
            counts[self.classify(text)] += 1
        total = len(texts) or 1
        return {
            emotion: {"count": counts[emotion], "pct": round(counts[emotion] / total * 100, 1)}
            for emotion in list(EMOTION_LEXICONS.keys()) + ["neutral"]
        }