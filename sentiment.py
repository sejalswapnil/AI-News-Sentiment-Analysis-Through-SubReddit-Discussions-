"""
analytics/sentiment.py
Sentiment Analysis Engine
VADER lexicon + rule-based ensemble scoring
"""

import re
import math


# ── Lexicons ───────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "good": 1.5, "great": 2.0, "excellent": 2.5, "amazing": 2.5, "awesome": 2.0,
    "love": 2.0, "best": 2.0, "fantastic": 2.5, "wonderful": 2.0, "happy": 1.5,
    "positive": 1.5, "brilliant": 2.0, "superb": 2.5, "perfect": 2.5, "outstanding": 2.5,
    "incredible": 2.0, "beautiful": 1.5, "joy": 2.0, "excited": 1.5, "glad": 1.0,
    "pleased": 1.0, "delighted": 2.0, "impressed": 1.5, "helpful": 1.5, "friendly": 1.0,
    "useful": 1.0, "nice": 1.0, "pleasant": 1.0, "enjoy": 1.5, "fun": 1.5,
    "win": 1.5, "success": 2.0, "benefit": 1.5, "improve": 1.5, "growth": 1.5,
    "gain": 1.0, "progress": 1.5, "hope": 1.0, "safe": 1.0, "easy": 1.0,
    "powerful": 1.5, "innovative": 2.0, "revolutionary": 2.0, "breakthrough": 2.0,
    "efficient": 1.5, "reliable": 1.5, "affordable": 1.0, "free": 0.5, "fast": 1.0,
}

NEGATIVE_WORDS = {
    "bad": -1.5, "terrible": -2.5, "awful": -2.5, "hate": -2.0, "worst": -2.5,
    "horrible": -2.5, "disgusting": -2.0, "ugly": -1.5, "angry": -1.5, "sad": -1.5,
    "negative": -1.5, "poor": -1.5, "failure": -2.0, "problem": -1.0, "issue": -0.5,
    "wrong": -1.5, "fail": -2.0, "broken": -1.5, "useless": -2.0, "boring": -1.0,
    "annoying": -1.5, "stupid": -2.0, "dumb": -1.5, "waste": -1.5, "corrupt": -2.5,
    "dangerous": -2.0, "fear": -1.5, "scary": -1.5, "threat": -2.0, "crisis": -2.0,
    "disaster": -2.5, "fraud": -2.5, "lie": -2.0, "fake": -1.5, "scam": -2.5,
    "toxic": -2.0, "abuse": -2.5, "violence": -3.0, "crime": -2.5, "evil": -2.5,
    "harm": -2.0, "hurt": -2.0, "damage": -2.0, "loss": -1.5, "decline": -1.5,
    "crash": -2.5, "collapse": -2.5, "panic": -2.0, "chaos": -2.0, "war": -3.0,
    "death": -3.0, "kill": -3.0, "destroy": -2.5, "attack": -2.5, "terror": -3.0,
    "racist": -3.0, "sexist": -2.5, "bigot": -2.5, "wrong": -1.5, "expensive": -1.0,
    "slow": -0.5, "complicated": -0.5, "confusing": -0.5, "disappointing": -2.0,
    "overrated": -1.5, "useless": -2.0, "misleading": -2.0,
}

NEGATORS = {"not", "no", "never", "cannot", "can't", "won't", "don't", "doesn't", "isn't", "aren't", "wasn't"}
INTENSIFIERS = {"very": 1.3, "extremely": 1.6, "incredibly": 1.5, "absolutely": 1.4, "totally": 1.2, "really": 1.2, "so": 1.1}


class SentimentEngine:

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of text.
        Returns: {pos, neg, neu, compound, label}
        """
        if not text or not text.strip():
            return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0, "label": "NEUTRAL"}

        text_clean = self._clean(text)
        tokens = text_clean.lower().split()

        pos_score = 0.0
        neg_score = 0.0
        total_tokens = len(tokens)

        i = 0
        while i < len(tokens):
            word = tokens[i]
            multiplier = 1.0

            # Check for negator in previous 3 tokens
            negated = any(tokens[max(0, i-j)] in NEGATORS for j in range(1, 4))

            # Check for intensifier before this word
            if i > 0 and tokens[i-1] in INTENSIFIERS:
                multiplier = INTENSIFIERS[tokens[i-1]]

            if word in POSITIVE_WORDS:
                val = POSITIVE_WORDS[word] * multiplier
                if negated:
                    neg_score += abs(val) * 0.5
                else:
                    pos_score += val
            elif word in NEGATIVE_WORDS:
                val = abs(NEGATIVE_WORDS[word]) * multiplier
                if negated:
                    pos_score += val * 0.5
                else:
                    neg_score += val
            i += 1

        # Punctuation boosts
        exclamations = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if exclamations > 0:
            if pos_score > neg_score:
                pos_score *= (1 + min(exclamations * 0.1, 0.3))
            else:
                neg_score *= (1 + min(exclamations * 0.1, 0.3))
        if caps_ratio > 0.5 and len(text) > 10:
            neg_score *= 1.2  # ALL CAPS tends to be more negative/intense

        # Normalize
        norm = pos_score + neg_score + 1e-8
        pos_norm = min(pos_score / (norm + math.sqrt(norm)), 1.0)
        neg_norm = min(neg_score / (norm + math.sqrt(norm)), 1.0)
        neu_norm = max(0.0, 1.0 - pos_norm - neg_norm)
        compound = (pos_score - neg_score) / (norm + math.sqrt(norm)) * 2
        compound = max(-1.0, min(1.0, compound))

        label = "POSITIVE" if compound > 0.1 else "NEGATIVE" if compound < -0.1 else "NEUTRAL"

        return {
            "pos":      round(pos_norm, 4),
            "neg":      round(neg_norm, 4),
            "neu":      round(neu_norm, 4),
            "compound": round(compound, 4),
            "label":    label,
        }

    def batch_analyze(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]

    def _clean(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s!?.,']", " ", text)
        return text.strip()