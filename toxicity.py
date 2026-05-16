"""
analytics/toxicity.py
Toxicity & Hate Speech Detection Engine
Rule-based lexicon scoring with flag extraction
"""

import re


# ── Lexicons by category ──────────────────────────────────────────────────────

HATE_SPEECH = {
    "racist", "sexist", "nazi", "bigot", "slur", "supremacist",
    "extremist", "terrorist", "fascist",
}

HARASSMENT = {
    "stupid", "idiot", "moron", "loser", "pathetic", "worthless",
    "useless", "trash", "garbage", "disgusting",
}

THREATS = {
    "kill", "murder", "attack", "destroy", "hurt", "harm", "die",
    "burn", "shoot", "bomb", "explode", "threaten",
}

OFFENSIVE = {
    "hate", "awful", "terrible", "horrible", "disgusting",
    "vile", "repulsive", "revolting",
}

EXPLICIT = {
    "fuck", "shit", "ass", "bitch", "bastard", "damn", "crap",
    "hell", "piss", "cock", "dick", "pussy",
}

# Weights per category
CATEGORY_WEIGHTS = {
    "hate_speech":  0.35,
    "harassment":   0.20,
    "threats":      0.40,
    "offensive":    0.10,
    "explicit":     0.15,
}


class ToxicityEngine:

    def score(self, text: str) -> float:
        """
        Returns a toxicity score 0.0 → 1.0
        """
        if not text or not text.strip():
            return 0.0

        t = text.lower()
        tokens = set(re.findall(r"\b\w+\b", t))

        raw_score = 0.0
        raw_score += len(tokens & HATE_SPEECH)  * CATEGORY_WEIGHTS["hate_speech"]
        raw_score += len(tokens & HARASSMENT)   * CATEGORY_WEIGHTS["harassment"]
        raw_score += len(tokens & THREATS)      * CATEGORY_WEIGHTS["threats"]
        raw_score += len(tokens & OFFENSIVE)    * CATEGORY_WEIGHTS["offensive"]
        raw_score += len(tokens & EXPLICIT)     * CATEGORY_WEIGHTS["explicit"]

        # Punctuation signals
        if text.count("!!") >= 2:
            raw_score += 0.15
        if text.count("?!") >= 1:
            raw_score += 0.1
        # All caps
        words = text.split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        if caps_words >= 3:
            raw_score += 0.2

        return round(min(1.0, raw_score), 4)

    def get_flags(self, text: str) -> list[str]:
        """Return list of flagged words found in text."""
        t = text.lower()
        tokens = set(re.findall(r"\b\w+\b", t))
        all_toxic = HATE_SPEECH | HARASSMENT | THREATS | OFFENSIVE | EXPLICIT
        flagged = list(tokens & all_toxic)
        return sorted(flagged)

    def classify_risk(self, score: float) -> str:
        if score >= 0.6:  return "HIGH"
        if score >= 0.3:  return "MODERATE"
        return "LOW"

    def breakdown(self, text: str) -> dict:
        """Full breakdown by category."""
        t = text.lower()
        tokens = set(re.findall(r"\b\w+\b", t))
        return {
            "hate_speech_words": list(tokens & HATE_SPEECH),
            "harassment_words":  list(tokens & HARASSMENT),
            "threat_words":      list(tokens & THREATS),
            "offensive_words":   list(tokens & OFFENSIVE),
            "explicit_words":    list(tokens & EXPLICIT),
            "total_score":       self.score(text),
            "risk":              self.classify_risk(self.score(text)),
        }