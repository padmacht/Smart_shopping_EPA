"""
Rule-based chat logic for the Shopping assistant's Chat mode.

The assistant decides what the customer wants (intent), then uses the project's own
tools: catalogue search (with optional Ollama query expansion), the recommender and the FAQ.
It never invents products: every product it mentions comes from those tools.

    intent = detect_intent("how long is delivery?")      # -> "faq"
    match = faq_answer("can I return it?", faq)          # -> FAQ row or None
"""
from __future__ import annotations

import re

import pandas as pd

INTENTS = ("greeting", "thanks", "order", "faq", "recommend", "search")

_STOP = {"a", "an", "the", "i", "my", "me", "to", "is", "it", "do", "you", "your", "can", "of", "for",
         "in", "on", "and", "or", "what", "how", "please", "with", "be", "are", "does", "this", "that"}

_PATTERNS = {
    "greeting": r"^(hi|hello|hey|hiya|good (morning|afternoon|evening))\b",
    "thanks": r"\b(thanks|thank you|cheers|thx)\b",
    "order": r"\b(where('?s| is) my order|track(ing)? (my )?order|cancel (my )?order|order (status|#?\d+))\b",
    "faq": r"\b(deliver(y|ed)?|shipping|return(s|ed)?|refund|pay(ment)?|paypal|klarna|apple pay|"
           r"discount|voucher|code|gift wrap(ping)?|warranty|contact|customer service|in stock|out of stock)\b",
    "recommend": r"\b(what should i buy|recommend|suggest(ion)?s? for me|for me\b|my (top|picks)|"
                 r"what would i like|buy next)\b",
}


def detect_intent(text: str) -> str:
    """Return one of INTENTS for a customer message (simple, explainable rules)."""
    t = text.strip().lower()
    if not t:
        return "search"
    for intent in ("order", "greeting", "thanks", "faq", "recommend"):
        if re.search(_PATTERNS[intent], t):
            # "hi, I need a yoga mat" is a search, not just a greeting
            if intent in ("greeting", "thanks") and len(t.split()) > 4:
                continue
            return intent
    return "search"


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w.rstrip("s") for w in words if w not in _STOP and len(w) > 2}


def faq_answer(question: str, faq: pd.DataFrame, min_overlap: float = 0.15) -> pd.Series | None:
    """Best-matching FAQ row by word overlap (Jaccard), or None if nothing is close enough."""
    q = _tokens(question)
    if not q or faq.empty:
        return None
    scores = faq.apply(lambda r: len(q & _tokens(f"{r['Topic']} {r['Question']}"))
                       / max(1, len(q | _tokens(f"{r['Topic']} {r['Question']}"))), axis=1)
    best = scores.idxmax()
    return faq.loc[best] if scores.loc[best] >= min_overlap else None


def extract_max_price(text: str) -> float | None:
    """'under £60', 'below 30', 'less than £25.50' -> 60.0 / 30.0 / 25.5"""
    m = re.search(r"\b(?:under|below|less than|max(?:imum)?|up to)\s*£?\s*(\d+(?:\.\d{1,2})?)", text.lower())
    return float(m.group(1)) if m else None


def strip_price(text: str) -> str:
    """Remove the price phrase so it does not pollute the keyword search."""
    return re.sub(r"\b(?:under|below|less than|max(?:imum)?|up to)\s*£?\s*\d+(?:\.\d{1,2})?", "", text,
                  flags=re.IGNORECASE).strip(" ,.")


HELP_TEXT = (
    "I can help you find products in your own words (for example *something to help me sleep under £60*), "
    "suggest what to buy next, and answer questions about delivery, returns and payment."
)
ORDER_TEXT = (
    "I can't see orders, but our customer service team can track or change it for you: "
    "live chat 8am–8pm, or support@shopsmart.example."
)
