"""Entity-level and sector news/sentiment fetcher.

Queries NewsAPI and MarketAux for headlines about the ASX200 heavyweights and
the key China / iron-ore / bank narratives, then computes a lightweight
sentiment score from keyword counts.  Falls back cleanly when no API key is
configured or the service is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)

POSITIVE_WORDS = {
    "upgrade",
    "bullish",
    "outperform",
    "beat",
    "strong",
    "growth",
    "recovery",
    "rally",
    "surge",
    "gain",
    "rise",
    "optimistic",
    "positive",
    "opportunity",
    "momentum",
}

NEGATIVE_WORDS = {
    "downgrade",
    "bearish",
    "underperform",
    "miss",
    "weak",
    "slowdown",
    "recession",
    "crash",
    "plunge",
    "drop",
    "fall",
    "negative",
    "concern",
    "risk",
    "warning",
    "cut",
    "tariff",
    "inflation",
    "default",
    "crisis",
}

ENTITY_QUERIES = [
    ("CBA", '"Commonwealth Bank" OR CBA.AX OR CBA'),
    ("BHP", '"BHP Group" OR BHP.AX OR BHP'),
    ("RIO", '"Rio Tinto" OR RIO.AX OR RIO'),
    ("FMG", '"Fortescue" OR FMG.AX OR FMG'),
    ("WDS", '"Woodside" OR WDS.AX OR WDS'),
    ("banks", 'Australian banks OR ASX banks OR "financial sector"'),
    ("iron_ore", '"iron ore" OR "iron ore price" OR "steel"'),
    ("china", 'China economy OR China stimulus OR China property OR "PBOC"'),
]


@dataclass
class SentimentResult:
    name: str = "news_sentiment"
    status: str = "ok"
    data: dict[str, Any] | None = None
    error: str | None = None
    score: float | None = None
    components: dict[str, float] | None = None
    headline_count: int = 0
    source: str = ""


def _score_text(text: str) -> tuple[int, int, int]:
    """Return (positive, negative, total) keyword counts for a headline/body."""
    lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in lower)
    return pos, neg, pos + neg


def _aggregate_score(headlines: list[str]) -> float:
    """Map a list of headlines to a sentiment score in [-1, 1]."""
    if not headlines:
        return 0.0
    pos_total = neg_total = 0
    for h in headlines:
        p, n, _ = _score_text(h)
        pos_total += p
        neg_total += n
    total = pos_total + neg_total
    if total == 0:
        return 0.0
    return round((pos_total - neg_total) / total, 4)


class NewsSentimentFetcher:
    """Fetch and score news/sentiment for ASX heavyweights and macro themes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self) -> SentimentResult:
        if not self.settings.news_sentiment_enabled:
            return SentimentResult(status="disabled", error="news_sentiment_enabled=False")

        has_newsapi = bool((self.settings.newsapi_api_key or "").strip())
        has_marketaux = bool((self.settings.marketaux_api_key or "").strip())
        if not has_newsapi and not has_marketaux:
            return SentimentResult(
                status="disabled",
                error="NEWSAPI_API_KEY / MARKETAUX_API_KEY not set (optional enrichment)",
            )

        try:
            return self._fetch_newsapi()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsAPI sentiment failed: %s", exc)
        try:
            return self._fetch_marketaux()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MarketAux sentiment failed: %s", exc)
        return SentimentResult(status="failed", error="all news/sentiment sources failed")

    def _fetch_newsapi(self) -> SentimentResult:
        key = self.settings.newsapi_api_key
        if not key:
            raise RuntimeError("NEWSAPI_API_KEY not configured")

        since = (datetime.utcnow() - timedelta(hours=48)).strftime("%Y-%m-%d")
        until = datetime.utcnow().strftime("%Y-%m-%d")

        components: dict[str, float] = {}
        all_headlines: list[str] = []

        for name, query in ENTITY_QUERIES:
            try:
                url = (
                    "https://newsapi.org/v2/everything"
                    f"?q={requests.utils.quote(query)}"
                    f"&from={since}&to={until}"
                    "&language=en&sortBy=relevancy"
                    f"&pageSize={self.settings.news_headlines_per_entity}"
                    f"&apiKey={key}"
                )
                resp = requests.get(url, timeout=12)
                resp.raise_for_status()
                data = resp.json()
                articles = data.get("articles", [])
                headlines = [f"{a.get('title', '')} {a.get('description', '')}" for a in articles]
                if headlines:
                    components[name] = _aggregate_score(headlines)
                    all_headlines.extend(headlines)
            except Exception as exc:  # noqa: BLE001
                logger.debug("NewsAPI query %s failed: %s", name, exc)

        if not all_headlines:
            raise RuntimeError("no NewsAPI headlines returned")

        score = _aggregate_score(all_headlines)
        return SentimentResult(
            status="ok",
            data={"source": "newsapi", "components": components},
            score=score,
            components=components,
            headline_count=len(all_headlines),
            source="newsapi",
        )

    def _fetch_marketaux(self) -> SentimentResult:
        key = self.settings.marketaux_api_key
        if not key:
            raise RuntimeError("MARKETAUX_API_KEY not configured")

        components: dict[str, float] = {}
        all_headlines: list[str] = []

        for name, query in ENTITY_QUERIES:
            try:
                url = (
                    "https://api.marketaux.com/v1/news/all"
                    f"?api_token={key}&language=en"
                    f"&limit={self.settings.news_headlines_per_entity}"
                    f"&search={requests.utils.quote(query)}"
                )
                resp = requests.get(url, timeout=12)
                resp.raise_for_status()
                data = resp.json()
                articles = data.get("data", [])
                headlines = [
                    a.get("title", "") + " " + (a.get("description", "") or "") for a in articles
                ]
                if headlines:
                    components[name] = _aggregate_score(headlines)
                    all_headlines.extend(headlines)
            except Exception as exc:  # noqa: BLE001
                logger.debug("MarketAux query %s failed: %s", name, exc)

        if not all_headlines:
            raise RuntimeError("no MarketAux headlines returned")

        score = _aggregate_score(all_headlines)
        return SentimentResult(
            status="ok",
            data={"source": "marketaux", "components": components},
            score=score,
            components=components,
            headline_count=len(all_headlines),
            source="marketaux",
        )


def fetch_news_sentiment(settings: Settings | None = None) -> dict[str, Any]:
    """Convenience wrapper returning a plain dict for the DataFetcher pipeline."""
    result = NewsSentimentFetcher(settings).fetch()
    return {
        "name": result.name,
        "status": result.status,
        "data": result.data or {},
        "error": result.error,
        "score": result.score,
        "components": result.components,
        "headline_count": result.headline_count,
        "source": result.source,
    }
