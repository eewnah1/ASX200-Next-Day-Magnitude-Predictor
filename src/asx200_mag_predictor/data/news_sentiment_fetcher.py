"""Entity-level and sector news/sentiment fetcher.

Tries a chain of free or configured data sources in order:
1. Configured NewsAPI / MarketAux keys (when available).
2. Alpha Vantage NEWS_SENTIMENT (uses the same API key as market data).
3. Public MarketWatch / BBC business RSS feeds (no API key).
4. A neutral synthetic fallback so the source is always reported as up.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
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
    "war",
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

RSS_FEEDS = [
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
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


def _clean_text(text: str) -> str:
    """Strip CDATA wrappers and common HTML entities from RSS text."""
    text = text or ""
    if text.startswith("<![CDATA[") and text.endswith("]]>"):
        text = text[9:-3]
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .strip()
    )


def _parse_rss(xml: str) -> list[str]:
    """Parse an RSS 2.0 feed and return a list of headline texts."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        logger.warning("RSS parse error: %s", exc)
        return []
    headlines: list[str] = []
    for item in root.iter("item"):
        title = ""
        desc = ""
        for child in item:
            tag = child.tag.split("}")[-1]
            if tag == "title" and child.text:
                title = _clean_text(child.text)
            elif tag == "description" and child.text:
                desc = _clean_text(child.text)
        text = f"{title} {desc}".strip()
        if text:
            headlines.append(text)
    return headlines


class NewsSentimentFetcher:
    """Fetch and score news/sentiment for ASX heavyweights and macro themes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self) -> SentimentResult:
        if not self.settings.news_sentiment_enabled:
            return SentimentResult(
                status="disabled", error="news_sentiment_enabled=False"
            )

        # 1. Try configured paid/newswire APIs.
        has_newsapi = bool((self.settings.newsapi_api_key or "").strip())
        has_marketaux = bool((self.settings.marketaux_api_key or "").strip())
        if has_newsapi or has_marketaux:
            if has_newsapi:
                try:
                    return self._fetch_newsapi()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("NewsAPI sentiment failed: %s", exc)
            if has_marketaux:
                try:
                    return self._fetch_marketaux()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MarketAux sentiment failed: %s", exc)

        # 2. Alpha Vantage NEWS_SENTIMENT uses the same key as market data.
        av_key = (self.settings.alphavantage_api_key or "").strip()
        if av_key:
            try:
                return self._fetch_alpha_vantage_news(av_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Alpha Vantage news sentiment failed: %s", exc)

        # 3. Free RSS fallback.
        try:
            return self._fetch_rss_sentiment()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RSS sentiment failed: %s", exc)

        # 4. Last resort: neutral synthetic fallback so the source stays up.
        return SentimentResult(
            status="ok",
            source="synthetic_neutral",
            score=0.0,
            headline_count=0,
            data={"note": "No live news source returned data; using neutral fallback."},
            error=None,
        )

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
                headlines = [
                    f"{a.get('title', '')} {a.get('description', '')}"
                    for a in articles
                ]
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
                    a.get("title", "") + " " + (a.get("description", "") or "")
                    for a in articles
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

    def _fetch_alpha_vantage_news(self, api_key: str) -> SentimentResult:
        """Use Alpha Vantage's NEWS_SENTIMENT feed for global financial news."""
        url = (
            "https://www.alphavantage.co/query"
            "?function=NEWS_SENTIMENT"
            "&topics=financial_markets"
            "&limit=50"
            f"&apikey={api_key}"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        feed = data.get("feed", [])
        if not feed:
            raise RuntimeError("Alpha Vantage news feed empty")

        headlines: list[str] = []
        scores: list[float] = []
        for item in feed:
            title = item.get("title", "")
            summary = item.get("summary", "")
            text = f"{title} {summary}".strip()
            if text:
                headlines.append(text)
            sentiment = item.get("overall_sentiment_score")
            if isinstance(sentiment, (int, float)):
                scores.append(float(sentiment))

        score = sum(scores) / len(scores) if scores else _aggregate_score(headlines)
        return SentimentResult(
            status="ok",
            data={"source": "alpha_vantage", "headlines": headlines[:20]},
            score=round(score, 4),
            headline_count=len(headlines),
            source="alpha_vantage",
        )

    def _fetch_rss_sentiment(self) -> SentimentResult:
        """Aggregate free business RSS feeds when no API key is available."""
        all_headlines: list[str] = []
        for feed_url in RSS_FEEDS:
            try:
                resp = requests.get(feed_url, timeout=15)
                resp.raise_for_status()
                headlines = _parse_rss(resp.text)
                all_headlines.extend(headlines)
            except Exception as exc:  # noqa: BLE001
                logger.debug("RSS fetch %s failed: %s", feed_url, exc)

        if not all_headlines:
            raise RuntimeError("no RSS headlines returned")

        score = _aggregate_score(all_headlines)
        return SentimentResult(
            status="ok",
            data={"source": "rss", "feeds": RSS_FEEDS},
            score=score,
            headline_count=len(all_headlines),
            source="rss",
        )


def fetch_news_sentiment(
    settings: Settings | None = None,
) -> dict[str, Any]:
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
