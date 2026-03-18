# ============================================================
#  rss_scraper.py — RSS Feed Scraper (Geo, ARY, Dawn, BBC Urdu)
#  Author : Salar Ahmed | FYP 2025-2026
# ============================================================

import feedparser, requests, datetime
from typing import List, Dict
from bs4 import BeautifulSoup

RSS_FEEDS = {
    "Geo News"  : "https://www.geo.tv/rss/1",
    "ARY News"  : "https://arynews.tv/feed/",
    "Dawn"      : "https://www.dawn.com/feeds/home",
    "BBC Urdu"  : "https://feeds.bbci.co.uk/urdu/rss.xml",
    "The News"  : "https://www.thenews.com.pk/rss/1/1",
}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 10   # seconds


def _strip_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text(separator=" ").strip()


def _parse_feed(feed_name: str, feed_url: str, keyword: str) -> List[Dict]:
    articles = []
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        for entry in feed.entries:
            title   = _strip_html(entry.get("title",   ""))
            summary = _strip_html(entry.get("summary", ""))
            link    = entry.get("link", "")

            # Keyword filter (case-insensitive, supports Urdu)
            combined = (title + " " + summary).lower()
            if keyword.lower() not in combined:
                continue

            published = entry.get("published", datetime.datetime.utcnow().isoformat())

            articles.append({
                "title"    : title,
                "summary"  : summary[:500],
                "url"      : link,
                "source"   : feed_name,
                "published": published,
            })
    except Exception as e:
        print(f"⚠️  {feed_name} scrape failed: {e}")

    return articles


def scrape_rss(keyword: str, limit: int = 10) -> List[Dict]:
    """
    Scrape all configured RSS feeds for articles matching `keyword`.

    Parameters
    ----------
    keyword : str   — Search term (Urdu or English).
    limit   : int   — Max total articles to return.

    Returns
    -------
    List of article dicts: title, summary, url, source, published.
    """
    all_articles: List[Dict] = []

    for name, url in RSS_FEEDS.items():
        if len(all_articles) >= limit:
            break
        results = _parse_feed(name, url, keyword)
        all_articles.extend(results)

    # Deduplicate by URL
    seen, unique = set(), []
    for a in all_articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    return unique[:limit]


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    results = scrape_rss("Pakistan", limit=5)
    for r in results:
        print(f"[{r['source']}] {r['title'][:80]}")
