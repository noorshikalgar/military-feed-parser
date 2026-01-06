#!/usr/bin/env python3
"""
Military News RSS Feed Parser - SSL Fixed Version
Uses requests library to bypass SSL certificate issues.
"""

import argparse
import hashlib
import json
import sys
import time
import re
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urlparse, parse_qs, unquote
from io import BytesIO

try:
    import requests
    import feedparser
    import pandas as pd
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
except ImportError as e:
    print(f"Missing required package: {e.name if hasattr(e, 'name') else e}")
    print("\nInstall dependencies with:")
    print("pip install requests feedparser pandas rich openpyxl")
    sys.exit(1)

# Optional
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

RSS_FEEDS = {
    # Google News
    "India Defense News": "https://news.google.com/rss/search?q=india+defense&hl=en-IN&gl=IN&ceid=IN:en",
    "Indian Military": "https://news.google.com/rss/search?q=indian+army+OR+navy+OR+%22air+force%22&hl=en-IN&gl=IN&ceid=IN:en",
    "Defense Contracts": "https://news.google.com/rss/search?q=defense+contract+india&hl=en-IN&gl=IN&ceid=IN:en",
    "Military Exams": "https://news.google.com/rss/search?q=NDA+OR+CDS+OR+AFCAT&hl=en-IN&gl=IN&ceid=IN:en",
    "Border Security": "https://news.google.com/rss/search?q=india+border+security&hl=en-IN&gl=IN&ceid=IN:en",
    
    # News sources
    "Defense News": "https://www.defensenews.com/arc/outboundfeeds/rss/",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "Indian Express": "https://indianexpress.com/feed/",
    "Economic Times Defense": "https://economictimes.indiatimes.com/defence/rssfeeds/11000003.cms",
}

RELEVANCE_KEYWORDS = [
    # Core military
    "army", "navy", "air force", "military", "defense", "defence", 
    "soldier", "troop", "officer", "chief", "general", "admiral", "marshal",
    
    # Weapons
    "weapon", "missile", "aircraft", "jet", "fighter", "helicopter", "tank",
    "submarine", "ship", "warship", "drone", "gun", "rifle", "artillery",
    
    # Indian specific
    "tejas", "brahmos", "akash", "arjun", "ins ", "iaf ", "drdo", "hal",
    "bsf", "crpf", "itbp", "cisf",
    
    # Operations
    "exercise", "drill", "operation", "training", "border", "patrol",
    "deployment", "combat", "strategic",
    
    # Administrative
    "recruitment", "exam", "nda", "cds", "afcat", "agniveer", "ssb",
    "budget", "deal", "contract", "acquisition", "procurement",
    "scheme", "policy", "modernization", "upgrade", "indigenous",
    
    # Organizations
    "mod", "ministry of defense", "ministry of defence", "headquarters",
]

BLOCK_KEYWORDS = [
    "movie", "film", "actor", "actress", "bollywood", "hollywood",
    "cricket", "football", "sports", "match", "ipl", "tournament",
    "horoscope", "astrology", "recipe", "fashion", "lifestyle",
]

CACHE_FILE = "seen_articles.json"
FUZZY_THRESHOLD = 85

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_feed_with_requests(url: str, timeout: int = 15) -> feedparser.FeedParserDict:
    """Fetch RSS feed using requests to bypass SSL issues."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    try:
        # Use requests with SSL verification disabled as fallback
        response = requests.get(url, headers=headers, timeout=timeout, verify=True)
        response.raise_for_status()
        
        # Parse the content with feedparser
        return feedparser.parse(BytesIO(response.content))
    except requests.exceptions.SSLError:
        # Retry without SSL verification
        try:
            response = requests.get(url, headers=headers, timeout=timeout, verify=False)
            response.raise_for_status()
            return feedparser.parse(BytesIO(response.content))
        except Exception as e:
            raise Exception(f"SSL retry failed: {e}")
    except Exception as e:
        raise Exception(f"Request failed: {e}")

def hash_entry(title: str, url: str) -> str:
    """Generate unique hash."""
    normalized_url = normalize_url(url)
    content = f"{title.lower().strip()}|{normalized_url}"
    return hashlib.md5(content.encode()).hexdigest()

def normalize_url(url: str) -> str:
    """Normalize URL."""
    try:
        parsed = urlparse(url)
        if "news.google.com" in parsed.netloc:
            query_params = parse_qs(parsed.query)
            if "url" in query_params:
                return normalize_url(unquote(query_params["url"][0]))
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except:
        return url

def is_relevant(title: str, description: str, debug: bool = False) -> bool:
    """Check relevance with relaxed matching."""
    text = f"{title} {description}".lower()
    
    # Block unwanted
    for block in BLOCK_KEYWORDS:
        if block in text:
            if debug:
                console.print(f"[red]  ✗ Blocked by: {block}[/red]")
            return False
    
    # Check for military keywords
    matches = []
    for keyword in RELEVANCE_KEYWORDS:
        if keyword in text:
            matches.append(keyword)
            if len(matches) >= 1:  # At least 1 match
                if debug:
                    console.print(f"[green]  ✓ Matched: {', '.join(matches[:3])}[/green]")
                return True
    
    if debug:
        console.print("[yellow]  ✗ No keyword match[/yellow]")
    
    return False

def load_seen_cache() -> Dict:
    """Load cache."""
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"hashes": [], "titles": [], "urls": []}

def save_seen_cache(cache: Dict):
    """Save cache."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        console.print(f"[yellow]Cache save error: {e}[/yellow]")

def is_duplicate(title: str, url: str, seen_hashes: Set[str], 
                 seen_titles: List[str], seen_urls: Set[str]) -> bool:
    """Check duplicates."""
    article_hash = hash_entry(title, url)
    if article_hash in seen_hashes:
        return True
    
    normalized = normalize_url(url)
    if normalized in seen_urls:
        return True
    
    if FUZZY_AVAILABLE:
        for seen_title in seen_titles[-300:]:
            if fuzz.ratio(title.lower(), seen_title.lower()) >= FUZZY_THRESHOLD:
                return True
    
    return False

def parse_date(date_str: str) -> datetime:
    """Parse date from various formats."""
    if not date_str:
        return datetime.now()
    
    formats = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            # Handle timezone in date string
            clean_date = date_str.replace("GMT", "+0000")
            return datetime.strptime(clean_date[:25], fmt)
        except:
            continue
    
    return datetime.now()

# ============================================================================
# PARSER
# ============================================================================

def fetch_feeds(days_back: int = 7, max_items: int = 100, 
                debug: bool = False) -> List[Dict]:
    """Fetch feeds using requests library."""
    articles = []
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    # Disable SSL warnings when using verify=False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Fetching...", total=len(RSS_FEEDS))
        
        for category, feed_url in RSS_FEEDS.items():
            progress.update(task, description=f"[cyan]Fetching: {category}")
            
            try:
                feed = fetch_feed_with_requests(feed_url)
                
                if debug:
                    console.print(f"\n[bold]{category}[/bold]")
                    console.print(f"URL: {feed_url[:80]}...")
                    console.print(f"Entries: {len(feed.entries)}")
                    if feed.get('feed', {}).get('title'):
                        console.print(f"Feed title: {feed.feed.title}")
                
                if not feed.entries:
                    if debug:
                        console.print(f"[yellow]No entries found[/yellow]")
                    progress.advance(task)
                    continue
                
                console.print(f"[green]✓ {category}: {len(feed.entries)} entries[/green]")
                
                # Show sample in debug
                if debug and feed.entries:
                    first = feed.entries[0]
                    console.print(f"Sample: {first.get('title', 'N/A')[:80]}")
                
                for entry in feed.entries[:max_items]:
                    # Parse date
                    date_str = entry.get("published", entry.get("updated", 
                                        entry.get("pubDate", "")))
                    pub_date = parse_date(date_str)
                    
                    # Filter by date
                    if pub_date < cutoff_date:
                        continue
                    
                    # Extract data
                    title = entry.get("title", "").strip()
                    url = entry.get("link", "").strip()
                    
                    # Get description
                    description = (entry.get("summary", "") or 
                                 entry.get("description", "") or 
                                 entry.get("content", [{}])[0].get("value", "") if isinstance(entry.get("content"), list) else "")
                    
                    # Clean HTML
                    description = re.sub(r'<[^>]+>', '', str(description)).strip()
                    
                    if not title or not url:
                        continue
                    
                    article = {
                        "title": title,
                        "url": url,
                        "published_date": pub_date,
                        "source_category": category,
                        "description": description[:500],
                        "fetch_timestamp": datetime.now(),
                    }
                    
                    articles.append(article)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                console.print(f"[red]✗ {category}: {str(e)[:80]}[/red]")
                if debug:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()[:300]}[/dim]")
            
            progress.advance(task)
    
    console.print(f"\n[bold green]✓ Fetched {len(articles)} articles[/bold green]")
    return articles

def filter_and_deduplicate(articles: List[Dict], debug: bool = False) -> Tuple[List[Dict], int, int]:
    """Filter articles."""
    cache = load_seen_cache()
    seen_hashes = set(cache.get("hashes", []))
    seen_titles = cache.get("titles", [])
    seen_urls = set(cache.get("urls", []))
    
    filtered = []
    irrelevant_count = 0
    duplicate_count = 0
    
    if debug:
        console.print("\n[cyan]Filtering sample articles...[/cyan]")
    
    for i, article in enumerate(articles):
        show_debug = debug and i < 5
        
        if show_debug:
            console.print(f"\n[bold]Article {i+1}:[/bold]")
            console.print(f"Title: {article['title'][:100]}")
        
        # Check relevance
        if not is_relevant(article["title"], article["description"], show_debug):
            irrelevant_count += 1
            continue
        
        # Check duplicates
        if is_duplicate(article["title"], article["url"], 
                       seen_hashes, seen_titles, seen_urls):
            duplicate_count += 1
            if show_debug:
                console.print("[yellow]  ✗ Duplicate[/yellow]")
            continue
        
        # Track
        article_hash = hash_entry(article["title"], article["url"])
        article["content_hash"] = article_hash
        
        filtered.append(article)
        seen_hashes.add(article_hash)
        seen_titles.append(article["title"])
        seen_urls.add(normalize_url(article["url"]))
        
        if show_debug:
            console.print("[green]  ✓ Added to results[/green]")
    
    # Save cache
    cache["hashes"] = list(seen_hashes)[-2000:]
    cache["titles"] = seen_titles[-2000:]
    cache["urls"] = list(seen_urls)[-2000:]
    save_seen_cache(cache)
    
    return filtered, irrelevant_count, duplicate_count

def save_to_file(articles: List[Dict], output_format: str = "csv"):
    """Save to file."""
    if not articles:
        console.print("[yellow]No new articles to save.[/yellow]")
        return None
    
    df = pd.DataFrame(articles)
    df["published_date"] = pd.to_datetime(df["published_date"])
    df["fetch_timestamp"] = pd.to_datetime(df["fetch_timestamp"])
    df = df.sort_values("published_date", ascending=False)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    if output_format == "excel":
        filename = f"military_news_{date_str}.xlsx"
        output_path = Path(filename)
        
        if output_path.exists():
            existing_df = pd.read_excel(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["content_hash"])
        
        df.to_excel(filename, index=False, engine="openpyxl")
    else:
        filename = f"military_news_{date_str}.csv"
        output_path = Path(filename)
        
        if output_path.exists():
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["content_hash"])
        
        df.to_csv(filename, index=False, encoding="utf-8")
    
    console.print(f"\n[green bold]✓ Saved {len(df)} articles to {filename}[/green bold]")
    return filename

def display_summary(articles: List[Dict], irrelevant: int, 
                   duplicates: int, total_fetched: int):
    """Display summary."""
    table = Table(title="Results Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    
    table.add_row("Total Fetched", str(total_fetched))
    table.add_row("Filtered (Irrelevant)", str(irrelevant))
    table.add_row("Filtered (Duplicates)", str(duplicates))
    table.add_row("✓ New Articles", str(len(articles)), style="green bold")
    
    console.print(table)
    
    if articles:
        console.print("\n[bold cyan]Recent Articles:[/bold cyan]")
        for i, article in enumerate(articles[:15], 1):
            date_str = article['published_date'].strftime('%Y-%m-%d %H:%M')
            console.print(
                f"\n{i}. [bold]{article['title']}[/bold]\n"
                f"   [dim]Source: {article['source_category']} | {date_str}[/dim]\n"
                f"   [blue]{article['url'][:100]}...[/blue]"
            )

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Military News RSS Parser (SSL Fixed)")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument("--output", choices=["csv", "excel"], default="csv")
    parser.add_argument("--days", type=int, default=7, help="Fetch last N days")
    parser.add_argument("--max-items", type=int, default=100, help="Max items per feed")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        if Path(CACHE_FILE).exists():
            Path(CACHE_FILE).unlink()
            console.print("[green]✓ Cache cleared[/green]\n")
    
    console.print(Panel.fit(
        "[bold cyan]Military News RSS Feed Parser[/bold cyan]\n"
        f"Fetching last {args.days} days from {len(RSS_FEEDS)} sources\n"
        f"[dim]SSL issues fixed with requests library[/dim]",
        border_style="cyan"
    ))
    
    # Fetch
    articles = fetch_feeds(args.days, args.max_items, args.debug)
    
    if not articles:
        console.print("\n[red bold]No articles fetched![/red bold]")
        console.print("Try: python3 feed-parser.py --debug --days 30")
        return
    
    # Filter
    filtered, irrelevant, duplicates = filter_and_deduplicate(articles, args.debug)
    
    # Save
    filename = save_to_file(filtered, args.output)
    
    # Summary
    display_summary(filtered, irrelevant, duplicates, len(articles))
    
    if filename:
        console.print(f"\n[green bold]✓ Results saved to: {filename}[/green bold]")
    
    console.print("\n[green bold]✓ Complete![/green bold]")

if __name__ == "__main__":
    main()