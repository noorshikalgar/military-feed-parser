#!/usr/bin/env python3
"""
Military News & Exam Feed Parser with Content Extraction
Extracts full article content for actionable information (exams, schemes, notifications)
Outputs clean JSON for API consumption
"""

import argparse
import hashlib
import json
import sys
import time
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from urllib.parse import urlparse, parse_qs, unquote
from io import BytesIO

try:
    import requests
    import feedparser
    import pandas as pd
    from bs4 import BeautifulSoup
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
except ImportError as e:
    print(f"Missing required package: {e.name if hasattr(e, 'name') else e}")
    print("\nInstall dependencies with:")
    print("pip install requests feedparser pandas beautifulsoup4 lxml rich openpyxl")
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
    # Exam-focused feeds
    "NDA Exam Updates": "https://news.google.com/rss/search?q=NDA+exam+notification+2026&hl=en-IN&gl=IN&ceid=IN:en",
    "CDS Exam Updates": "https://news.google.com/rss/search?q=CDS+exam+notification+2026&hl=en-IN&gl=IN&ceid=IN:en",
    "AFCAT Updates": "https://news.google.com/rss/search?q=AFCAT+exam+notification+2026&hl=en-IN&gl=IN&ceid=IN:en",
    "Agniveer Recruitment": "https://news.google.com/rss/search?q=agniveer+recruitment+notification&hl=en-IN&gl=IN&ceid=IN:en",
    "Military Schemes": "https://news.google.com/rss/search?q=india+military+scheme+announcement&hl=en-IN&gl=IN&ceid=IN:en",
    
    # General defense news
    "India Defense News": "https://news.google.com/rss/search?q=india+defense+news&hl=en-IN&gl=IN&ceid=IN:en",
    "Defense Technology": "https://news.google.com/rss/search?q=india+defense+technology&hl=en-IN&gl=IN&ceid=IN:en",
    "Defense Deals": "https://news.google.com/rss/search?q=india+defense+contract&hl=en-IN&gl=IN&ceid=IN:en",
    
    # News sources
    "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "Indian Express": "https://indianexpress.com/feed/",
}

# Actionable content keywords (high priority for students)
ACTIONABLE_KEYWORDS = [
    # Exam related
    "notification", "apply", "application", "registration", "admit card",
    "exam date", "test date", "result", "cutoff", "syllabus", "eligibility",
    "last date", "deadline", "extended", "postponed", "rescheduled",
    
    # Recruitment
    "vacancy", "recruitment", "hiring", "joining", "selection", "shortlist",
    
    # Schemes
    "scheme", "program", "initiative", "benefit", "eligibility", "how to apply",
    "guidelines", "procedure",
    
    # Updates
    "announced", "launched", "released", "published", "notification",
]

# Military/Defense keywords
MILITARY_KEYWORDS = [
    "army", "navy", "air force", "military", "defense", "defence",
    "nda", "cds", "afcat", "agniveer", "ssb", "capf",
    "soldier", "officer", "recruitment", "exam",
    "drdo", "mod", "ministry of defense", "ministry of defence",
    "weapon", "missile", "aircraft", "technology", "border",
]

# Block irrelevant content
BLOCK_PATTERNS = [
    r"\bmovie\b", r"\bfilm\b", r"\bactor\b", r"\bactress\b",
    r"\bbollywood\b", r"\bcricket\b", r"\bipl\b", r"\bfootball\b",
    r"\bhoroscope\b", r"\brecipe\b", r"\bfashion\b",
]

CACHE_FILE = "seen_articles.json"
OUTPUT_JSON = "military_feed.json"
FUZZY_THRESHOLD = 85
REQUEST_TIMEOUT = 15

# ============================================================================
# CONTENT EXTRACTION
# ============================================================================

def extract_article_content(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[Dict]:
    """
    Extract full article content from URL using BeautifulSoup.
    Returns cleaned text content without HTML tags.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script, style, nav, footer, ads
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe', 'noscript']):
            tag.decompose()
        
        # Try to find main content area (common patterns)
        content = None
        
        # Strategy 1: Look for article tag
        article = soup.find('article')
        if article:
            content = article
        
        # Strategy 2: Look for common content divs
        if not content:
            for selector in ['div.article-content', 'div.story-content', 'div.post-content',
                           'div.entry-content', 'div.content', 'main', 'div[itemprop="articleBody"]']:
                content = soup.select_one(selector)
                if content:
                    break
        
        # Strategy 3: Find largest text block
        if not content:
            divs = soup.find_all('div')
            if divs:
                content = max(divs, key=lambda d: len(d.get_text(strip=True)))
        
        if not content:
            content = soup.body if soup.body else soup
        
        # Extract text
        text = content.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        
        # Extract metadata
        title = soup.find('meta', property='og:title')
        title = title['content'] if title else (soup.title.string if soup.title else '')
        
        description = soup.find('meta', property='og:description')
        description = description['content'] if description else ''
        
        # Try to extract published date
        pub_date = None
        for meta in ['article:published_time', 'datePublished', 'publishedDate']:
            date_tag = soup.find('meta', property=meta) or soup.find('meta', {'name': meta})
            if date_tag and date_tag.get('content'):
                pub_date = date_tag['content']
                break
        
        return {
            'full_content': text[:10000],  # Limit to 10k chars
            'word_count': len(text.split()),
            'extracted_title': title,
            'extracted_description': description,
            'extracted_date': pub_date,
        }
        
    except Exception as e:
        console.print(f"[yellow]Content extraction failed: {str(e)[:100]}[/yellow]")
        return None

def calculate_relevance_score(title: str, description: str, content: str = "") -> Dict:
    """
    Calculate relevance scores for different categories.
    Returns scores and matched keywords.
    """
    text = f"{title} {description} {content}".lower()
    
    # Check if blocked
    for pattern in BLOCK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return {
                'is_relevant': False,
                'blocked_by': pattern,
                'scores': {}
            }
    
    # Calculate scores
    actionable_matches = [kw for kw in ACTIONABLE_KEYWORDS if kw in text]
    military_matches = [kw for kw in MILITARY_KEYWORDS if kw in text]
    
    actionable_score = len(actionable_matches)
    military_score = len(military_matches)
    
    # Priority scoring
    # High priority: Has actionable keywords (exam, apply, notification, etc.)
    # Medium priority: Military news without actionable items
    is_actionable = actionable_score >= 1
    is_military = military_score >= 1
    
    priority = "low"
    if is_actionable and is_military:
        priority = "high"
    elif is_actionable:
        priority = "high"
    elif is_military:
        priority = "medium"
    
    return {
        'is_relevant': is_military or is_actionable,
        'priority': priority,
        'scores': {
            'actionable': actionable_score,
            'military': military_score,
            'total': actionable_score + military_score,
        },
        'matched_keywords': {
            'actionable': actionable_matches[:5],
            'military': military_matches[:5],
        }
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_feed_with_requests(url: str, timeout: int = REQUEST_TIMEOUT) -> feedparser.FeedParserDict:
    """Fetch RSS feed."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout, verify=True)
        response.raise_for_status()
        return feedparser.parse(BytesIO(response.content))
    except requests.exceptions.SSLError:
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        return feedparser.parse(BytesIO(response.content))

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

def normalize_datetime(dt: datetime) -> datetime:
    """Normalize datetime to UTC."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def parse_date(date_str: str) -> datetime:
    """Parse date from string."""
    if not date_str:
        return datetime.now(timezone.utc)
    
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
            clean_date = date_str.replace("GMT", "+0000").replace("UTC", "+0000")
            parsed_date = datetime.strptime(clean_date[:30], fmt)
            return normalize_datetime(parsed_date)
        except:
            continue
    
    return datetime.now(timezone.utc)

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

# ============================================================================
# MAIN PARSER
# ============================================================================

def fetch_feeds(days_back: int = 7, max_items: int = 100, 
                extract_content: bool = True, debug: bool = False) -> List[Dict]:
    """Fetch feeds with optional content extraction."""
    articles = []
    cutoff_date = normalize_datetime(datetime.now() - timedelta(days=days_back))
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Fetching feeds...", total=len(RSS_FEEDS))
        
        for category, feed_url in RSS_FEEDS.items():
            progress.update(task, description=f"[cyan]Fetching: {category}")
            
            try:
                feed = fetch_feed_with_requests(feed_url)
                
                if not feed.entries:
                    progress.advance(task)
                    continue
                
                console.print(f"[green]✓ {category}: {len(feed.entries)} entries[/green]")
                
                for entry in feed.entries[:max_items]:
                    try:
                        date_str = entry.get("published", entry.get("updated", ""))
                        pub_date = parse_date(date_str)
                        
                        if pub_date < cutoff_date:
                            continue
                        
                        title = entry.get("title", "").strip()
                        url = entry.get("link", "").strip()
                        description = entry.get("summary", "").strip()
                        description = re.sub(r'<[^>]+>', '', description)
                        
                        if not title or not url:
                            continue
                        
                        article = {
                            "title": title,
                            "url": url,
                            "published_date": pub_date.isoformat(),
                            "source_category": category,
                            "description": description[:500],
                            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        
                        # Extract full content if enabled
                        if extract_content:
                            time.sleep(0.3)  # Rate limiting
                            content_data = extract_article_content(url)
                            if content_data:
                                article.update(content_data)
                        
                        articles.append(article)
                        
                    except Exception as e:
                        if debug:
                            console.print(f"[yellow]Entry error: {str(e)[:80]}[/yellow]")
                        continue
                
                time.sleep(0.5)
                
            except Exception as e:
                console.print(f"[red]✗ {category}: {str(e)[:80]}[/red]")
            
            progress.advance(task)
    
    console.print(f"\n[bold green]✓ Fetched {len(articles)} articles[/bold green]")
    return articles

def filter_and_score_articles(articles: List[Dict], debug: bool = False) -> Tuple[List[Dict], int, int]:
    """Filter and score articles by relevance."""
    cache = load_seen_cache()
    seen_hashes = set(cache.get("hashes", []))
    seen_titles = cache.get("titles", [])
    seen_urls = set(cache.get("urls", []))
    
    filtered = []
    irrelevant_count = 0
    duplicate_count = 0
    
    console.print("\n[cyan]Scoring articles...[/cyan]")
    
    for i, article in enumerate(articles):
        # Check duplicates
        if is_duplicate(article["title"], article["url"], 
                       seen_hashes, seen_titles, seen_urls):
            duplicate_count += 1
            continue
        
        # Calculate relevance score
        content = article.get("full_content", "")
        relevance = calculate_relevance_score(
            article["title"], 
            article["description"], 
            content
        )
        
        if not relevance['is_relevant']:
            irrelevant_count += 1
            continue
        
        # Add scoring data
        article["relevance"] = relevance
        article["content_hash"] = hash_entry(article["title"], article["url"])
        
        filtered.append(article)
        seen_hashes.add(article["content_hash"])
        seen_titles.append(article["title"])
        seen_urls.add(normalize_url(article["url"]))
    
    # Sort by priority (high first) and then by date
    filtered.sort(key=lambda x: (
        0 if x['relevance']['priority'] == 'high' else (1 if x['relevance']['priority'] == 'medium' else 2),
        -datetime.fromisoformat(x['published_date']).timestamp()
    ))
    
    # Save cache
    cache["hashes"] = list(seen_hashes)[-2000:]
    cache["titles"] = seen_titles[-2000:]
    cache["urls"] = list(seen_urls)[-2000:]
    save_seen_cache(cache)
    
    return filtered, irrelevant_count, duplicate_count

def save_to_json(articles: List[Dict], filename: str = OUTPUT_JSON):
    """Save articles to JSON file for API consumption."""
    if not articles:
        console.print("[yellow]No articles to save.[/yellow]")
        return None
    
    # Create API-ready structure
    output = {
        "meta": {
            "total_count": len(articles),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "priority_counts": {
                "high": len([a for a in articles if a['relevance']['priority'] == 'high']),
                "medium": len([a for a in articles if a['relevance']['priority'] == 'medium']),
                "low": len([a for a in articles if a['relevance']['priority'] == 'low']),
            }
        },
        "articles": articles
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green bold]✓ Saved JSON to {filename}[/green bold]")
    return filename

def save_to_csv(articles: List[Dict], output_format: str = "csv"):
    """Save to CSV/Excel (legacy support)."""
    if not articles:
        return None
    
    # Flatten for CSV
    df_data = []
    for article in articles:
        row = {
            'title': article['title'],
            'url': article['url'],
            'published_date': article['published_date'],
            'source_category': article['source_category'],
            'description': article['description'],
            'priority': article['relevance']['priority'],
            'actionable_score': article['relevance']['scores']['actionable'],
            'military_score': article['relevance']['scores']['military'],
            'word_count': article.get('word_count', 0),
            'content_preview': article.get('full_content', '')[:200],
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    if output_format == "excel":
        filename = f"military_feed_{date_str}.xlsx"
        df.to_excel(filename, index=False, engine="openpyxl")
    else:
        filename = f"military_feed_{date_str}.csv"
        df.to_csv(filename, index=False, encoding="utf-8")
    
    console.print(f"[green]✓ Saved {output_format.upper()} to {filename}[/green]")
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
    table.add_row("✓ Final Articles", str(len(articles)), style="green bold")
    
    if articles:
        high = len([a for a in articles if a['relevance']['priority'] == 'high'])
        medium = len([a for a in articles if a['relevance']['priority'] == 'medium'])
        table.add_row("  - High Priority", str(high), style="green")
        table.add_row("  - Medium Priority", str(medium), style="yellow")
    
    console.print(table)
    
    # Show high priority articles
    high_priority = [a for a in articles if a['relevance']['priority'] == 'high']
    if high_priority:
        console.print("\n[bold green]🎯 High Priority (Actionable Items):[/bold green]")
        for i, article in enumerate(high_priority[:10], 1):
            date_str = datetime.fromisoformat(article['published_date']).strftime('%Y-%m-%d')
            keywords = ', '.join(article['relevance']['matched_keywords']['actionable'][:3])
            console.print(
                f"\n{i}. [bold]{article['title']}[/bold]\n"
                f"   [dim]Source: {article['source_category']} | {date_str}[/dim]\n"
                f"   [green]Keywords: {keywords}[/green]\n"
                f"   [blue]{article['url'][:90]}...[/blue]"
            )

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Military News & Exam Feed Parser with Content Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (with content extraction)
  python3 feed-parser.py --days 7
  
  # Fast mode (no content extraction)
  python3 feed-parser.py --no-extract --days 14
  
  # Debug mode
  python3 feed-parser.py --debug --days 30
  
  # Export to CSV/Excel
  python3 feed-parser.py --export csv
  python3 feed-parser.py --export excel
        """
    )
    
    parser.add_argument("--days", type=int, default=7, help="Fetch last N days (default: 7)")
    parser.add_argument("--max-items", type=int, default=50, help="Max items per feed (default: 50)")
    parser.add_argument("--no-extract", action="store_true", help="Skip full content extraction (faster)")
    parser.add_argument("--export", choices=["csv", "excel", "none"], default="none", help="Export to CSV/Excel")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument("--clear-cache", action="store_true", help="Clear seen articles cache")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        if Path(CACHE_FILE).exists():
            Path(CACHE_FILE).unlink()
            console.print("[green]✓ Cache cleared[/green]\n")
    
    console.print(Panel.fit(
        "[bold cyan]Military News & Exam Feed Parser[/bold cyan]\n"
        f"Fetching last {args.days} days • Content extraction: {'ON' if not args.no_extract else 'OFF'}",
        border_style="cyan"
    ))
    
    # Fetch
    extract_content = not args.no_extract
    articles = fetch_feeds(args.days, args.max_items, extract_content, args.debug)
    
    if not articles:
        console.print("\n[red bold]No articles fetched![/red bold]")
        return
    
    # Filter and score
    filtered, irrelevant, duplicates = filter_and_score_articles(articles, args.debug)
    
    # Save to JSON (main output)
    save_to_json(filtered)
    
    # Optional CSV/Excel export
    if args.export != "none":
        save_to_csv(filtered, args.export)
    
    # Display summary
    display_summary(filtered, irrelevant, duplicates, len(articles))
    
    console.print(f"\n[green bold]✓ JSON API file: {OUTPUT_JSON}[/green bold]")
    console.print("\n[green bold]✓ Complete![/green bold]")

if __name__ == "__main__":
    main()