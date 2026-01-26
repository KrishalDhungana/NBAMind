"""
==========================================================================================
NBAMind: scraper.py
------------------------------------------------------------------------------------------
Scrapes NBA player contract/salary data from ESPN.
Leverages fetcher.py for rate limiting and caching patterns.
==========================================================================================
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import polars as pl
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from nbamind.data import fetcher

logger = logging.getLogger(__name__)

ESPN_BASE_URL = "https://www.espn.com/nba/salaries/_/year/{year}/page/{page}"

# Headers specifically for ESPN to avoid 403 Forbidden caused by NBA-specific headers (Host/Origin)
ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

def _get_cache_path(season: str) -> Path:
    """Defines a stable cache path for salary data."""
    fetcher.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return fetcher.CACHE_DIR / f"{season}_espn_salaries.json"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_page_html(url: str) -> str:
    """Fetches a single URL with rate limiting and retries."""
    fetcher.rate_limit()
    logger.info(f"Scraping URL: {url}")
    response = requests.get(url, headers=ESPN_HEADERS, timeout=15)
    
    response.raise_for_status()
    return response.text


def fetch_salaries(season: str) -> Optional[pl.DataFrame]:
    """
    Fetches salary data for a specific season (e.g., '2023-24').
    Checks cache first unless it is the current season.
    """
    is_current_season = (season == fetcher.get_current_season())
    cache_path = _get_cache_path(season)

    # Try cache
    if not is_current_season and cache_path.exists():
        logger.info(f"CACHE HIT: ESPN salaries for {season}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return pl.DataFrame(json.load(f))

    # Scrape
    logger.info(f"Starting scrape for ESPN salaries: {season}")
    year_end = int(season[:4]) + 1
    all_salaries = []
    page = 1
    
    while True:
        url = ESPN_BASE_URL.format(year=year_end, page=page)
        try:
            html = _fetch_page_html(url)
            soup = BeautifulSoup(html, "html.parser")
            
            # Find table
            table = soup.find("table", class_="tablehead")
            if not table:
                break

            rows = table.find_all("tr")

            extracted_count = 0
            for row in rows:
                # Skip header rows
                classes = row.get("class", [])
                if "colhead" in classes or "stathead" in classes:
                    continue

                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                
                # Format: Rank, Name (includes position), Team, Salary
                raw_name = cols[1].get_text(strip=True)
                raw_salary = cols[3].get_text(strip=True)

                # Clean Data
                name = raw_name.split(",")[0].strip()
                salary_str = re.sub(r"[^\d]", "", raw_salary)
                
                if name and salary_str:
                    all_salaries.append({
                        "PLAYER_NAME": name,
                        "SALARY": int(salary_str),
                        "SEASON_YEAR": season
                    })
                    extracted_count += 1
            
            if extracted_count == 0:
                break
                
            page += 1
            
        except Exception as e:
            logger.error(f"Error scraping page {page} for season {season}: {e}")
            break

    if not all_salaries:
        logger.warning(f"No salary data found for {season}.")
        return None

    # Save to Cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_salaries, f, indent=2)
    logger.info(f"Successfully scraped {len(all_salaries)} salary records for {season}.")
    return pl.DataFrame(all_salaries)