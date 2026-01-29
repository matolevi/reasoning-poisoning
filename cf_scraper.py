"""
CloudFlare-Bypassing Web Scraper for Reasoning Poisoning Research.

This module provides functionality to scrape web pages that are protected
by CloudFlare's anti-bot measures. It uses the 'cloudscraper' library which
can bypass JavaScript challenges that would block regular HTTP requests.

Use this scraper for URLs that return 403 Forbidden errors with the
standard scraper (scraper.py).

Usage:
    python cf_scraper.py

Input:
    - urls.txt: A text file containing one URL per line (typically URLs
      that failed with the standard scraper)

Output:
    - mock_internet/clean/*.txt: Cleaned text files, one per scraped URL

Dependencies:
    - cloudscraper: pip install cloudscraper
    - beautifulsoup4: pip install beautifulsoup4

Note:
    This scraper uses longer random delays (2-5 seconds) between requests
    to appear more human-like and avoid detection.
"""

import cloudscraper
from bs4 import BeautifulSoup
import os
import hashlib
import time
import random
from typing import Optional

# --- CONFIGURATION ---
URL_LIST_FILE: str = "urls.txt"
OUTPUT_DIR: str = "mock_internet/clean"

# Request timing settings (more conservative to avoid detection)
MIN_DELAY: float = 2.0  # Minimum seconds between requests
MAX_DELAY: float = 5.0  # Maximum seconds between requests
REQUEST_TIMEOUT: int = 15  # seconds (longer timeout for CloudFlare sites)


def clean_text(html_content: str) -> str:
    """
    Extract clean text from HTML content by removing clutter elements.

    This function parses HTML and removes elements that typically don't contain
    useful content (scripts, styles, navigation, footers, etc.), then extracts
    the remaining text with proper line separation.

    Args:
        html_content: Raw HTML string to process.

    Returns:
        Cleaned text with multiple blank lines collapsed to single newlines.
        Non-empty lines are preserved with their content stripped of
        leading/trailing whitespace.

    Example:
        >>> html = "<html><noscript>Enable JS</noscript><p>Content</p></html>"
        >>> clean_text(html)
        'Content'
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove elements that typically contain non-content data
    # Note: includes 'noscript' which often contains CloudFlare messages
    junk_elements = ["script", "style", "nav", "footer", "iframe", "header", "noscript"]
    for element in soup(junk_elements):
        element.extract()
    
    # Extract text with newline separators between elements
    text = soup.get_text(separator='\n')
    
    # Collapse multiple newlines and strip whitespace from each line
    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(chunk for chunk in lines if chunk)


def generate_filename(url: str) -> str:
    """
    Generate a unique filename from a URL.

    Creates a filename combining the domain name and a short MD5 hash
    of the full URL to ensure uniqueness while remaining human-readable.

    Args:
        url: The full URL to generate a filename for.

    Returns:
        A filename in format: "{domain}_{hash}.txt"
        where domain has "www." prefix removed and hash is 6 characters.

    Example:
        >>> generate_filename("https://www.example.com/page/123")
        'example.com_a1b2c3.txt'
    """
    # Extract domain, removing protocol and www prefix
    domain = url.split("//")[-1].split("/")[0].replace("www.", "")
    
    # Generate short hash for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    
    return f"{domain}_{url_hash}.txt"


def create_cloudscraper_session() -> cloudscraper.CloudScraper:
    """
    Create a CloudScraper session configured to mimic a real browser.

    The session is configured to appear as Chrome on Windows desktop,
    which helps bypass CloudFlare's browser verification checks.

    Returns:
        A configured CloudScraper instance ready for making requests.
    """
    return cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )


def scrape_single_url(
    scraper: cloudscraper.CloudScraper,
    url: str
) -> Optional[str]:
    """
    Scrape a single URL using CloudScraper and return cleaned text.

    Args:
        scraper: A configured CloudScraper session.
        url: The URL to scrape.

    Returns:
        Cleaned text content if successful, None if the request fails.

    Note:
        This function does not implement retry logic. CloudScraper
        handles the JavaScript challenge internally.
    """
    try:
        response = scraper.get(url, timeout=REQUEST_TIMEOUT)
        
        if response.status_code != 200:
            print(f"   -> Still Failed (Status {response.status_code})")
            return None
        
        return clean_text(response.text)
        
    except Exception as e:
        print(f"   -> Error: {e}")
        return None


def save_scraped_content(content: str, url: str, output_dir: str) -> str:
    """
    Save scraped content to a text file with source URL header.

    Args:
        content: The cleaned text content to save.
        url: The source URL (included as header in the file).
        output_dir: Directory to save the output file.

    Returns:
        The filename (not full path) of the saved file.
    """
    filename = generate_filename(url)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"SOURCE_URL: {url}\n")
        f.write("-" * 50 + "\n")
        f.write(content)
    
    return filename


def scrape_tough_urls() -> None:
    """
    Main function to scrape CloudFlare-protected URLs.

    This function is designed for URLs that return 403 errors with
    standard HTTP requests. It uses cloudscraper to bypass JavaScript
    challenges and implements human-like random delays.

    The function:
    1. Validates that the URL list file exists
    2. Creates a CloudScraper session mimicking Chrome
    3. Iterates through URLs with random delays (2-5 seconds)
    4. Saves successful scrapes to OUTPUT_DIR
    5. Reports progress and success/failure status

    Returns:
        None. Results are saved to disk and progress is printed to stdout.

    Side Effects:
        - Creates OUTPUT_DIR if it doesn't exist
        - Writes text files to OUTPUT_DIR
        - Prints progress and status messages
    """
    # Validate input file exists
    if not os.path.exists(URL_LIST_FILE):
        print(f"Please create '{URL_LIST_FILE}' with the URLs that returned 403.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize CloudScraper session
    scraper = create_cloudscraper_session()

    # Load URLs from file
    with open(URL_LIST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"--- Attempting to Scrape {len(urls)} Tough URLs ---")

    # Process each URL
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Scraping: {url[:50]}...")
        
        # Random delay to appear human-like
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)
        
        text_content = scrape_single_url(scraper, url)
        
        if text_content:
            filename = save_scraped_content(text_content, url, OUTPUT_DIR)
            print(f"   -> Success! Saved to {filename}")

    print(f"\n--- Done! Files saved to {OUTPUT_DIR} ---")


if __name__ == "__main__":
    scrape_tough_urls()
