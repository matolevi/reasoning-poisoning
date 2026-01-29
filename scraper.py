"""
Web Scraper Module for Reasoning Poisoning Research.

This module provides functionality to scrape web pages from a list of URLs,
extract clean text content by removing HTML clutter (scripts, styles, navigation),
and save the results to text files for further analysis.

The scraper is designed to build a "mock internet" corpus that can be used
to test how Large Language Models respond to web-retrieved context.

Usage:
    python scraper.py

Input:
    - urls.txt: A text file containing one URL per line

Output:
    - mock_internet/clean/*.txt: Cleaned text files, one per scraped URL
      Each file contains the source URL header followed by the extracted text content.

Example Output File Format:
    SOURCE_URL: https://example.com/page
    --------------------------------------------------
    [Extracted clean text content...]
"""

import requests
from bs4 import BeautifulSoup
import os
import hashlib
import time
from typing import Optional

# --- CONFIGURATION ---
URL_LIST_FILE: str = "urls.txt"
OUTPUT_DIR: str = "mock_internet/clean"

# HTTP request settings
REQUEST_TIMEOUT: int = 10  # seconds
REQUEST_DELAY: int = 1  # seconds between requests (rate limiting)
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)


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
        >>> html = "<html><script>alert('x')</script><p>Hello World</p></html>"
        >>> clean_text(html)
        'Hello World'
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove elements that typically contain non-content data
    junk_elements = ["script", "style", "nav", "footer", "iframe", "header"]
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


def scrape_single_url(url: str, session_headers: dict) -> Optional[str]:
    """
    Scrape a single URL and return its cleaned text content.

    Args:
        url: The URL to scrape.
        session_headers: HTTP headers to use for the request.

    Returns:
        Cleaned text content if successful, None if the request fails.

    Raises:
        requests.RequestException: If there's a network error (caught internally).
    """
    try:
        response = requests.get(url, headers=session_headers, timeout=REQUEST_TIMEOUT)
        
        if response.status_code != 200:
            print(f"   -> Failed (Status {response.status_code})")
            return None
        
        return clean_text(response.text)
        
    except requests.RequestException as e:
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
        The path to the saved file.
    """
    filename = generate_filename(url)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"SOURCE_URL: {url}\n")
        f.write("-" * 50 + "\n")
        f.write(content)
    
    return filepath


def scrape_urls() -> None:
    """
    Main function to scrape all URLs from the input file.

    Reads URLs from URL_LIST_FILE, scrapes each one with rate limiting,
    and saves the cleaned content to OUTPUT_DIR.

    The function:
    1. Validates that the URL list file exists
    2. Creates the output directory if needed
    3. Iterates through URLs with progress reporting
    4. Implements polite rate limiting (1 second delay between requests)
    5. Handles errors gracefully, continuing with remaining URLs

    Returns:
        None. Results are saved to disk and progress is printed to stdout.

    Side Effects:
        - Creates OUTPUT_DIR if it doesn't exist
        - Writes text files to OUTPUT_DIR
        - Prints progress and status messages
    """
    # Validate input file exists
    if not os.path.exists(URL_LIST_FILE):
        print(f"Error: {URL_LIST_FILE} not found. Please create it and paste your URLs there.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load URLs from file
    with open(URL_LIST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"--- Starting Scraping of {len(urls)} URLs ---")

    # Configure request headers to mimic a browser
    headers = {'User-Agent': USER_AGENT}

    # Process each URL
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Scraping: {url[:50]}...")
        
        text_content = scrape_single_url(url, headers)
        
        if text_content:
            save_scraped_content(text_content, url, OUTPUT_DIR)
        
        # Rate limiting - be polite to servers
        time.sleep(REQUEST_DELAY)

    print(f"\n--- Done! Files saved to {OUTPUT_DIR} ---")


if __name__ == "__main__":
    scrape_urls()
