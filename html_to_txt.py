"""
HTML to Text Converter for Manual Web Page Processing.

This module converts locally saved HTML files to clean text format.
It's designed for cases where automated scraping fails and you need
to manually save web pages (using browser's "Save As" feature) and
then batch-convert them to text.

Use Cases:
    - Pages that require login/authentication
    - Pages that heavily rely on JavaScript rendering
    - Pages where automated scrapers are consistently blocked
    - Archive.org saved pages or other pre-downloaded HTML

Usage:
    1. Save HTML files manually to: mock_internet/manual_html/
    2. Run: python html_to_txt.py
    3. Find cleaned text files in: mock_internet/clean/

Input:
    - mock_internet/manual_html/*.html (or .htm, .mhtml)

Output:
    - mock_internet/clean/manual_{original_name}.txt

Note:
    Output files are prefixed with "manual_" to distinguish them
    from automatically scraped content.
"""

import os
from bs4 import BeautifulSoup
from typing import Optional

# --- CONFIGURATION ---
INPUT_DIR: str = "mock_internet/manual_html"  # Upload your HTML files here
OUTPUT_DIR: str = "mock_internet/clean"        # Where the clean .txt files go

# Supported HTML file extensions
SUPPORTED_EXTENSIONS: tuple = (".html", ".htm", ".mhtml")


def clean_html_file(filepath: str) -> Optional[str]:
    """
    Read and clean an HTML file, extracting only meaningful text content.

    This function handles various HTML encodings and removes clutter elements
    like scripts, styles, navigation bars, and other non-content elements.

    Args:
        filepath: Path to the HTML file to process.

    Returns:
        Cleaned text content with collapsed whitespace, or None if
        processing fails due to errors.

    Encoding Handling:
        - First attempts UTF-8 encoding
        - Falls back to Latin-1 if UTF-8 fails (handles most legacy encodings)

    Example:
        >>> content = clean_html_file("page.html")
        >>> print(content[:50])
        'Welcome to our website...'
    """
    try:
        # Try opening with UTF-8, fallback to Latin-1 if needed
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                content = f.read()

        soup = BeautifulSoup(content, 'html.parser')

        # Remove clutter elements (non-content HTML)
        junk_elements = [
            "script",   # JavaScript code
            "style",    # CSS styles
            "nav",      # Navigation menus
            "footer",   # Page footers
            "header",   # Page headers
            "noscript", # No-JavaScript fallback content
            "iframe",   # Embedded frames
            "svg"       # Vector graphics (often icons)
        ]
        for element in soup(junk_elements):
            element.extract()

        # Extract text with newline separators
        text = soup.get_text(separator='\n')

        # Clean whitespace: collapse multiple empty lines into one
        lines = (line.strip() for line in text.splitlines())
        clean_text = '\n'.join(chunk for chunk in lines if chunk)

        return clean_text

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def generate_output_filename(input_filename: str) -> str:
    """
    Generate output filename for a converted HTML file.

    Args:
        input_filename: Original HTML filename (e.g., "page.html")

    Returns:
        Output filename with "manual_" prefix and .txt extension
        (e.g., "manual_page.txt")

    Example:
        >>> generate_output_filename("tripadvisor_review.html")
        'manual_tripadvisor_review.txt'
    """
    base_name = os.path.splitext(input_filename)[0]
    return f"manual_{base_name}.txt"


def save_cleaned_content(
    content: str,
    source_filename: str,
    output_path: str
) -> None:
    """
    Save cleaned text content to a file with source header.

    Args:
        content: The cleaned text content to save.
        source_filename: Original HTML filename (for the header).
        output_path: Full path where the output file will be saved.

    Side Effects:
        Writes to the specified output_path with UTF-8 encoding.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"SOURCE_FILENAME: {source_filename}\n")
        f.write("-" * 50 + "\n")
        f.write(content)


def main() -> None:
    """
    Main function to convert all HTML files in INPUT_DIR to text.

    This function:
    1. Creates INPUT_DIR if it doesn't exist (prompting user to add files)
    2. Creates OUTPUT_DIR if it doesn't exist
    3. Finds all supported HTML files in INPUT_DIR
    4. Converts each file to clean text
    5. Saves results with "manual_" prefix in OUTPUT_DIR

    Returns:
        None. Results are saved to disk and progress is printed to stdout.

    Directory Structure:
        Input:  mock_internet/manual_html/
                ├── page1.html
                ├── page2.htm
                └── saved_page.mhtml

        Output: mock_internet/clean/
                ├── manual_page1.txt
                ├── manual_page2.txt
                └── manual_saved_page.txt
    """
    # Create input directory if it doesn't exist
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created '{INPUT_DIR}'. Please upload your HTML files there.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all HTML files
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(SUPPORTED_EXTENSIONS)
    ]
    
    print(f"--- Processing {len(files)} Manual HTML Files ---")

    # Process each file
    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        
        print(f"Processing: {filename}...")
        text_content = clean_html_file(input_path)
        
        if text_content:
            output_filename = generate_output_filename(filename)
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            save_cleaned_content(text_content, filename, output_path)
            print(f"   -> Saved to {output_filename}")
        else:
            print("   -> Skipped (Empty or Error)")

    print(f"\n--- Done! Check '{OUTPUT_DIR}' for your files. ---")


if __name__ == "__main__":
    main()
