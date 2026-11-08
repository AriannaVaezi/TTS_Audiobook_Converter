"""
url_to_txt.py
=============
Scrapes a URL using Playwright, extracts clean prose text,
and saves the result as a .txt file inside raw_txt/.

Usage:
  python url_to_txt.py <URL> [options]

Options:
  --reader       Enable Reader Mode (strips nav/headers/sidebars via Readability.js).
                 Recommended for Ellipsus, Tumblr, AO3, and blog-style pages.
  --visible      Run the browser visibly (non-headless) for debugging.
  --out NAME     Override the output filename (without .txt extension).

Examples:
  python url_to_txt.py https://archiveofourown.org/works/12345 --reader
  python url_to_txt.py https://ellipsus.com/story/abc --reader
  python url_to_txt.py https://example.com/chapter1 --out my_story
"""

import argparse
import os
import re
import sys

RAW_TXT_DIR = "raw_txt"


def sanitize_filename(title: str) -> str:
    """Removes characters unsafe for filenames and trims length."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:120] if cleaned else "webpage_output"


def html_to_clean_text(html_content: str) -> str:
    """
    Converts HTML to plain prose text:
    - Preserves paragraph structure with blank lines
    - Converts <br> to newlines
    - Strips all HTML tags
    - Normalises unicode dashes and smart quotes for TTS compatibility
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "button", "form", "iframe", "noscript"]):
        tag.decompose()

    # Convert <br> → newline markers before stripping
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Convert block-level elements to double-newline paragraph breaks
    block_tags = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                  "li", "blockquote", "pre", "tr", "section", "article"]
    for tag in soup.find_all(block_tags):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")

    text = soup.get_text(separator="")

    # Normalise whitespace while preserving paragraph structure
    # Collapse more-than-two blank lines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Normalise unicode dashes, smart quotes, non-breaking spaces
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")

    return text.strip()


def scrape_url_to_txt(
    url: str,
    reader_mode: bool = False,
    headless: bool = True,
    output_name: str = None,
):
    """
    Main scraping function.
    Returns the saved txt file path on success, or None on failure.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[X] Playwright is not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    os.makedirs(RAW_TXT_DIR, exist_ok=True)

    print(f"[*] Opening '{url}'...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )

        page = context.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=40000)
        except Exception as e:
            print(f"[!] Warning during page load: {e} — continuing anyway.")

        # Wait for initial story paragraphs to hydrate
        print("[*] Waiting for page content to load...")
        try:
            page.wait_for_function(
                "document.querySelectorAll('p').length > 3", timeout=20000
            )
        except Exception:
            print("[!] Paragraph wait timed out. Proceeding with current content.")

        # Scroll through the page to trigger lazy-loading / infinite scroll
        print("[*] Scrolling to load full content...")
        previous_height = 0
        for _ in range(20):
            current_height = page.evaluate("document.body.scrollHeight")
            if current_height == previous_height:
                break
            previous_height = current_height
            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            page.wait_for_timeout(900)
        page.evaluate("window.scrollTo(0, 0);")
        page.wait_for_timeout(500)

        # Determine output filename from page title
        page_title = page.title()
        base_filename = output_name or sanitize_filename(page_title)

        # Reader Mode: apply Readability.js to strip non-content and return clean HTML
        if reader_mode:
            print("[*] Reader Mode enabled — extracting main article content...")
            extracted_html = page.evaluate("""
                async () => {
                    let html = "";
                    try {
                        const mod = await import('https://cdn.skypack.dev/@mozilla/readability');
                        const article = new mod.Readability(document.cloneNode(true)).parse();
                        if (article && article.content) {
                            html = `<h1>${article.title || document.title}</h1>` + article.content;
                        }
                    } catch (e) {}

                    if (!html) {
                        // Fallback: strip obvious chrome from body
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll(
                            'nav, header, footer, button, script, style, aside, [role="navigation"]'
                        ).forEach(el => el.remove());
                        html = clone.innerHTML;
                    }
                    return html;
                }
            """)
            if not extracted_html or len(extracted_html.strip()) < 100:
                print("[!] Reader Mode extraction returned minimal content — falling back to full page.")
                extracted_html = page.content()
        else:
            extracted_html = page.content()

        browser.close()

    # Convert the HTML to clean prose text
    print("[*] Converting HTML to clean text...")
    clean_text = html_to_clean_text(extracted_html)

    if len(clean_text) < 200:
        print("[!] Warning: Extracted text is very short. The page may be JS-gated or require login.")

    # Save to raw_txt/
    output_path = os.path.join(RAW_TXT_DIR, f"{base_filename}.txt")
    # Avoid overwriting: append suffix if file exists
    counter = 1
    stem = output_path
    while os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        output_path = stem.replace(".txt", f"_{counter}.txt")
        counter += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"\n[+] Saved to: {output_path}  ({len(clean_text):,} characters)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape a webpage and save clean prose text to raw_txt/."
    )
    parser.add_argument("url", help="URL of the webpage to convert.")
    parser.add_argument(
        "--reader",
        action="store_true",
        help="Enable Reader Mode (recommended for Ellipsus, Tumblr, AO3, blog pages).",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Run browser visibly (non-headless) for debugging.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output filename (without .txt).",
    )

    args = parser.parse_args()
    scrape_url_to_txt(
        url=args.url,
        reader_mode=args.reader,
        headless=not args.visible,
        output_name=args.out,
    )
