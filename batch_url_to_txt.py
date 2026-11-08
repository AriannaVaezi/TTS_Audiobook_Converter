"""
batch_url_to_txt.py
====================
Reads a text file of URLs (one per line) and runs url_to_txt.py on each,
saving all results as .txt files into raw_txt/.

Usage:
  python batch_url_to_txt.py <links_file.txt> [options]

Options:
  --reader-all   Force Reader Mode for every link regardless of domain.
  --no-reader    Disable automatic Reader Mode for Ellipsus/Tumblr/AO3 links.

Links file format:
  - One URL per line
  - Lines starting with # are treated as comments and skipped
  - Blank lines are skipped

Domains that automatically use Reader Mode (unless --no-reader):
  ellipsus.com, tumblr.com, archiveofourown.org, ao3.org

Example:
  python batch_url_to_txt.py my_links.txt
  python batch_url_to_txt.py my_links.txt --reader-all
"""

import argparse
import os
import sys

RAW_TXT_DIR = "raw_txt"

# Domains where Reader Mode is applied automatically
READER_MODE_DOMAINS = [
    "ellipsus.com",
    "tumblr.com",
    "archiveofourown.org",
    "ao3.org",
]


def should_use_reader_mode(url: str) -> bool:
    """Returns True if the URL belongs to a domain that benefits from Reader Mode."""
    lower = url.lower()
    return any(domain in lower for domain in READER_MODE_DOMAINS)


def process_links_file(
    txt_file_path: str,
    force_reader: bool = False,
    no_reader: bool = False,
):
    if not os.path.exists(txt_file_path):
        print(f"[X] Error: The file '{txt_file_path}' was not found.")
        return

    os.makedirs(RAW_TXT_DIR, exist_ok=True)

    print(f"[*] Reading links from '{txt_file_path}'...")
    with open(txt_file_path, "r", encoding="utf-8") as f:
        links = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    if not links:
        print("[!] No valid links found in the file.")
        return

    print(f"[*] Found {len(links)} link(s) to process.\n")

    # Import the scraping function directly to avoid subprocess overhead
    try:
        from url_to_txt import scrape_url_to_txt
    except ImportError:
        print("[X] Cannot import url_to_txt.py. Make sure it is in the same directory.")
        sys.exit(1)

    success_count = 0
    failed_links = []

    for i, url in enumerate(links, 1):
        if force_reader:
            use_reader = True
        elif no_reader:
            use_reader = False
        else:
            use_reader = should_use_reader_mode(url)

        mode_label = "Reader Mode" if use_reader else "Standard"
        print(f"[{i}/{len(links)}] ({mode_label}) {url}")
        print("-" * 60)

        try:
            result_path = scrape_url_to_txt(url=url, reader_mode=use_reader)
            if result_path and os.path.exists(result_path):
                success_count += 1
                print(f"[+] Done: {result_path}\n")
            else:
                failed_links.append(url)
                print(f"[-] No output produced for: {url}\n")
        except Exception as e:
            failed_links.append(url)
            print(f"[-] Error processing {url}: {e}\n")

    # Summary
    print("=" * 60)
    print("Batch processing complete!")
    print(f"  Successful : {success_count}/{len(links)}")
    print(f"  Failed     : {len(failed_links)}/{len(links)}")
    print(f"  Output dir : {os.path.abspath(RAW_TXT_DIR)}")
    if failed_links:
        print("\nFailed URLs:")
        for url in failed_links:
            print(f"  - {url}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-convert a list of URLs to .txt files in raw_txt/."
    )
    parser.add_argument(
        "txt_file",
        help="Path to a text file containing one URL per line.",
    )
    parser.add_argument(
        "--reader-all",
        action="store_true",
        help="Force Reader Mode for every URL.",
    )
    parser.add_argument(
        "--no-reader",
        action="store_true",
        help="Disable automatic Reader Mode for all URLs.",
    )

    args = parser.parse_args()
    process_links_file(
        txt_file_path=args.txt_file,
        force_reader=args.reader_all,
        no_reader=args.no_reader,
    )
