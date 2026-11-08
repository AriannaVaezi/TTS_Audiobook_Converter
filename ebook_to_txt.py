"""
ebook_to_txt.py
================
Converts a PDF or EPUB file into a clean .txt file saved to raw_txt/.
Supports batch processing of entire folders.

Usage:
  python ebook_to_txt.py <file_or_folder> [options]

Options:
  --out NAME     Override the output filename (without .txt). Single file only.
  --no-clean     Skip paragraph cleaning (keeps raw extracted text as-is).

Supported formats:
  .epub   Extracted via ebooklib + BeautifulSoup HTML parsing
  .pdf    Extracted via PyMuPDF (pymupdf) — handles multi-column, headers, footers

Output:
  All .txt files are saved to raw_txt/

Examples:
  python ebook_to_txt.py my_book.epub
  python ebook_to_txt.py my_book.pdf
  python ebook_to_txt.py C:/Books/            (converts all PDFs & EPUBs in folder)
  python ebook_to_txt.py my_book.epub --out custom_name
"""

import argparse
import glob
import os
import re
import sys

RAW_TXT_DIR = "raw_txt"


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Removes characters unsafe for filenames and trims to a reasonable length."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] if cleaned else "ebook_output"


def clean_prose_text(text: str) -> str:
    """
    Post-processes raw extracted text into clean prose:
    - Collapses excessive blank lines
    - Normalises dashes, smart quotes, non-breaking spaces
    - Trims trailing whitespace per line
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ==================== EPUB EXTRACTION ====================

def epub_to_txt(epub_path: str, output_name: str = None, apply_clean: bool = True) -> str:
    """
    Extracts prose text from an EPUB file and saves it to raw_txt/.
    Returns the output path on success.
    """
    try:
        import ebooklib
        from ebooklib import epub as epublib
        from bs4 import BeautifulSoup
    except ImportError:
        print("[X] Missing dependencies. Run: pip install ebooklib beautifulsoup4")
        sys.exit(1)

    print(f"[*] Reading EPUB: {epub_path}")
    try:
        book = epublib.read_epub(epub_path, options={"ignore_ncx": True})
    except Exception as e:
        print(f"[X] Failed to read EPUB: {e}")
        return None

    # Determine output filename: prefer book title, fallback to file stem
    title = None
    try:
        title_meta = book.get_metadata("DC", "title")
        if title_meta:
            title = title_meta[0][0]
    except Exception:
        pass

    base_name = output_name or sanitize_filename(title or os.path.splitext(os.path.basename(epub_path))[0])

    # Extract text from each spine document item in reading order
    all_text_parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")

        # Remove nav, scripts, styles
        for tag in soup(["script", "style", "nav", "head"]):
            tag.decompose()

        # Convert <br> to newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Add double newlines around block elements for paragraph separation
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4",
                                   "h5", "h6", "li", "blockquote", "section"]):
            tag.insert_before("\n\n")
            tag.insert_after("\n\n")

        part_text = soup.get_text(separator="")
        part_text = re.sub(r"\n{3,}", "\n\n", part_text).strip()

        if len(part_text) > 50:
            all_text_parts.append(part_text)

    full_text = "\n\n".join(all_text_parts)

    if apply_clean:
        full_text = clean_prose_text(full_text)

    if len(full_text) < 100:
        print("[!] Warning: Extracted text is very short. The EPUB may be image-based or DRM-protected.")

    return _save_to_raw_txt(full_text, base_name)


# ==================== PDF EXTRACTION ====================

def pdf_to_txt(pdf_path: str, output_name: str = None, apply_clean: bool = True) -> str:
    """
    Extracts prose text from a PDF using PyMuPDF and saves it to raw_txt/.
    Returns the output path on success.
    """
    try:
        import pymupdf
    except ImportError:
        print("[X] PyMuPDF not found. Run: pip install pymupdf")
        sys.exit(1)

    print(f"[*] Reading PDF: {pdf_path}")
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"[X] Failed to open PDF: {e}")
        return None

    base_name = output_name or sanitize_filename(
        doc.metadata.get("title", "").strip() or os.path.splitext(os.path.basename(pdf_path))[0]
    )

    all_pages = []
    total = doc.page_count
    print(f"[*] Extracting text from {total} pages...")

    for i, page in enumerate(doc, 1):
        # Use "blocks" layout to preserve reading order better than raw text
        blocks = page.get_text("blocks", sort=True)
        page_lines = []
        for block in blocks:
            # block format: (x0, y0, x1, y1, text, block_no, block_type)
            if block[6] != 0:  # 0 = text block; skip images
                continue
            block_text = block[4].strip()
            if block_text:
                page_lines.append(block_text)

        page_text = "\n\n".join(page_lines)
        if page_text:
            all_pages.append(page_text)

        if i % 50 == 0 or i == total:
            print(f"    ... {i}/{total} pages processed")

    doc.close()

    full_text = "\n\n".join(all_pages)

    if apply_clean:
        full_text = clean_prose_text(full_text)

    if len(full_text) < 100:
        print("[!] Warning: Extracted text is very short. PDF may be scanned images (not OCR text).")

    return _save_to_raw_txt(full_text, base_name)


# ==================== SHARED HELPERS ====================

def _save_to_raw_txt(text: str, base_name: str) -> str:
    """Saves text to raw_txt/, avoiding overwrite of existing files."""
    os.makedirs(RAW_TXT_DIR, exist_ok=True)
    output_path = os.path.join(RAW_TXT_DIR, f"{base_name}.txt")

    # Avoid overwriting by adding a suffix counter
    counter = 1
    stem = output_path
    while os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        output_path = stem.replace(".txt", f"_{counter}.txt")
        counter += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    char_count = len(text)
    word_count = len(text.split())
    print(f"\n[+] Saved: {output_path}")
    print(f"    {char_count:,} characters | ~{word_count:,} words")
    return output_path


def convert_file(input_path: str, output_name: str = None, apply_clean: bool = True) -> str:
    """Dispatches a single file to the appropriate converter based on extension."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".epub":
        return epub_to_txt(input_path, output_name=output_name, apply_clean=apply_clean)
    elif ext == ".pdf":
        return pdf_to_txt(input_path, output_name=output_name, apply_clean=apply_clean)
    else:
        print(f"[!] Unsupported format: '{ext}'. Only .epub and .pdf are supported.")
        return None


def convert_folder(folder_path: str, apply_clean: bool = True):
    """Finds and converts all .epub and .pdf files in a folder."""
    patterns = ["*.epub", "*.pdf", "*.EPUB", "*.PDF"]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(folder_path, pattern)))

    # Deduplicate
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"[!] No .epub or .pdf files found in '{folder_path}'.")
        return

    print(f"[*] Found {len(all_files)} file(s) in '{folder_path}'.\n")
    success, failed = 0, []

    for i, path in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] {os.path.basename(path)}")
        print("-" * 50)
        try:
            result = convert_file(path, apply_clean=apply_clean)
            if result:
                success += 1
                print()
            else:
                failed.append(path)
        except Exception as e:
            print(f"[X] Error: {e}\n")
            failed.append(path)

    print("=" * 50)
    print("Batch conversion complete!")
    print(f"  Successful: {success}/{len(all_files)}")
    print(f"  Failed    : {len(failed)}/{len(all_files)}")
    print(f"  Output    : {os.path.abspath(RAW_TXT_DIR)}")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PDF or EPUB file (or folder) to a .txt file in raw_txt/."
    )
    parser.add_argument(
        "input",
        help="Path to a .epub or .pdf file, or a folder containing them.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output filename (without .txt). Single file only.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip post-processing text cleanup.",
    )

    args = parser.parse_args()
    apply_clean = not args.no_clean

    if os.path.isdir(args.input):
        convert_folder(args.input, apply_clean=apply_clean)
    elif os.path.isfile(args.input):
        convert_file(args.input, output_name=args.out, apply_clean=apply_clean)
    else:
        print(f"[X] Path not found: '{args.input}'")
        sys.exit(1)
