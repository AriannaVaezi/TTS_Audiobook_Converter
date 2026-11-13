"""
url_to_txt.py
=============
Scrapes a URL using Playwright (or reads a local .html/.txt file), extracts clean prose,
and saves the result as a .txt file (in raw_txt/), a .epub ebook (in raw_epub/), or both.

Usage:
  python url_to_txt.py <URL_OR_FILE> [options]

Options:
  --reader           Enable Reader Mode (strips nav/headers/sidebars via Readability.js).
                     Recommended for Ellipsus, Tumblr, AO3, and blog-style pages.
  --visible          Run the browser visibly (non-headless) for debugging.
  --epub             Save output as .epub instead of .txt (shorthand for --format epub).
  --format FORMAT    Output format: txt (default), epub, or both.
  --out NAME         Override the output filename or target path (.txt/.epub extension optional).
  --title TITLE      Override story/book title metadata for EPUB output.
  --author AUTHOR    Override author metadata for EPUB output.

Examples:
  python url_to_txt.py https://archiveofourown.org/works/12345 --reader
  python url_to_txt.py https://archiveofourown.org/works/12345 --reader --epub
  python url_to_txt.py https://ellipsus.com/story/abc --reader --format both
  python url_to_txt.py https://example.com/chapter1 --out my_story.epub
  python url_to_txt.py raw_txt/my_story.txt --epub --author "Author Name"
"""

import argparse
import html
import os
import re
import sys
import uuid
import json
import urllib.parse
import http.cookiejar

RAW_TXT_DIR = "raw_txt"
RAW_EPUB_DIR = "raw_epub"
SUPPORTED_OUTPUT_FORMATS = ("txt", "epub", "both")
DEFAULT_STORAGE_STATE_FILE = ".playwright_auth.json"

_NO_CONTENT_TAGS = (
    "script", "style", "nav", "header", "footer", "aside", "button",
    "form", "iframe", "noscript", "svg", "input", "textarea", "select",
    "link", "meta", "object", "embed", "canvas", "video", "audio",
)
_BLOCK_SPLIT_TRIGGER_TAGS = {
    "p", "div", "section", "article", "main", "blockquote", "pre",
    "ul", "ol", "table", "hr", "dl", "h1", "h2", "h3", "h4", "h5", "h6",
}
_WRAPPER_BLOCK_TAGS = {
    "html", "body", "div", "section", "article", "main",
    "figure", "figcaption", "center", "address",
}
_INLINE_ALLOWED_TAGS = {
    "em", "i", "strong", "b", "u", "s", "del", "ins", "mark",
    "small", "sup", "sub", "code", "kbd", "samp", "q", "cite",
    "abbr", "time", "span",
}
_SITE_TITLE_SUFFIX_RE = re.compile(
    r"\s*(?:\[Archive of Our Own\]|[-|\u2013\u2014]\s*(?:Archive of Our Own|AO3|Royal Road|FanFiction\.Net|FanFiction|Ellipsus|Tumblr|@?[\w.-]+\s+on\s+Tumblr))\s*$",
    re.IGNORECASE,
)
_GENERIC_TITLE_RE = re.compile(r"^post\s+by\s+@?\S+$", re.IGNORECASE)
_CHAPTER_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:chapter|ch\.?|part(?:\s+(?:\d+|[ivxlc]+|one|two|three|four|five|six|seven|eight|nine|ten)\b)|book\s+(?:\d+|[ivxlc]+)|act\s+(?:\d+|[ivxlc]+)|prologue|epilogue|interlude|afterword|preface|foreword)\b"
    r"|(?:[ivxlc]+|\d+)(?:\s*[:.\-\u2013\u2014]\s*\S.*|\s*$)"
    r")",
    re.IGNORECASE,
)
_XML_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_DEFAULT_EPUB_CSS = """\
body {
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.6;
  margin: 1em;
}
h1, h2, h3, h4 {
  line-height: 1.25;
  margin-top: 1.2em;
  margin-bottom: 0.6em;
}
p {
  margin: 0.5em 0;
}
blockquote {
  margin: 1em 1.5em;
  padding-left: 0.8em;
  border-left: 2px solid #888;
  font-style: italic;
}
pre {
  white-space: pre-wrap;
  font-family: monospace;
  margin: 0.8em 0;
}
hr {
  border: 0;
  border-top: 1px solid #999;
  margin: 1.5em auto;
  width: 40%;
}
"""


def sanitize_filename(title: str) -> str:
    """Removes characters unsafe for filenames and trims length."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:120] if cleaned else "webpage_output"


def clean_story_title(raw_title: str) -> str:
    """Strips common web-platform suffixes from scraped page titles."""
    if not raw_title:
        return ""
    stripped = _SITE_TITLE_SUFFIX_RE.sub("", raw_title).strip()
    return stripped or raw_title.strip()


def _strip_xml_control_chars(text: str) -> str:
    """Removes XML-invalid ASCII control characters and normalizes non-breaking spaces."""
    if not text:
        return ""
    return _XML_CONTROL_CHARS_RE.sub("", text).replace("\u00a0", " ")


def _normalize_ws(text: str) -> str:
    """Collapses internal whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", _strip_xml_control_chars(text)).strip()


def _resolve_output_path(
    output_name: str,
    fallback_stem: str,
    ext: str,
    default_dir: str,
) -> str:
    """Resolves a collision-free output path for .txt or .epub exports."""
    if not ext.startswith("."):
        ext = f".{ext}"

    if output_name:
        candidate_arg = output_name.strip()
        target_dir = os.path.dirname(candidate_arg) or default_dir
        raw_stem = os.path.basename(candidate_arg)
        if raw_stem.lower().endswith((".txt", ".epub")):
            raw_stem = os.path.splitext(raw_stem)[0]
        base_stem = raw_stem or sanitize_filename(fallback_stem)
    else:
        target_dir = default_dir
        base_stem = sanitize_filename(fallback_stem)

    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, f"{base_stem}{ext}")

    counter = 1
    stem, file_ext = os.path.splitext(output_path)
    while os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        output_path = f"{stem}_{counter}{file_ext}"
        counter += 1
    return output_path


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

    # Convert <br> -> newline markers before stripping
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


def extract_metadata_from_html(
    html_content: str,
    fallback_title: str = None,
    fallback_author: str = None,
) -> tuple[str, str]:
    """Extracts clean (title, author) metadata from HTML with caller fallbacks."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_strip_xml_control_chars(html_content), "html.parser")

    title_candidates = []
    for selector in ("#workskin .preface h2.title", "h1.title", "h1.entry-title"):
        el = soup.select_one(selector)
        if el:
            txt = _normalize_ws(el.get_text(" ", strip=True))
            if txt:
                title_candidates.append(txt)

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title_candidates.append(og_title["content"])

    if fallback_title:
        title_candidates.append(fallback_title)

    for selector in ("h1", "title"):
        el = soup.select_one(selector)
        if el:
            txt = _normalize_ws(el.get_text(" ", strip=True))
            if txt:
                title_candidates.append(txt)

    resolved_title = ""
    for cand in title_candidates:
        cleaned = clean_story_title(cand)
        if cleaned and not _GENERIC_TITLE_RE.match(cleaned):
            resolved_title = cleaned
            break
    if not resolved_title:
        resolved_title = "Untitled Story"

    resolved_author = _normalize_ws(fallback_author or "")
    if not resolved_author:
        meta_author = soup.find("meta", attrs={"name": "author"})
        if meta_author and meta_author.get("content"):
            resolved_author = _normalize_ws(meta_author["content"])
    if not resolved_author:
        for selector in ("#workskin .preface a[rel='author']", "a[rel='author']", ".byline a", ".byline"):
            el = soup.select_one(selector)
            if el:
                cand = _normalize_ws(el.get_text(" ", strip=True))
                cand = re.sub(r"^by\s+", "", cand, flags=re.IGNORECASE).strip()
                if cand and len(cand) <= 80:
                    resolved_author = cand
                    break

    return resolved_title, resolved_author


def _plain_text_from_xhtml(fragment: str) -> str:
    """Returns plain normalized text content of an XHTML snippet."""
    if not fragment:
        return ""
    stripped = re.sub(r"<[^>]+>", " ", fragment)
    return _normalize_ws(html.unescape(stripped))


def _serialize_inline_nodes(nodes) -> str:
    """Recursively serializes BeautifulSoup inline nodes to clean, escaped XHTML."""
    from bs4 import Comment, Doctype, NavigableString, ProcessingInstruction, Tag

    parts = []
    for node in nodes:
        if isinstance(node, (Comment, Doctype, ProcessingInstruction)):
            continue
        if isinstance(node, NavigableString):
            parts.append(html.escape(_strip_xml_control_chars(str(node)), quote=False))
            continue
        if not isinstance(node, Tag):
            continue

        tag_name = node.name.lower()
        if tag_name == "br":
            parts.append("<br/>")
        elif tag_name == "img":
            alt = _normalize_ws(node.get("alt") or "")
            if alt and len(alt) <= 120 and alt.lower() not in {
                "image", "img", "icon", "logo", "avatar", "photo",
            }:
                parts.append(f"<span>[{html.escape(alt, quote=False)}]</span>")
        elif tag_name == "a":
            inner = _serialize_inline_nodes(node.children)
            if not _plain_text_from_xhtml(inner):
                continue
            href = _strip_xml_control_chars((node.get("href") or "").strip())
            if href.lower().startswith(("http://", "https://", "mailto:")):
                parts.append(f'<a href="{html.escape(href, quote=True)}">{inner}</a>')
            else:
                parts.append(inner)
        elif tag_name in _INLINE_ALLOWED_TAGS:
            inner = _serialize_inline_nodes(node.children)
            if inner:
                parts.append(f"<{tag_name}>{inner}</{tag_name}>")
        else:
            parts.append(_serialize_inline_nodes(node.children))
    return "".join(parts)


def _split_inline_xhtml_on_double_br(inline_xhtml: str) -> list[str]:
    """Splits inline XHTML segments separated by two or more <br/> tags into paragraphs."""
    segments = re.split(r"(?:\s*<br/>\s*){2,}", inline_xhtml)
    cleaned_segments = []
    for seg in segments:
        seg = re.sub(r"^(?:\s*<br/>\s*)+|(?:\s*<br/>\s*)+$", "", seg).strip()
        if _plain_text_from_xhtml(seg):
            cleaned_segments.append(seg)
    return cleaned_segments


def _flush_inline_paragraphs(inline_buf: list, blocks_out: list[tuple[str, str, str]]) -> None:
    """Converts buffered inline nodes into one or more clean <p> XHTML blocks."""
    if not inline_buf:
        return
    raw_inline = _serialize_inline_nodes(inline_buf)
    inline_buf.clear()
    for para_inner in _split_inline_xhtml_on_double_br(raw_inline):
        blocks_out.append(("p", "", f"<p>{para_inner}</p>"))


def _serialize_list_tag(tag) -> str:
    """Serializes <ul> or <ol> lists into clean XHTML."""
    tag_name = tag.name.lower()
    items_out = []
    for child in tag.find_all("li", recursive=False):
        li_blocks = []
        inline_buf = []
        for sub in child.children:
            sub_name = getattr(sub, "name", None)
            if sub_name and sub_name.lower() in ("ul", "ol"):
                if inline_buf:
                    text_part = _serialize_inline_nodes(inline_buf).strip()
                    inline_buf.clear()
                    if _plain_text_from_xhtml(text_part):
                        li_blocks.append(text_part)
                nested = _serialize_list_tag(sub)
                if nested:
                    li_blocks.append(nested)
            elif sub_name and sub_name.lower() in ("p", "div"):
                if inline_buf:
                    text_part = _serialize_inline_nodes(inline_buf).strip()
                    inline_buf.clear()
                    if _plain_text_from_xhtml(text_part):
                        li_blocks.append(f"<p>{text_part}</p>")
                p_inner = _serialize_inline_nodes(sub.children).strip()
                if _plain_text_from_xhtml(p_inner):
                    li_blocks.append(f"<p>{p_inner}</p>")
            else:
                inline_buf.append(sub)
        if inline_buf:
            text_part = _serialize_inline_nodes(inline_buf).strip()
            if _plain_text_from_xhtml(text_part):
                li_blocks.append(text_part)
        if li_blocks:
            items_out.append(f"<li>{''.join(li_blocks)}</li>")
    return f"<{tag_name}>{''.join(items_out)}</{tag_name}>" if items_out else ""


def _serialize_table_tag(tag) -> str:
    """Serializes an HTML table cleanly, preserving spans while stripping presentation attrs."""
    rows_out = []
    for tr in tag.find_all("tr"):
        cells_out = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            c_name = cell.name.lower()
            inner = _serialize_inline_nodes(cell.children).strip()
            attrs = []
            for span_attr in ("colspan", "rowspan"):
                val = str(cell.get(span_attr, "")).strip()
                if val.isdigit() and int(val) > 1:
                    attrs.append(f'{span_attr}="{val}"')
            attr_str = f" {' '.join(attrs)}" if attrs else ""
            cells_out.append(f"<{c_name}{attr_str}>{inner}</{c_name}>")
        if cells_out:
            rows_out.append(f"<tr>{''.join(cells_out)}</tr>")
    return f"<table><tbody>{''.join(rows_out)}</tbody></table>" if rows_out else ""


def _serialize_atomic_block(tag) -> str:
    """Serializes blockquote, pre, ul, ol, dl, or table tags into valid XHTML."""
    tag_name = tag.name.lower()
    if tag_name in ("ul", "ol"):
        return _serialize_list_tag(tag)
    if tag_name == "table":
        return _serialize_table_tag(tag)
    if tag_name == "dl":
        parts = []
        for child in tag.find_all(["dt", "dd"], recursive=False):
            inner = _serialize_inline_nodes(child.children).strip()
            if _plain_text_from_xhtml(inner):
                parts.append(f"<{child.name.lower()}>{inner}</{child.name.lower()}>")
        return f"<dl>{''.join(parts)}</dl>" if parts else ""
    if tag_name == "pre":
        pre_text = _strip_xml_control_chars(tag.get_text())
        if not pre_text.strip():
            return ""
        return f"<pre>{html.escape(pre_text, quote=False)}</pre>"
    if tag_name == "blockquote":
        inner_blocks = []
        _nodes_to_clean_blocks(tag, inner_blocks)
        inner_html = "".join(b[2] for b in inner_blocks)
        return f"<blockquote>{inner_html}</blockquote>" if _plain_text_from_xhtml(inner_html) else ""
    return ""


def _nodes_to_clean_blocks(container, blocks_out: list[tuple[str, str, str]]) -> None:
    """Walks a BeautifulSoup tree and collects sanitized atomic XHTML blocks in order."""
    from bs4 import Comment, Doctype, NavigableString, ProcessingInstruction, Tag

    inline_buf = []
    for child in container.children:
        if isinstance(child, (Comment, Doctype, ProcessingInstruction)):
            continue
        if isinstance(child, NavigableString):
            inline_buf.append(child)
            continue
        if not isinstance(child, Tag):
            continue

        tag_name = child.name.lower()
        if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            _flush_inline_paragraphs(inline_buf, blocks_out)
            inner = _serialize_inline_nodes(child.children).strip()
            heading_text = _plain_text_from_xhtml(inner)
            if heading_text:
                blocks_out.append((tag_name, heading_text, f"<{tag_name}>{inner}</{tag_name}>"))
        elif tag_name == "hr":
            _flush_inline_paragraphs(inline_buf, blocks_out)
            blocks_out.append(("hr", "", "<hr/>"))
        elif tag_name == "p":
            if child.find(_BLOCK_SPLIT_TRIGGER_TAGS):
                _flush_inline_paragraphs(inline_buf, blocks_out)
                _nodes_to_clean_blocks(child, blocks_out)
            else:
                _flush_inline_paragraphs(inline_buf, blocks_out)
                inner = _serialize_inline_nodes(child.children)
                for seg in _split_inline_xhtml_on_double_br(inner):
                    blocks_out.append(("p", "", f"<p>{seg}</p>"))
        elif tag_name in ("blockquote", "pre", "ul", "ol", "table", "dl"):
            _flush_inline_paragraphs(inline_buf, blocks_out)
            xhtml_str = _serialize_atomic_block(child)
            if xhtml_str:
                blocks_out.append((tag_name, "", xhtml_str))
        elif tag_name in _WRAPPER_BLOCK_TAGS or child.find(_BLOCK_SPLIT_TRIGGER_TAGS):
            _flush_inline_paragraphs(inline_buf, blocks_out)
            _nodes_to_clean_blocks(child, blocks_out)
            _flush_inline_paragraphs(inline_buf, blocks_out)
        else:
            inline_buf.append(child)

    _flush_inline_paragraphs(inline_buf, blocks_out)


def _blocks_to_body_html(blocks: list[tuple[str, str, str]]) -> str:
    """Joins sanitized XHTML blocks into body markup."""
    return "\n".join(block[2] for block in blocks)


def _split_explicit_chapter_containers(root) -> list[tuple[str, str]]:
    """
    Splits multi-chapter HTML pages using explicit #chapter-N containers (e.g. AO3)
    even when chapter headings do not begin with the word 'Chapter'.
    Returns [] if fewer than 2 valid containers are found.
    """
    candidates = [
        el for el in root.find_all(id=re.compile(r"^chapter[-_]?\d+$", re.IGNORECASE))
        if el.find(["p", "div", "blockquote", "h1", "h2", "h3", "h4"])
    ]
    if len(candidates) < 2:
        return []

    candidate_set = set(id(el) for el in candidates)
    top_containers = [
        el for el in candidates
        if not any(id(parent) in candidate_set for parent in el.parents)
    ]
    if len(top_containers) < 2:
        return []

    chapters: list[tuple[str, str]] = []
    for idx, container in enumerate(top_containers, 1):
        blocks: list[tuple[str, str, str]] = []
        _nodes_to_clean_blocks(container, blocks)
        has_body = any(b[0] not in ("h1", "h2", "h3", "h4", "h5", "h6") for b in blocks)
        if not blocks or not has_body:
            continue
        heading = next((b[1] for b in blocks if b[0] in ("h1", "h2", "h3", "h4")), f"Chapter {idx}")
        body_html = _blocks_to_body_html(blocks)
        if blocks[0][0] not in ("h1", "h2", "h3", "h4"):
            body_html = f"<h2>{html.escape(heading, quote=False)}</h2>\n{body_html}"
        chapters.append((heading, body_html))

    return chapters if len(chapters) >= 2 else []


def html_to_epub_chapters(
    html_content: str,
    book_title: str = "Untitled Story",
    raw_page_title: str = None,
) -> list[tuple[str, str]]:
    """
    Cleans raw HTML into valid XHTML blocks and splits multi-chapter stories into
    [(chapter_title, body_xhtml), ...] pairs suitable for EPUB assembly.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_strip_xml_control_chars(html_content), "html.parser")
    for tag in soup(list(_NO_CONTENT_TAGS)):
        tag.decompose()
    for nav_el in soup.select('[role="navigation"], p.skip, .landmark'):
        nav_el.decompose()

    root = soup.body or soup

    # 1. Try explicit #chapter-N containers when multiple chapters exist
    container_chapters = _split_explicit_chapter_containers(root)
    if container_chapters:
        return container_chapters

    # 2. Otherwise linearize into clean blocks and split on chapter-style headings
    blocks: list[tuple[str, str, str]] = []
    _nodes_to_clean_blocks(root, blocks)

    if not blocks:
        fallback_txt = html_to_clean_text(html_content)
        return text_to_epub_chapters(fallback_txt, book_title=book_title)

    # Replace synthetic Reader Mode raw document title heading with cleaned story title
    if (
        raw_page_title
        and blocks[0][0] == "h1"
        and _normalize_ws(blocks[0][1]) == _normalize_ws(raw_page_title)
        and clean_story_title(raw_page_title)
    ):
        cleaned_h1 = book_title or clean_story_title(raw_page_title)
        blocks[0] = ("h1", cleaned_h1, f"<h1>{html.escape(cleaned_h1, quote=False)}</h1>")

    boundary_indices = [
        idx
        for idx, (tag_name, heading_text, _) in enumerate(blocks)
        if tag_name in ("h1", "h2", "h3", "h4")
        and len(heading_text) <= 140
        and _CHAPTER_HEADING_RE.match(heading_text)
    ]

    chapters: list[tuple[str, str]] = []
    if len(boundary_indices) >= 2:
        preamble = blocks[: boundary_indices[0]]
        preamble_prose_len = sum(
            len(_plain_text_from_xhtml(b[2]))
            for b in preamble
            if b[0] not in ("h1", "h2", "h3", "h4", "h5", "h6", "hr")
        )
        if preamble_prose_len >= 60:
            preface_title = next(
                (
                    b[1] for b in preamble
                    if b[0] in ("h1", "h2", "h3")
                    and _normalize_ws(b[1]).lower() != _normalize_ws(book_title).lower()
                ),
                "Preface",
            )
            chapters.append((preface_title, _blocks_to_body_html(preamble)))

        for pos, start_idx in enumerate(boundary_indices):
            end_idx = boundary_indices[pos + 1] if pos + 1 < len(boundary_indices) else len(blocks)
            section_blocks = blocks[start_idx:end_idx]
            has_body = any(b[0] not in ("h1", "h2", "h3", "h4", "h5", "h6") for b in section_blocks)
            if not section_blocks or not has_body:
                continue
            chapters.append((section_blocks[0][1], _blocks_to_body_html(section_blocks)))

    if not chapters:
        first_is_title = blocks[0][0] in ("h1", "h2") and (
            _normalize_ws(blocks[0][1]).lower() == _normalize_ws(book_title).lower()
        )
        body_html = _blocks_to_body_html(blocks)
        if not first_is_title and blocks[0][0] != "h1":
            body_html = f"<h1>{html.escape(book_title, quote=False)}</h1>\n{body_html}"
        chapters.append((book_title, body_html))

    return chapters


def text_to_epub_chapters(
    text: str,
    book_title: str = "Untitled Story",
) -> list[tuple[str, str]]:
    """
    Converts plain prose text (including optional === CHAPTER === banners or Chapter headings)
    into [(chapter_title, body_xhtml), ...] pairs for EPUB output.
    """
    normalized = _strip_xml_control_chars(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return [(book_title, f"<h1>{html.escape(book_title, quote=False)}</h1><p></p>")]

    paragraphs = re.split(r"\n{2,}", normalized)
    raw_items: list[tuple[bool, str, str]] = []  # (is_heading, heading_text, xhtml)

    for para in paragraphs:
        chunk = para.strip()
        if not chunk:
            continue
        banner_match = re.match(r"^={3,}\s*(.+?)\s*={3,}$", chunk)
        if banner_match:
            title_txt = _normalize_ws(banner_match.group(1))
            raw_items.append((True, title_txt, f"<h2>{html.escape(title_txt, quote=False)}</h2>"))
            continue
        if re.fullmatch(r"(?:\*\s*\*\s*\*|-{3,}|_{3,})", chunk):
            raw_items.append((False, "", "<hr/>"))
            continue
        if "\n" not in chunk and len(chunk) <= 140 and _CHAPTER_HEADING_RE.match(chunk):
            heading_txt = _normalize_ws(chunk)
            raw_items.append((True, heading_txt, f"<h2>{html.escape(heading_txt, quote=False)}</h2>"))
            continue

        escaped_lines = [html.escape(line.rstrip(), quote=False) for line in chunk.splitlines()]
        raw_items.append((False, "", f"<p>{'<br/>'.join(escaped_lines)}</p>"))

    boundary_indices = [idx for idx, item in enumerate(raw_items) if item[0]]
    chapters: list[tuple[str, str]] = []

    if len(boundary_indices) >= 2:
        preamble = raw_items[: boundary_indices[0]]
        if any(not item[0] and item[2] != "<hr/>" for item in preamble):
            chapters.append(("Preface", "\n".join(item[2] for item in preamble)))
        for pos, start_idx in enumerate(boundary_indices):
            end_idx = boundary_indices[pos + 1] if pos + 1 < len(boundary_indices) else len(raw_items)
            sec = raw_items[start_idx:end_idx]
            if any(not item[0] for item in sec):
                chapters.append((sec[0][1], "\n".join(item[2] for item in sec)))

    if not chapters:
        body_html = "\n".join(item[2] for item in raw_items)
        if not raw_items or not raw_items[0][0]:
            body_html = f"<h1>{html.escape(book_title, quote=False)}</h1>\n{body_html}"
        chapters.append((book_title, body_html))

    return chapters


def write_epub_from_chapters(
    chapters: list[tuple[str, str]],
    output_path: str,
    title: str,
    author: str = None,
    source_url: str = None,
    language: str = "en",
) -> str:
    """Packages cleaned XHTML chapters into a valid EPUB archive via ebooklib."""
    try:
        from ebooklib import epub
    except ImportError:
        print("[X] Missing dependency 'ebooklib'. Run: pip install ebooklib")
        sys.exit(1)

    safe_title = _normalize_ws(title) or "Untitled Story"
    safe_author = _normalize_ws(author or "")

    book = epub.EpubBook()
    book.set_identifier(f"url-to-txt-{uuid.uuid4()}")
    book.set_title(safe_title)
    book.set_language(language)
    if safe_author:
        book.add_author(safe_author)
    if source_url:
        book.add_metadata("DC", "source", source_url)

    style_item = epub.EpubItem(
        uid="style_default",
        file_name="style/default.css",
        media_type="text/css",
        content=_DEFAULT_EPUB_CSS.encode("utf-8"),
    )
    book.add_item(style_item)

    epub_chapters = []
    for idx, (chap_title, body_xhtml) in enumerate(chapters, 1):
        clean_chap_title = _normalize_ws(chap_title) or f"Chapter {idx}"
        item = epub.EpubHtml(
            title=clean_chap_title,
            file_name=f"chap_{idx:03d}.xhtml",
            lang=language,
        )
        escaped_chap_title = html.escape(clean_chap_title, quote=False)
        item.content = (
            f"<html><head><title>{escaped_chap_title}</title></head>"
            f"<body>{body_xhtml}</body></html>"
        )
        item.add_item(style_item)
        book.add_item(item)
        epub_chapters.append(item)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = (["nav"] if len(epub_chapters) > 1 else []) + epub_chapters

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    epub.write_epub(output_path, book)
    return output_path


def html_to_epub(
    html_content: str,
    output_path: str,
    title: str = None,
    author: str = None,
    source_url: str = None,
    raw_page_title: str = None,
) -> str:
    """Converts scraped/local HTML directly into an EPUB file at output_path."""
    detected_title, detected_author = extract_metadata_from_html(
        html_content,
        fallback_title=raw_page_title,
        fallback_author=author,
    )
    resolved_title = _normalize_ws(title) if title else detected_title
    resolved_author = _normalize_ws(author) if author else detected_author
    chapters = html_to_epub_chapters(
        html_content,
        book_title=resolved_title,
        raw_page_title=raw_page_title,
    )
    return write_epub_from_chapters(
        chapters=chapters,
        output_path=output_path,
        title=resolved_title,
        author=resolved_author,
        source_url=source_url,
    )


def text_to_epub(
    text: str,
    output_path: str,
    title: str = None,
    author: str = None,
    source_url: str = None,
) -> str:
    """Converts plain text directly into an EPUB file at output_path."""
    resolved_title = _normalize_ws(title) if title else "Untitled Story"
    chapters = text_to_epub_chapters(text, book_title=resolved_title)
    return write_epub_from_chapters(
        chapters=chapters,
        output_path=output_path,
        title=resolved_title,
        author=author,
        source_url=source_url,
    )


def _expand_tumblr_keep_reading(page, url: str, max_clicks: int = 5) -> int:
    """
    Clicks Tumblr 'Keep reading' / '.read_more' expansion controls inside the
    primary post container before scrolling and Reader Mode extraction.
    Returns the number of clicks performed.
    """
    is_tumblr_url = "tumblr.com" in (url or "").lower()
    has_tumblr_markers = page.evaluate("""
        () => Boolean(
            document.querySelector('a.read_more, .tmblr-truncated, article[data-id], [data-testid="post-content"]')
            || window.location.hostname.toLowerCase().includes('tumblr.com')
        )
    """)
    if not (is_tumblr_url or has_tumblr_markers):
        return 0

    clicks_done = 0
    for _ in range(max_clicks):
        found = page.evaluate("""
            () => {
                document.querySelectorAll('[data-keep-reading-target]').forEach(
                    el => el.removeAttribute('data-keep-reading-target')
                );
                const root = document.querySelector(
                    'main article, [role="main"] article, article[data-id], article, .post, #posts .post'
                ) || document.body;

                const candidates = root.querySelectorAll('button, a.read_more, [role="button"], a');
                for (const el of candidates) {
                    if (el.closest('nav, footer, aside, [role="navigation"], [aria-label*="related" i]')) {
                        continue;
                    }
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) {
                        continue;
                    }
                    const text = (el.innerText || el.textContent || '')
                        .replace(/\\s+/g, ' ')
                        .trim()
                        .toLowerCase();
                    const isReadMoreLink = el.matches('a.read_more') && /^(?:keep\\s+reading|read\\s+more)\\b/.test(text);
                    const isKeepReadingBtn = /^keep\\s+reading(?:\\s*(?:\\([^)]{1,30}\\)|[\\u2192\\u2026.!>»]+))?$/.test(text);
                    if (isReadMoreLink || isKeepReadingBtn) {
                        el.setAttribute('data-keep-reading-target', '1');
                        return true;
                    }
                }
                return false;
            }
        """)
        if not found:
            break

        target = page.locator('[data-keep-reading-target="1"]').first
        clicks_done += 1
        print(f"[*] Clicking Tumblr 'Keep reading' button ({clicks_done})...")
        try:
            target.scroll_into_view_if_needed(timeout=3000)
            target.click(timeout=5000)
        except Exception:
            page.evaluate("""
                () => {
                    const el = document.querySelector('[data-keep-reading-target="1"]');
                    if (el) el.click();
                }
            """)

        page.wait_for_timeout(1200)
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

    page.evaluate("""
        () => document.querySelectorAll('[data-keep-reading-target]').forEach(
            el => el.removeAttribute('data-keep-reading-target')
        )
    """)
    return clicks_done


def _normalize_tumblr_url(url: str) -> tuple[str, str | None, str | None, str | None]:
    """
    Normalizes Tumblr URLs and identifies custom subdomain blogs.
    Returns (canonical_url, blog_name, post_id, archive_fallback_url).
    Dashboard/login-walled URLs like `https://www.tumblr.com/<blog>/<post_id>/<slug>`
    often redirect to a login wall on `www.tumblr.com`, whereas the canonical blog
    subdomain `https://<blog>.tumblr.com/post/<post_id>/<slug>` or `/mobile` / `/embed`
    can be publicly accessible without logging in.
    """
    if not url:
        return url, None, None, None
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    reserved_paths = {
        "login", "register", "dashboard", "settings", "explore", "search",
        "tagged", "likes", "following", "blog", "post", "reblog", "inbox",
        "messages", "activity", "customize", "communities", "privacy", "policy",
    }

    # Case 1: https://www.tumblr.com/<blog>/<post_id>[/<slug>] or /blog/view/<blog>/<post_id>
    if host in ("www.tumblr.com", "tumblr.com"):
        m_view = re.match(r"^/blog/view/([a-zA-Z0-9_-]+)(?:/(\d+)(?:/([^/?#]*))?)?", path)
        if m_view:
            blog, post_id, slug = m_view.group(1), m_view.group(2), m_view.group(3)
            if post_id:
                slug_part = f"/{slug}" if slug else ""
                canon = f"https://{blog}.tumblr.com/post/{post_id}{slug_part}"
                fallback = f"https://{blog}.tumblr.com/post/{post_id}/mobile"
                return canon, blog, post_id, fallback
            return f"https://{blog}.tumblr.com", blog, None, None

        m_dash = re.match(r"^/([a-zA-Z0-9_-]+)/(\d+)(?:/([^/?#]*))?", path)
        if m_dash and m_dash.group(1).lower() not in reserved_paths:
            blog, post_id, slug = m_dash.group(1), m_dash.group(2), m_dash.group(3)
            slug_part = f"/{slug}" if slug else ""
            canon = f"https://{blog}.tumblr.com/post/{post_id}{slug_part}"
            fallback = f"https://{blog}.tumblr.com/post/{post_id}/mobile"
            return canon, blog, post_id, fallback

    # Case 2: https://<blog>.tumblr.com/post/<post_id>[/<slug>] or https://<blog>.tumblr.com/<post_id>
    m_sub = re.match(r"^([a-zA-Z0-9_-]+)\.tumblr\.com$", host)
    if m_sub and m_sub.group(1).lower() != "www":
        blog = m_sub.group(1)
        m_post = re.match(r"^/(?:post/)?(\d+)(?:/([^/?#]*))?", path)
        if m_post:
            post_id, slug = m_post.group(1), m_post.group(2)
            slug_part = f"/{slug}" if slug and slug.lower() != "mobile" else ""
            canon = f"https://{blog}.tumblr.com/post/{post_id}{slug_part}"
            fallback = f"https://{blog}.tumblr.com/post/{post_id}/mobile"
            return canon, blog, post_id, fallback

    return url, None, None, None


def _load_cookies_from_file(cookie_path: str) -> list[dict]:
    """
    Loads cookies from either a Netscape cookies.txt file or a JSON cookie export
    (e.g. from Cookie-Editor / EditThisCookie / Playwright storage state).
    """
    if not cookie_path or not os.path.isfile(cookie_path):
        return []

    with open(cookie_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # JSON format (list of cookie dicts or Playwright storage_state dict)
    if raw.startswith(("{", "[")):
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "cookies" in data:
                data = data["cookies"]
            if isinstance(data, list):
                out = []
                for c in data:
                    if not isinstance(c, dict) or not c.get("name"):
                        continue
                    cookie_entry = {
                        "name": str(c["name"]),
                        "value": str(c.get("value", "")),
                        "domain": str(c.get("domain", ".tumblr.com")),
                        "path": str(c.get("path", "/")),
                    }
                    if "secure" in c:
                        cookie_entry["secure"] = bool(c["secure"])
                    if "httpOnly" in c:
                        cookie_entry["httpOnly"] = bool(c["httpOnly"])
                    out.append(cookie_entry)
                return out
        except Exception as e:
            print(f"[!] Warning: Could not parse JSON cookies from '{cookie_path}': {e}")
            return []

    # Netscape cookies.txt format
    cookies = []
    try:
        cj = http.cookiejar.MozillaCookieJar()
        cj.load(cookie_path, ignore_discard=True, ignore_expires=True)
        for c in cj:
            cookies.append({
                "name": c.name,
                "value": c.value or "",
                "domain": c.domain or ".tumblr.com",
                "path": c.path or "/",
                "secure": bool(c.secure),
            })
    except Exception:
        # Fallback manual tab-separated line parser for Netscape format
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies.append({
                    "name": parts[5],
                    "value": parts[6],
                    "domain": parts[0],
                    "path": parts[2] or "/",
                    "secure": parts[3].upper() == "TRUE",
                })
    return cookies


def _dismiss_tumblr_walls_and_check_login(page) -> bool:
    """
    Dismisses soft Tumblr login/consent/safe-mode overlays and returns True
    if the page is still hard-blocked by a Tumblr login wall.
    """
    try:
        state = page.evaluate("""
            () => {
                // 1. Remove fixed login / glass / signup banners & modal backdrops
                const overlaySelectors = [
                    '[data-testid="glass-container"]',
                    '[aria-label*="log in" i][role="dialog"]',
                    '[aria-label*="sign up" i][role="dialog"]',
                    '#glass-container',
                    '.standalone-ad-container',
                    '[class*="loginWall" i]',
                    '[class*="LoginWall" i]',
                    '[class*="signupWall" i]',
                ];
                for (const sel of overlaySelectors) {
                    document.querySelectorAll(sel).forEach(el => {
                        // Only remove if it doesn't contain the primary article
                        if (!el.querySelector('article')) {
                            el.remove();
                        }
                    });
                }
                if (document.body) {
                    document.body.style.overflow = 'auto';
                }
                if (document.documentElement) {
                    document.documentElement.style.overflow = 'auto';
                }

                // 2. Click "View post" / "Yes, I am over 18" / "Proceed" / mature content buttons if present
                const buttons = document.querySelectorAll('button, a[role="button"]');
                for (const btn of buttons) {
                    const txt = (btn.innerText || btn.textContent || '').trim().toLowerCase();
                    if (
                        txt === 'view post' ||
                        txt === 'show post' ||
                        txt === 'proceed' ||
                        txt === 'yes, i am over 18' ||
                        txt === 'dismiss' ||
                        txt === 'maybe later' ||
                        txt === 'not now'
                    ) {
                        try { btn.click(); } catch (e) {}
                    }
                }

                // 3. Check if we are redirected to /login or /safe-mode or have no post content
                const href = (window.location.href || '').toLowerCase();
                const bodyText = (document.body ? document.body.innerText : '').trim();
                const hasArticle = Boolean(
                    document.querySelector('article, .post, #posts .post, .body-text, .post-body')
                    && bodyText.length > 350
                );
                const isLoginUrl = (
                    href.includes('tumblr.com/login') ||
                    href.includes('tumblr.com/register') ||
                    href.includes('tumblr.com/safe-mode') ||
                    href.includes('tumblr.com/privacy/consent')
                );
                const isLoginText = (
                    !hasArticle && (
                        /log\\s+in\\s+to\\s+tumblr/i.test(bodyText) ||
                        /sign\\s+up\\s+or\\s+log\\s+in/i.test(bodyText) ||
                        /this\\s+blog\\s+may\\s+contain\\s+sensitive\\s+media/i.test(bodyText) ||
                        /only\\s+logged-in\\s+users\\s+can\\s+view/i.test(bodyText)
                    )
                );
                return Boolean(isLoginUrl || isLoginText);
            }
        """)
        return bool(state)
    except Exception:
        return False


def _fetch_url_html_with_playwright(
    url: str,
    reader_mode: bool = False,
    headless: bool = True,
    cookies_file: str = None,
    storage_state_file: str = None,
    interactive_login: bool = False,
) -> tuple[str, str, str, str]:
    """
    Loads a URL with Playwright Chromium and returns
    (extracted_html, raw_page_title, meta_title, meta_author).
    Supports Tumblr URL canonicalization, login wall dismissal, saved storage state,
    cookie import, and interactive login saving.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[X] Playwright is not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    canon_url, tumblr_blog, tumblr_post_id, tumblr_mobile_url = _normalize_tumblr_url(url)
    target_url = canon_url or url
    if target_url != url:
        print(f"[*] Normalized Tumblr URL to blog subdomain: '{target_url}'")

    resolved_state_file = storage_state_file or DEFAULT_STORAGE_STATE_FILE
    if interactive_login:
        headless = False

    print(f"[*] Opening '{target_url}'...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        context_kwargs = {
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "viewport": {"width": 1280, "height": 800},
        }
        if resolved_state_file and os.path.isfile(resolved_state_file):
            print(f"[*] Loading saved browser session from '{resolved_state_file}'...")
            context_kwargs["storage_state"] = resolved_state_file

        context = browser.new_context(**context_kwargs)

        if cookies_file:
            loaded_cookies = _load_cookies_from_file(cookies_file)
            if loaded_cookies:
                try:
                    context.add_cookies(loaded_cookies)
                    print(f"[*] Loaded {len(loaded_cookies)} cookie(s) from '{cookies_file}'.")
                except Exception as e:
                    print(f"[!] Warning: Could not add cookies from '{cookies_file}': {e}")

        page = context.new_page()
        try:
            page.goto(target_url, wait_until="networkidle", timeout=40000)
        except Exception as e:
            print(f"[!] Warning during page load: {e} — continuing anyway.")

        if interactive_login:
            print("\n" + "=" * 68)
            print(" [LOGIN MODE] Browser window is open.")
            print(" Please log in (e.g. to Tumblr / AO3) in the opened Chromium window.")
            print(" Once the story page is visible, press Enter here to save your session.")
            print("=" * 68)
            try:
                input("Press Enter after logging in > ")
            except EOFError:
                pass
            try:
                context.storage_state(path=resolved_state_file)
                print(f"[✓] Saved authenticated session to '{resolved_state_file}' for future runs!")
            except Exception as e:
                print(f"[!] Warning: Could not save session state: {e}")
            try:
                if page.url != target_url and ("login" in page.url.lower() or "dashboard" in page.url.lower()):
                    page.goto(target_url, wait_until="networkidle", timeout=40000)
            except Exception:
                pass

        # Wait for initial story paragraphs to hydrate
        print("[*] Waiting for page content to load...")
        try:
            page.wait_for_function(
                "document.querySelectorAll('p').length > 3", timeout=20000
            )
        except Exception:
            print("[!] Paragraph wait timed out. Proceeding with current content.")

        # Check for Tumblr login wall / overlays and try public mobile fallback if needed
        is_tumblr_candidate = "tumblr.com" in target_url.lower() or bool(tumblr_blog)
        if is_tumblr_candidate:
            still_walled = _dismiss_tumblr_walls_and_check_login(page)
            if still_walled and tumblr_mobile_url and page.url.rstrip("/") != tumblr_mobile_url.rstrip("/"):
                print(f"[*] Tumblr login wall detected; trying public mobile view: '{tumblr_mobile_url}'...")
                try:
                    page.goto(tumblr_mobile_url, wait_until="networkidle", timeout=30000)
                    page.wait_for_timeout(1000)
                    still_walled = _dismiss_tumblr_walls_and_check_login(page)
                except Exception:
                    pass
            if still_walled:
                print(
                    "[!] Warning: This Tumblr blog appears to be dashboard-only or age-restricted.\n"
                    "    Tip: Run once with `--login` (opens a browser to sign in and saves your session\n"
                    "    to '.playwright_auth.json' automatically) or pass `--cookies cookies.txt`."
                )

        # Expand Tumblr 'Keep reading' cuts before scrolling / Reader Mode
        keep_reading_clicks = _expand_tumblr_keep_reading(page, target_url)
        is_tumblr_page = is_tumblr_candidate or keep_reading_clicks > 0

        if is_tumblr_page:
            # Tumblr uses a virtualized feed: scrolling to document.body.scrollHeight
            # unmounts expanded paragraphs from the primary article.
            page.evaluate("window.scrollTo(0, 0);")
            page.wait_for_timeout(500)
        else:
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

        page_title = page.title()
        page_meta = page.evaluate("""
            () => {
                const pick = (sel) => {
                    const el = document.querySelector(sel);
                    if (!el) return '';
                    return (el.getAttribute('content') || el.innerText || '').trim();
                };
                const rawOg = pick('meta[property="og:title"]');
                const usableOg = /^post\\s+by\\s+@?\\S+$/i.test(rawOg) ? '' : rawOg;
                const workTitle =
                    pick('#workskin .preface h2.title') ||
                    pick('h1.title') ||
                    usableOg ||
                    '';
                const authorEl = document.querySelector(
                    '#workskin .preface a[rel="author"], a[rel="author"], .byline a'
                );
                let metaAuthor = authorEl
                    ? (authorEl.innerText || '').trim()
                    : (pick('meta[name="author"]') || '');
                if (!metaAuthor && window.location.hostname.toLowerCase().includes('tumblr.com')) {
                    const tumblrAuthor = document.querySelector(
                        'article header a[href*="/"], .post-author, .blog-title'
                    );
                    if (tumblrAuthor) {
                        const txt = (tumblrAuthor.innerText || '').trim();
                        if (txt && txt.toLowerCase() !== 'follow' && txt.length <= 60) {
                            metaAuthor = txt;
                        }
                    }
                }
                return { workTitle, metaAuthor };
            }
        """)
        meta_title = (page_meta or {}).get("workTitle", "") or ""
        meta_author = (page_meta or {}).get("metaAuthor", "") or ""

        # Reader Mode: apply Readability.js or fiction-platform container extraction
        if reader_mode:
            print("[*] Reader Mode enabled — extracting main article content...")
            extracted = page.evaluate("""
                async () => {
                    // 1. Direct container extraction for fiction platforms (AO3, ao3saver, Royal Road, FFnet, etc.)
                    const directContainer = document.querySelector(
                        '#chapters, #workskin, .chapter-inner, #storytext, .chp_raw, .entry-content'
                    );
                    if (directContainer && directContainer.innerText && directContainer.innerText.trim().length > 300) {
                        return { html: `<h1>${document.title}</h1>` + directContainer.innerHTML, title: '', byline: '' };
                    }

                    // 2. Readability.js for generic articles / blogs, protecting chapter headings
                    let html = "";
                    let rTitle = "";
                    let rByline = "";
                    try {
                        const mod = await import('https://cdn.skypack.dev/@mozilla/readability');
                        const clone = document.cloneNode(true);
                        // Prevent Readability from discarding chapter headers wrapped in meta/byline containers
                        clone.querySelectorAll('.meta, .byline, .header').forEach(el => {
                            if (el.querySelector('h1, h2, h3, h4, h5, h6') || /chapter/i.test(el.className)) {
                                el.classList.remove('meta', 'byline', 'header');
                            }
                        });
                        const article = new mod.Readability(clone).parse();
                        if (article && article.content) {
                            rTitle = (article.title || '').trim();
                            rByline = (article.byline || '').trim();
                            html = `<h1>${rTitle || document.title}</h1>` + article.content;
                        }
                    } catch (e) {}

                    // 3. Tumblr post fallback (avoids CSP-blocked Readability pulling recommended feed posts)
                    if (!html && (window.location.hostname.toLowerCase().includes('tumblr.com')
                                  || document.querySelector('article[data-id], [data-testid="post-content"], .tmblr-truncated, a.read_more'))) {
                        const tumblrPost = document.querySelector(
                            'main article, [role="main"] article, article[data-id], article, .post .body-text, .post .post-body, .post'
                        );
                        if (tumblrPost && tumblrPost.innerText && tumblrPost.innerText.trim().length > 100) {
                            const clone = tumblrPost.cloneNode(true);
                            clone.querySelectorAll(
                                'nav, footer, aside, button, script, style, [role="navigation"], .post-footer, .notes, #notes, ol.notes, [data-testid="post-footer"], [aria-label*="notes" i], [aria-label*="tags" i]'
                            ).forEach(el => el.remove());
                            clone.querySelectorAll('header').forEach(el => {
                                if (!el.querySelector('h1, h2, h3, p')) {
                                    el.remove();
                                }
                            });
                            html = `<h1>${document.title}</h1>` + clone.innerHTML;
                        }
                    }

                    if (!html) {
                        // 4. Generic fallback: strip obvious chrome from body
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll(
                            'nav, header, footer, button, script, style, aside, [role="navigation"]'
                        ).forEach(el => el.remove());
                        html = clone.innerHTML;
                    }
                    return { html, title: rTitle, byline: rByline };
                }
            """)
            extracted_html = (extracted or {}).get("html", "")
            if not meta_title and (extracted or {}).get("title"):
                meta_title = extracted["title"]
            if not meta_author and (extracted or {}).get("byline"):
                meta_author = extracted["byline"]

            if not extracted_html or len(extracted_html.strip()) < 100:
                print("[!] Reader Mode extraction returned minimal content — falling back to full page.")
                extracted_html = page.content()
        else:
            extracted_html = page.content()

        browser.close()

    return extracted_html, page_title, meta_title, meta_author


def _save_outputs(
    html_content: str,
    raw_page_title: str,
    meta_title: str,
    meta_author: str,
    output_name: str,
    output_format: str,
    source_url: str = None,
    plain_text_override: str = None,
) -> str:
    """Writes requested .txt and/or .epub outputs and returns primary saved path."""
    fallback_stem = raw_page_title or meta_title or "webpage_output"
    clean_text = plain_text_override if plain_text_override is not None else None
    saved_txt_path = None
    saved_epub_path = None

    if output_format in ("txt", "both"):
        if clean_text is None:
            print("[*] Converting HTML to clean text...")
            clean_text = html_to_clean_text(html_content)
        if len(clean_text) < 200:
            print("[!] Warning: Extracted text is very short. The page may be JS-gated or require login.")
        txt_path = _resolve_output_path(output_name, fallback_stem, ".txt", RAW_TXT_DIR)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"\n[+] Saved to: {txt_path}  ({len(clean_text):,} characters)")
        saved_txt_path = txt_path

    if output_format in ("epub", "both"):
        print("[*] Converting content to EPUB...")
        epub_path = _resolve_output_path(
            output_name,
            meta_title or clean_story_title(fallback_stem),
            ".epub",
            RAW_EPUB_DIR,
        )
        epub_title = meta_title or clean_story_title(raw_page_title) or fallback_stem
        if plain_text_override is not None:
            text_to_epub(
                plain_text_override,
                output_path=epub_path,
                title=epub_title,
                author=meta_author,
                source_url=source_url,
            )
        else:
            html_to_epub(
                html_content,
                output_path=epub_path,
                title=meta_title or None,
                author=meta_author or None,
                source_url=source_url,
                raw_page_title=raw_page_title or None,
            )
        epub_size_kb = os.path.getsize(epub_path) / 1024.0
        print(f"\n[+] Saved EPUB to: {epub_path}  ({epub_size_kb:,.1f} KB)")
        saved_epub_path = epub_path

    return saved_txt_path if saved_txt_path else saved_epub_path


def scrape_url_to_txt(
    url: str,
    reader_mode: bool = False,
    headless: bool = True,
    output_name: str = None,
    output_format: str = "txt",
    title: str = None,
    author: str = None,
    cookies_file: str = None,
    storage_state_file: str = None,
    interactive_login: bool = False,
):
    """
    Scrapes a webpage (or reads a local .html/.txt file) and saves clean .txt and/or .epub.
    Returns the primary saved output path (.txt when output_format is 'txt' or 'both';
    .epub when output_format is 'epub'), or None on failure.
    """
    fmt = (output_format or "txt").lower().strip()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{output_format}'. Expected one of: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
        )

    is_local_file = not url.lower().startswith(("http://", "https://")) and os.path.isfile(url)
    if is_local_file:
        ext = os.path.splitext(url)[1].lower()
        stem_from_file = os.path.splitext(os.path.basename(url))[0]
        with open(url, "r", encoding="utf-8") as f:
            raw_content = f.read()

        if ext in (".html", ".htm", ".xhtml"):
            detected_title, detected_author = extract_metadata_from_html(
                raw_content, fallback_title=stem_from_file, fallback_author=author
            )
            return _save_outputs(
                html_content=raw_content,
                raw_page_title=stem_from_file,
                meta_title=title or detected_title,
                meta_author=author or detected_author,
                output_name=output_name,
                output_format=fmt,
            )

        clean_plain = raw_content.strip()
        if not clean_plain:
            print(f"[!] Warning: Input file '{url}' is empty.")
        return _save_outputs(
            html_content="",
            raw_page_title=stem_from_file,
            meta_title=title or stem_from_file.replace("_", " "),
            meta_author=author or "",
            output_name=output_name,
            output_format=fmt,
            plain_text_override=clean_plain,
        )

    extracted_html, page_title, meta_title, meta_author = _fetch_url_html_with_playwright(
        url=url,
        reader_mode=reader_mode,
        headless=headless,
        cookies_file=cookies_file,
        storage_state_file=storage_state_file,
        interactive_login=interactive_login,
    )
    resolved_meta_title = title or clean_story_title(meta_title)
    resolved_meta_author = author or meta_author

    return _save_outputs(
        html_content=extracted_html,
        raw_page_title=page_title,
        meta_title=resolved_meta_title,
        meta_author=resolved_meta_author,
        output_name=output_name,
        output_format=fmt,
        source_url=url,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape a webpage (or local .html/.txt) and save as .txt (raw_txt/), .epub (raw_epub/), or both."
    )
    parser.add_argument("url", help="URL of the webpage (or path to a local .html/.txt file) to convert.")
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
        "--login", "--save-login",
        dest="interactive_login",
        action="store_true",
        help="Open a visible browser so you can log in (e.g. Tumblr/AO3) and save your session to .playwright_auth.json.",
    )
    parser.add_argument(
        "--cookies",
        dest="cookies_file",
        type=str,
        default=None,
        help="Path to a Netscape cookies.txt or JSON cookies file for authenticated scraping.",
    )
    parser.add_argument(
        "--auth-state", "--storage-state",
        dest="storage_state_file",
        type=str,
        default=None,
        help=f"Path to Playwright storage state file (default: {DEFAULT_STORAGE_STATE_FILE}).",
    )
    parser.add_argument(
        "--epub",
        action="store_true",
        help="Save output as .epub instead of .txt (shorthand for --format epub).",
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_OUTPUT_FORMATS,
        default=None,
        help="Output format: txt (default), epub, or both.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output filename or path (optional .txt/.epub extension).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override book/story title metadata (especially for EPUB exports).",
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Override author metadata for EPUB exports.",
    )

    args = parser.parse_args()
    if args.epub and args.format == "txt":
        parser.error("--epub cannot be combined with --format txt.")

    if args.format is not None:
        selected_format = args.format
    elif args.epub or (args.out and args.out.strip().lower().endswith(".epub")):
        selected_format = "epub"
    else:
        selected_format = "txt"

    scrape_url_to_txt(
        url=args.url,
        reader_mode=args.reader,
        headless=not (args.visible or args.interactive_login),
        output_name=args.out,
        output_format=selected_format,
        title=args.title,
        author=args.author,
        cookies_file=args.cookies_file,
        storage_state_file=args.storage_state_file,
        interactive_login=args.interactive_login,
    )
