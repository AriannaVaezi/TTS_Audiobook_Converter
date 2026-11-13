import argparse
import glob
import json
import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==================== CONFIGURATION ====================
RAW_DIR = "raw_txt"
OUTPUT_DIR = "parsed_chapters"
TAGGED_DIR = "tagged_chapters"
FINISHED_DIR = "finished"
MANIFEST_FILE = "chapters.txt"
REPLACEMENTS_FILE = "replacements.json"
# ========================================================


def ensure_replacements_file():
    """Creates a default replacements.json template if it doesn't exist."""
    if not os.path.exists(REPLACEMENTS_FILE):
        default_data = {
            "Dr.": "Doctor",
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "etc.": "et cetera",
        }
        with open(REPLACEMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=4)
        print(f"[*] Created default '{REPLACEMENTS_FILE}' template.")


def load_replacements():
    """Loads pronunciation word replacements from JSON."""
    ensure_replacements_file()
    try:
        with open(REPLACEMENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Warning loading {REPLACEMENTS_FILE}: {e}. Skipping word replacements.")
        return {}


def apply_word_replacements(text, replacements):
    """Applies word replacement dictionary safely using word boundaries where possible."""
    if not replacements:
        return text

    sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
    for word in sorted_keys:
        replacement = replacements[word]
        if re.match(r"^\w", word) and re.match(r"\w$", word):
            pattern = r"\b" + re.escape(word) + r"\b"
        else:
            pattern = re.escape(word)
        text = re.sub(pattern, replacement, text)

    return text


def sanitize_filename(name):
    """Converts a string into a clean, filesystem-safe filename."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned


DIVIDER_LINE_REGEX = re.compile(
    r"^\s*(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|(?:[—–]\s*){3,})\s*$",
    re.MULTILINE,
)


def normalize_dividers(text):
    """
    Ensures there is never 2 '***' in a row.
    Collapses multiple consecutive '***' lines into a single '***'.
    Standardizes * * *, ---, ___, and em-dash dividers into '***'.
    """
    text = DIVIDER_LINE_REGEX.sub("***", text)
    text = re.sub(r"(?:\n\s*\*\*\*\s*)+", "\n\n***\n\n", text)
    return text.strip()


def turn_multi_newlines_to_dividers(text):
    """
    Turns sequences of more than one new line in a row (consecutive blank lines
    marking scene breaks) into '***' section dividers.
    Matches 2 or more blank lines (3 or more consecutive newlines), preserving
    single-blank-line paragraph breaks while converting multi-blank-line scene
    transitions into '***'.
    """
    pattern = r"(?:[ \t]*\r?\n[ \t]*){3,}"
    text = re.sub(pattern, "\n\n***\n\n", text)
    return normalize_dividers(text)


def unwrap_hard_wraps(text):
    """
    Intelligently heals paragraphs split across lines/blank spaces by checking
    if blocks end with sentence punctuation or if the next block starts with lowercase,
    while safely preserving dialog blocks, short lines, and section dividers.
    """
    blocks = re.split(r"\n\s*\n", text)
    merged_paragraphs = []
    current_paragraph = ""

    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue

        # Keep dividers and chapter headers isolated
        if re.match(
            r"^(?:Chapter\s+\d+|Chapter\s+\d+\s*[:\-].*|\*\*\*|\—|\-|\–|\.)$",
            stripped,
            re.IGNORECASE,
        ):
            if current_paragraph:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = ""
            merged_paragraphs.append(stripped)
            continue

        if not current_paragraph:
            current_paragraph = stripped
        else:
            last_char = current_paragraph[-1]
            starts_with_lower = stripped[0].islower() if stripped else False
            is_strong_end = last_char in [".", "!", "?", '"', "’", "”", "***"]

            if not is_strong_end or starts_with_lower:
                current_paragraph += " " + stripped
            else:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = stripped

    if current_paragraph:
        merged_paragraphs.append(current_paragraph)

    return "\n\n".join(merged_paragraphs)


# ==================== CUT PIPELINES ====================

NOTE_POINTER_REGEX = re.compile(
    r"^\s*(?:\(?\s*See\s+the\s+end\s+of\s+the\s+(?:chapter|work)\s+for\s+(?:more\s+)?notes\.?\s*\)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

END_NOTES_REGEX = re.compile(
    r"(?im)^\s*(?:Chapter\s+End\s+Notes|End\s+Notes)\b.*$",
    re.DOTALL,
)


def cut_note_pointer_and_lines_above(text_after_header):
    """
    If 'See the end of the chapter for notes' (or variations) is found,
    cuts from the start of the body up to and including that note line.
    """
    match = NOTE_POINTER_REGEX.search(text_after_header)
    if match:
        cut_end_pos = match.end()
        # Return everything after the note pointer line
        return text_after_header[cut_end_pos:].strip()
    return text_after_header.strip()


def cut_end_notes(text):
    """
    If 'Afterword', 'Chapter End Notes', or 'End Notes' is found on a line,
    cuts that line and everything following it.
    """
    match = re.search(r"(?im)^\s*(?:Afterword|Chapter\s+End\s+Notes|End\s+Notes)\b", text)
    if match:
        return text[:match.start()].strip()
    return text.strip()


def cut_summary_and_next_text_line(text):
    """
    If 'Chapter Summary' or 'Summary' is found alone on a line,
    cuts that line and the next line containing text.
    """
    lines = text.split("\n")
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if re.match(r"^(?:Chapter\s+Summary|Summary)$", stripped, re.IGNORECASE):
            # Cut this summary header line
            i += 1
            # Skip empty lines until the next line containing text
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Cut the next line containing text (the summary content)
            if i < len(lines):
                i += 1
            continue

        result_lines.append(line)
        i += 1

    return "\n".join(result_lines).strip()


def cut_trailing_sections_interactive(body_text, chapter_label="", session_state=None):
    """
    Interactively reviews trailing sections separated by '***' or End Notes headers
    from the end of a chapter. Shows an excerpt before confirming the cut.
    Supports:
      [y] Cut section
      [n] Keep section (default)
      [a] Cut this & automatically cut trailing section from all remaining chapters
      [s] Skip / keep all remaining chapters
    """
    if session_state is None:
        session_state = {}

    if session_state.get("auto_cut_all"):
        return cut_trailing_sections_bulk(body_text, 1)
    if session_state.get("auto_keep_all"):
        return body_text

    # First check if an End Notes or Afterword header exists in the text
    end_notes_match = re.search(r"(?im)^\s*(?:Afterword|Chapter\s+End\s+Notes|End\s+Notes)\b", body_text)
    if end_notes_match:
        end_notes_excerpt = body_text[end_notes_match.start():min(len(body_text), end_notes_match.start() + 250)].replace("\n", " ").strip()
        print("\n" + "=" * 70)
        print(f"[{chapter_label}] End Notes / Afterword Detected")
        print(f'"{end_notes_excerpt}..."')
        print("-" * 70)
        print("Options: [y] Cut End Notes | [n] Keep End Notes (default) | [a] Cut in all chapters | [s] Keep in all chapters")
        raw_choice = input("Cut this End Notes section? [y/n/a/s] (default 'n'): ").strip().lower()
        choice = raw_choice if raw_choice else "n"
        if choice in ("a", "all"):
            session_state["auto_cut_all"] = True
            body_text = body_text[:end_notes_match.start()].strip()
            print("  [+] Cut End Notes. Will cut trailing notes from all remaining chapters.")
        elif choice in ("s", "skip"):
            session_state["auto_keep_all"] = True
            print("  [-] Kept End Notes. Will keep all remaining chapters.")
        elif choice in ("y", "yes"):
            body_text = body_text[:end_notes_match.start()].strip()
            print("  [+] Cut End Notes section.")
        else:
            print("  [-] Kept End Notes section.")

    body_text = normalize_dividers(body_text)
    cleaned_body = re.sub(r"(?:\n\s*\*\*\*\s*)+$", "", body_text).strip()
    parts = [p.strip() for p in re.split(r"\n\s*\*\*\*\s*\n", cleaned_body) if p.strip()]
    if len(parts) <= 1:
        return body_text

    while len(parts) > 1:
        last_section = parts[-1]
        excerpt_length = min(300, len(last_section))
        excerpt = last_section[:excerpt_length].replace("\n", " ").strip()
        if len(last_section) > excerpt_length:
            excerpt += "..."

        print("\n" + "=" * 70)
        print(f"[{chapter_label}] Trailing Section Review ({len(parts)} total sections)")
        print(f"Trailing section ({len(last_section)} chars):")
        print(f'"{excerpt}"')
        print("-" * 70)
        print("Options: [y] Cut | [n] Keep | [a] Cut this & all remaining | [s] Skip all remaining")
        raw_choice = input("Cut this section? [y/n/a/s] (default 'n'): ").strip().lower()
        choice = raw_choice if raw_choice else "n"

        if choice in ("a", "all"):
            session_state["auto_cut_all"] = True
            parts.pop()
            print("  [+] Cut trailing section. Will automatically cut 1 trailing section from all remaining chapters.")
            break
        elif choice in ("s", "skip"):
            session_state["auto_keep_all"] = True
            print("  [-] Kept section. Will keep all trailing sections in remaining chapters.")
            break
        elif choice in ("y", "yes"):
            parts.pop()
            print(f"  [+] Cut trailing section. {len(parts)} sections remaining.")
            break
        else:
            print("  [-] Kept section.")
            break

    return "\n\n***\n\n".join(parts).strip()


def cut_trailing_sections_bulk(body_text, cut_count):
    """Cuts N trailing sections or end notes from the end of a body."""
    if cut_count <= 0:
        return body_text
    body_text = cut_end_notes(body_text)
    body_text = normalize_dividers(body_text)
    cleaned_body = re.sub(r"(?:\n\s*\*\*\*\s*)+$", "", body_text).strip()
    parts = [p.strip() for p in re.split(r"\n\s*\*\*\*\s*\n", cleaned_body) if p.strip()]
    if len(parts) <= cut_count:
        return body_text
    retained = parts[:-cut_count]
    return "\n\n***\n\n".join(retained).strip()


# ==================== MAIN PARSING LOGIC ====================

def clean_web_fiction_header_and_metadata(raw_text, input_file_path):
    """
    Cleans AO3, Ellipsus, FicHub, and standard raw web fiction text:
    Extracts clean work title, author, clean short filename, and trims preambles & footers.
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    first_line = lines[0] if lines else ""

    work_title = ""
    author = ""

    # Detect platforms
    is_ao3 = (
        "archiveofourown.org" in raw_text
        or "Archive of Our Own" in raw_text
        or "ao3saver" in raw_text
        or "Rating:" in raw_text
        or "Archive Warning:" in raw_text
    )
    is_ellipsus = "Ellipsus" in first_line or "ellipsus.com" in raw_text
    is_fichub = "FicHub.net" in raw_text

    # 1. Extract Title and Author
    # Pattern A: Web archive line 1: "Title - Author - Category..."
    ao3_m = re.match(r"^(.*?)\s+-\s+([a-zA-Z0-9_\-]+)\s+-\s+.*$", first_line)
    if ao3_m:
        work_title = ao3_m.group(1).strip()
        author = ao3_m.group(2).strip()
        is_ao3 = True
    elif is_ao3:
        pref_m = re.search(r"(?im)^Preface\s*\n+([^\n]+)\n+\s*Posted originally on the Archive of Our Own", raw_text)
        if pref_m:
            work_title = pref_m.group(1).strip()
        author_m = re.search(r"(?im)^by\s+([a-zA-Z0-9_\-]+)\s*$", raw_text)
        if author_m:
            author = author_m.group(1).strip()

    # Pattern B: Ellipsus: "Title | Author | Ellipsus"
    if is_ellipsus and not work_title:
        ellipsus_m = re.match(r"^(.*?)\s*\|\s*([a-zA-Z0-9_\.\-]+)\s*\|\s*Ellipsus\b", first_line, re.IGNORECASE)
        if ellipsus_m:
            raw_t = ellipsus_m.group(1).strip()
            work_title = re.sub(r"\[.*?\]+", "", raw_t).strip()
            author = ellipsus_m.group(2).strip()

    # Pattern C: FicHub / Standard: "Title" then "By: Author"
    if not work_title:
        by_m = re.search(r"(?im)^\s*(?:By:\s*|by\s+)([a-zA-Z0-9_\.\-]+)\s*$", raw_text)
        if by_m:
            author = by_m.group(1).strip()
            work_title = first_line.strip()

    # Fallback to sanitized filename
    if not work_title:
        bname = os.path.splitext(os.path.basename(input_file_path))[0]
        bname = re.sub(r"_-_.*$", "", bname)
        work_title = bname.replace("_", " ").strip()

    if author:
        clean_file_base = f"{sanitize_filename(work_title)}_{sanitize_filename(author)}.txt"
    else:
        clean_file_base = f"{sanitize_filename(work_title)}.txt"

    if len(clean_file_base) > 60:
        clean_file_base = clean_file_base[:56] + ".txt"

    body_text = raw_text

    # Strip top platform metadata header (e.g. Title - Author - Fandom/Tags) so it never leaks into story content
    if first_line and (ao3_m or is_ellipsus or (work_title and work_title in first_line and "-" in first_line)):
        first_line_end = body_text.find("\n")
        if first_line_end != -1:
            body_text = body_text[first_line_end + 1:].strip()

    # Pre-clean AO3 preamble / summary / notes for single works
    if is_ao3 and work_title:
        title_escaped = re.escape(work_title)
        matches = list(re.finditer(rf"(?im)^\s*{title_escaped}\s*$", body_text))
        if len(matches) >= 2:
            last_title_match = matches[-1]
            body_text = body_text[last_title_match.end():].strip()

    # Pre-clean Ellipsus socials (header already stripped above)
    if is_ellipsus:
        body_text = re.sub(r"(?im)^\s*Socials\s*:.*$\n*", "", body_text)

    # Cut end notes / afterword / footer comments across all formats
    body_text = re.sub(r"(?im)^\s*(?:Afterword|Chapter\s+End\s+Notes|End\s+Notes)\b.*", "", body_text, flags=re.DOTALL)
    body_text = re.sub(r"(?im)Please drop by the Archive and comment.*", "", body_text, flags=re.DOTALL)
    body_text = re.sub(r"(?im)You need an Ellipsus account to comment.*", "", body_text, flags=re.DOTALL)

    return work_title, author, clean_file_base, body_text


def get_existing_books(parsed_dir=None, tagged_dir=None, finished_dir=None):
    """
    Returns a dictionary of existing book directories found across
    parsed_chapters/, tagged_chapters/, and finished/.
    """
    p_dir = parsed_dir or OUTPUT_DIR
    t_dir = tagged_dir or TAGGED_DIR
    f_dir = finished_dir or FINISHED_DIR

    books = {}
    for d in (p_dir, t_dir):
        if os.path.exists(d):
            for entry in os.scandir(d):
                if entry.is_dir() and not entry.name.startswith("."):
                    txts = [f for f in os.listdir(entry.path) if f.endswith(".txt") and os.path.getsize(os.path.join(entry.path, f)) > 0]
                    if entry.name not in books:
                        books[entry.name] = {"name": entry.name, "chapter_count": 0}
                    books[entry.name]["chapter_count"] = max(books[entry.name]["chapter_count"], len(txts))

    if os.path.exists(f_dir):
        for entry in os.scandir(f_dir):
            if entry.is_dir() and entry.name.endswith("_hybrid_audiobook_output"):
                bname = entry.name[:-len("_hybrid_audiobook_output")]
                if bname not in books:
                    books[bname] = {"name": bname, "chapter_count": 0}

    return books


def resolve_book_name(query, existing_books):
    """Fuzzy / case-insensitive match for existing book names."""
    if not query:
        return None
    q_norm = re.sub(r"[\s\-_]+", "", query.strip().lower())
    for name in existing_books.keys():
        if re.sub(r"[\s\-_]+", "", name.lower()) == q_norm:
            return name
    return sanitize_filename(query.strip())


def interactive_select_existing_book(existing_books):
    """Displays menu of existing books and returns selected book folder name."""
    book_list = sorted(existing_books.values(), key=lambda b: b["name"].lower()) if existing_books else []
    if book_list:
        print("\nExisting Books in Workspace:")
        for idx, b in enumerate(book_list, 1):
            cnt = b.get("chapter_count", 0)
            cnt_str = f" ({cnt} chapter{'s' if cnt != 1 else ''})" if cnt else ""
            print(f"  [{idx:2d}] {b['name']}{cnt_str}")
        print("  [ 0] Enter a different book name manually")

        choice = input(f"\nSelect book number [1-{len(book_list)}] or enter name (default 1): ").strip()
        if not choice or choice == "1":
            return book_list[0]["name"]
        if choice == "0":
            custom = input("Enter book title: ").strip()
            return sanitize_filename(custom) if custom else book_list[0]["name"]
        try:
            num = int(choice)
            if 1 <= num <= len(book_list):
                return book_list[num - 1]["name"]
        except ValueError:
            matched = resolve_book_name(choice, existing_books)
            if matched:
                return matched
            return sanitize_filename(choice)
        return book_list[0]["name"]
    else:
        custom = input("\nEnter the existing book title: ").strip()
        return sanitize_filename(custom) if custom else "audiobook"


def inspect_existing_book_chapters(book_name, parsed_dir=None, tagged_dir=None):
    """
    Scans existing chapters for book_name in parsed_chapters/ and tagged_chapters/.
    Returns info dict with max track prefix, max chapter number, intro status, and existing manifest files.
    """
    p_dir_base = parsed_dir or OUTPUT_DIR
    t_dir_base = tagged_dir or TAGGED_DIR

    safe_name = sanitize_filename(book_name)
    p_dir = os.path.join(p_dir_base, safe_name)
    t_dir = os.path.join(t_dir_base, safe_name)

    p_files = {}
    if os.path.exists(p_dir):
        for f in os.listdir(p_dir):
            if f.endswith(".txt") and os.path.getsize(os.path.join(p_dir, f)) > 0:
                p_files[f] = os.path.join(p_dir, f)

    t_files = {}
    if os.path.exists(t_dir):
        for f in os.listdir(t_dir):
            if f.endswith(".txt") and os.path.getsize(os.path.join(t_dir, f)) > 0:
                t_files[f] = os.path.join(t_dir, f)

    all_basenames = sorted(set(p_files.keys()) | set(t_files.keys()))

    has_intro = any("intro" in f.lower() for f in all_basenames)
    max_track = 0
    max_chapter = 0

    def track_sort_key(fname):
        m = re.match(r"^(\d+)", fname)
        return (int(m.group(1)) if m else 9999, fname)

    sorted_basenames = sorted(all_basenames, key=track_sort_key)
    existing_manifest = []
    existing_chapter_nums = set()

    for bname in sorted_basenames:
        m_track = re.match(r"^(\d+)", bname)
        if m_track:
            max_track = max(max_track, int(m_track.group(1)))

        m_chap = re.search(r"Chapter_(\d+)", bname, re.IGNORECASE)
        if m_chap:
            c_num = int(m_chap.group(1))
            max_chapter = max(max_chapter, c_num)
            existing_chapter_nums.add(c_num)

        if bname in t_files:
            existing_manifest.append(t_files[bname])
        elif bname in p_files:
            existing_manifest.append(p_files[bname])

    return {
        "book_name": safe_name,
        "has_intro": has_intro,
        "max_track": max_track,
        "max_chapter": max_chapter,
        "existing_chapter_nums": sorted(existing_chapter_nums),
        "existing_manifest_files": existing_manifest,
        "parsed_dir": p_dir,
        "tagged_dir": t_dir,
    }


def update_manifest_file(new_files, manifest_file=None, existing_files=None):
    """
    Updates manifest_file, preserving existing chapters and replacing
    matching basenames (e.g. replacing parsed chapter with tagged chapter) or appending new chapters.
    """
    m_file = manifest_file or MANIFEST_FILE
    current_lines = []
    if os.path.exists(m_file):
        with open(m_file, "r", encoding="utf-8") as mf:
            current_lines = [l.strip() for l in mf if l.strip() and not l.startswith("#")]

    combined_base = list(existing_files or current_lines)
    tagged_map = {os.path.basename(f): f for f in new_files}

    final_list = []
    for line in combined_base:
        base = os.path.basename(line)
        if base in tagged_map:
            final_list.append(tagged_map.pop(base))
        else:
            final_list.append(line)

    for remaining in tagged_map.values():
        if remaining not in final_list:
            final_list.append(remaining)

    with open(m_file, "w", encoding="utf-8") as mf:
        for f in final_list:
            mf.write(f"{f}\n")
    return final_list


def parse_chapter_selection(selection_str, available_splits):
    """
    Parses a user selection string into a filtered list of chapter split dicts.
    Supports:
      - 'all' / '*' -> all available splits
      - '5' -> chapter 5
      - '5,6,8' -> chapters 5, 6, 8
      - '5-8' or '5..8' -> chapters 5 through 8 inclusive
      - '5+' or '5:' or '5-' -> chapter 5 to the end
      - Combinations: '1, 3, 5-8'
    """
    if not selection_str or selection_str.strip().lower() in ("all", "*"):
        return list(available_splits)

    s = selection_str.strip()
    num_to_split = {ch["num"]: ch for ch in available_splits}
    max_num = max((ch["num"] for ch in available_splits), default=len(available_splits))

    selected_ids = set()
    tokens = [t.strip() for t in re.split(r"[,;|\s]+", s) if t.strip()]

    for token in tokens:
        # Range '5-8' or '5..8'
        m_range = re.match(r"^(\d+)\s*(?:-|--|\.\.)\s*(\d+)$", token)
        if m_range:
            start_n, end_n = int(m_range.group(1)), int(m_range.group(2))
            for n in range(min(start_n, end_n), max(start_n, end_n) + 1):
                if n in num_to_split:
                    selected_ids.add(id(num_to_split[n]))
            continue

        # Open-ended range '5+' or '5:' or '5-'
        m_open = re.match(r"^(\d+)\s*[:\+\-]$", token)
        if m_open:
            start_n = int(m_open.group(1))
            for n in range(start_n, max_num + 1):
                if n in num_to_split:
                    selected_ids.add(id(num_to_split[n]))
            continue

        # Single number '5'
        if token.isdigit():
            n = int(token)
            if n in num_to_split:
                selected_ids.add(id(num_to_split[n]))
            elif 1 <= n <= len(available_splits):
                selected_ids.add(id(available_splits[n - 1]))
            continue

    result = [ch for ch in available_splits if id(ch) in selected_ids]
    return result if result else list(available_splits)


def parse_book_file(
    input_file_path,
    replacements,
    should_convert_multi_nl=False,
    should_unwrap=False,
    trailing_mode="none",
    bulk_cut_count=0,
    append_to_book=None,
    track_offset=None,
    chapter_offset=None,
    only_new=None,
    select_chapters=None,
    interactive=True,
    overwrite_existing=None,
):
    print("\n" + "=" * 60)
    print(f"   Processing: {os.path.basename(input_file_path)}")
    print("=" * 60)

    with open(input_file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    raw_text = raw_text.replace("\r\n", "\n").replace("\xa0", " ")

    # Optional pre-cleaners for comment footers
    raw_text = re.sub(
        r"You need an Ellipsus account to comment or interact with comments\..*",
        "",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    raw_text = re.sub(
        r"Please drop by the Archive and comment to let the creator know if you enjoyed their work!.*",
        "",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    non_empty_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if not non_empty_lines:
        print("[!] File is empty.")
        return []

    # Clean web metadata, title, author, and filenames
    work_title, author, clean_file_base, cleaned_raw_text = clean_web_fiction_header_and_metadata(raw_text, input_file_path)
    raw_filename = os.path.basename(input_file_path)
    safe_work_title = sanitize_filename(work_title)
    author_line = f"by {author}" if author else ""

    # Rule 2: Chapter Detection
    # Chapters title are single line, typically 'Chapter [#]' or 'Chapter [#]: [title]'
    WORD_TO_NUM = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10
    }

    chapter_pattern = r"(?im)^\s*(Chapter\s+(\d+)(?:\s*[:\-]\s*(.*))?)\s*$"
    all_matches = list(re.finditer(chapter_pattern, raw_text))

    valid_splits = []
    for match in all_matches:
        start_idx = match.start(1)
        full_line = match.group(1).strip()

        # Ignore table of contents lines with trailing dots/spaces and page numbers
        if re.search(r"(?:\.{2,}|\s{3,})\s*\d+$", full_line):
            continue

        # Prevent duplicate / very close chapter triggers
        if valid_splits and start_idx - valid_splits[-1]["start"] < 100:
            continue

        chap_num = int(match.group(2))
        chap_title = match.group(3).strip() if match.group(3) else ""

        valid_splits.append({
            "start": start_idx,
            "header": full_line,
            "num": chap_num,
            "title": chap_title,
        })

    # Fallback Chapter Detection: If no standard 'Chapter N' found, check for 'Chapter One' or standalone word numbers
    if not valid_splits:
        word_chapter_pat = r"(?im)^\s*(?:Chapter\s+([a-zA-Z]+)(?:\s*[:\-]\s*(.*))?)\s*$"
        for match in re.finditer(word_chapter_pat, raw_text):
            word_part = match.group(1).strip().lower()
            if word_part in WORD_TO_NUM:
                start_idx = match.start(0)
                if valid_splits and start_idx - valid_splits[-1]["start"] < 100:
                    continue
                chap_num = WORD_TO_NUM[word_part]
                chap_title = match.group(2).strip() if match.group(2) else ""
                valid_splits.append({
                    "start": start_idx,
                    "header": match.group(0).strip(),
                    "num": chap_num,
                    "title": chap_title,
                })

    # Secondary Fallback: Standalone number words on their own line (e.g. 'One', 'Two', 'Three')
    if not valid_splits:
        standalone_pat = r"(?im)^\s*(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty)\s*$"
        standalone_matches = list(re.finditer(standalone_pat, raw_text))
        if len(standalone_matches) >= 2:
            expected_num = 1
            for match in standalone_matches:
                w = match.group(1).strip().lower()
                n = WORD_TO_NUM.get(w, 0)
                start_idx = match.start(0)
                # Verify sequential order and reasonable chapter spacing
                if n == expected_num and (not valid_splits or start_idx - valid_splits[-1]["start"] > 200):
                    valid_splits.append({
                        "start": start_idx,
                        "header": f"Chapter {n}: {match.group(1).strip()}",
                        "num": n,
                        "title": match.group(1).strip(),
                    })
                    expected_num += 1

    # Pre-calculate end positions for all identified chapters
    for idx, ch in enumerate(valid_splits):
        ch["end"] = valid_splits[idx + 1]["start"] if idx + 1 < len(valid_splits) else len(raw_text)

    safe_target_book = sanitize_filename(append_to_book) if append_to_book else None
    safe_work_title = safe_target_book or sanitize_filename(work_title)
    book_output_dir = os.path.join(OUTPUT_DIR, safe_work_title)
    os.makedirs(book_output_dir, exist_ok=True)

    existing_info = None
    if safe_target_book:
        existing_info = inspect_existing_book_chapters(safe_target_book)
        has_intro = existing_info["has_intro"]
        base_track = (existing_info["max_track"] + 1) if track_offset is None else track_offset
        base_chap = existing_info["max_chapter"] if chapter_offset is None else chapter_offset
        print(f"[*] Appending to existing book '{safe_work_title}' (existing chapters: {existing_info['max_track']}, next track: {base_track:02d}).")
    else:
        has_intro = False
        base_track = 1
        base_chap = 0

    created_files = []

    # =========================================================================
    # SCENARIO A: CHAPTERS FOUND
    # =========================================================================
    if valid_splits:
        print(f"[*] Found {len(valid_splits)} chapters in file.")

        if select_chapters:
            valid_splits = parse_chapter_selection(select_chapters, valid_splits)
            print(f"[*] Filtered to {len(valid_splits)} chapter(s) via selection: {[c['num'] for c in valid_splits]}")

        # Check for already-existing chapters when appending an updated full work
        if safe_target_book and existing_info and not select_chapters:
            existing_chapter_nums = set(existing_info.get("existing_chapter_nums", []))
            already_present = [ch for ch in valid_splits if ch["num"] in existing_chapter_nums]
            new_splits = [ch for ch in valid_splits if ch["num"] not in existing_chapter_nums]
            is_tty = (sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False) and interactive

            # Case A: Updated full work (contains both existing chapters and new chapters)
            if already_present and new_splits:
                present_nums = [ch["num"] for ch in already_present]
                range_str = f"Chapters {min(present_nums)}-{max(present_nums)}" if len(present_nums) > 1 else f"Chapter {present_nums[0]}"
                print(f"\n[*] {range_str} already exist in '{safe_work_title}'.")
                print(f"[*] Found {len(new_splits)} new chapter(s) in this file:")
                for nch in new_splits:
                    ch_title_display = f": {nch['title']}" if nch.get("title") else ""
                    print(f"    • Chapter {nch['num']}{ch_title_display}")

                if only_new is None and is_tty:
                    print("\nChapter selection options:")
                    print(f"  [Y] Add all {len(new_splits)} new chapter(s) [Default]")
                    print(f"  [A] Add all {len(valid_splits)} chapters from this file")
                    print(f"  [S] Select specific chapters or range (e.g. '5', '5-7', '5+')")
                    raw_choice = input("Select option or enter chapter numbers/range [Y/A/S] (default Y): ").strip()
                    choice = raw_choice.lower() if raw_choice else "y"

                    if choice in ("y", "yes"):
                        valid_splits = new_splits
                        print(f"[*] Extracting only the {len(valid_splits)} new chapter(s)...")
                    elif choice in ("a", "all"):
                        print(f"[*] Keeping all {len(valid_splits)} chapters.")
                    elif choice in ("s", "select"):
                        sel_str = input("Enter chapter numbers or ranges to append (e.g. '5', '5-7', '5+'): ").strip()
                        valid_splits = parse_chapter_selection(sel_str, valid_splits)
                        print(f"[*] Selected {len(valid_splits)} chapter(s): {[ch['num'] for ch in valid_splits]}")
                    else:
                        valid_splits = parse_chapter_selection(raw_choice, valid_splits)
                        print(f"[*] Selected {len(valid_splits)} chapter(s): {[ch['num'] for ch in valid_splits]}")
                elif only_new is True or (only_new is None and not is_tty):
                    valid_splits = new_splits
                    print(f"[*] Extracting only the {len(valid_splits)} new chapter(s)...")

            # Case B: File contains ONLY existing chapters (e.g. duplicate full file)
            elif already_present and not new_splits and len(already_present) >= len(existing_chapter_nums):
                print(f"\n[*] Notice: All {len(valid_splits)} chapter(s) in this file already exist in '{safe_work_title}'.")
                if overwrite_existing is False:
                    print("[*] Keeping existing book chapters intact (--no-overwrite).")
                    return existing_info.get("existing_manifest_files", [])
                elif overwrite_existing is True:
                    print("[*] Overwriting / re-parsing existing chapters (--overwrite).")
                elif only_new is None and is_tty:
                    reparse_choice = input(f"Do you want to re-parse / overwrite existing chapters? [y/N] (default N): ").strip().lower()
                    if reparse_choice != "y":
                        print("[*] Skipped parsing. Keeping existing book chapters intact.")
                        return existing_info.get("existing_manifest_files", [])
                elif only_new is True or not is_tty:
                    print("[*] No new chapters to add. Keeping existing book chapters intact.")
                    return existing_info.get("existing_manifest_files", [])

            # Case C: Multiple chapters, no overlap - prompt if user wants specific chapters
            elif len(valid_splits) > 1 and is_tty:
                print(f"\n[*] Chapters found in file: {[ch['num'] for ch in valid_splits]}")
                pick_choice = input(f"Append all {len(valid_splits)} chapters to '{safe_work_title}', or select specific chapters? [ALL / enter range e.g. '1-3', '5+'] (default ALL): ").strip()
                if pick_choice and pick_choice.lower() not in ("all", "a", "*"):
                    valid_splits = parse_chapter_selection(pick_choice, valid_splits)
                    print(f"[*] Filtered to {len(valid_splits)} chapter(s): {[ch['num'] for ch in valid_splits]}")

        # --- 1. Intro Processing ---
        if not has_intro:
            if author_line:
                intro_content = f"{work_title}\n\n{author_line}\n\n***"
            else:
                intro_content = f"{work_title}\n\n***"

            intro_content = apply_word_replacements(intro_content, replacements)
            intro_content = normalize_dividers(intro_content)

            intro_filename = os.path.join(book_output_dir, f"00_intro_{safe_work_title}.txt")
            with open(intro_filename, "w", encoding="utf-8") as out_f:
                out_f.write(intro_content)
            created_files.append(intro_filename)
            print(f"  [+] Created Intro: {intro_filename}")
        else:
            print(f"  [*] Preserved existing intro for '{safe_work_title}'.")

        # --- 2. Chapter by Chapter Processing ---
        trailing_session_state = {}
        track_counter = base_track
        for i, ch_info in enumerate(valid_splits):
            start_pos = ch_info["start"]
            end_pos = ch_info.get("end", len(raw_text))

            chapter_raw = raw_text[start_pos:end_pos].strip()
            chapter_header = ch_info["header"]

            # Separate header line from body
            header_end = chapter_raw.find("\n")
            if header_end != -1:
                body_raw = chapter_raw[header_end + 1 :].strip()
            else:
                body_raw = ""

            # Trim standalone repeated chapter titles at the end of the chapter
            if i + 1 < len(valid_splits):
                next_chap_title = valid_splits[i + 1]["title"]
                if next_chap_title:
                    esc_next = re.escape(next_chap_title)
                    body_raw = re.sub(rf"(?im)\n+\s*{esc_next}\s*$", "", body_raw)

            # Step: Make replacements using replacement json
            body_text = apply_word_replacements(body_raw, replacements)

            # Step: Cuts in EXACT specified order
            # Cut 1: If note pointer found, cut from chapter title up to and including lines above
            body_text = cut_note_pointer_and_lines_above(body_text)

            # End Notes cutting is now handled via interactive review or bulk mode

            # Cut 3: If 'Chapter Summary' or 'Summary' on a line, cut it and next text line
            body_text = cut_summary_and_next_text_line(body_text)

            # Strip any dangling trailing divider left after trimming next chapter title
            body_text = re.sub(r"(?:\n\s*(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|(?:[—–]\s*){3,})\s*)+$", "", body_text).strip()

            # Step: Turn more than one new line in a row into '***' option (before unwrap and before trailing cut)
            if should_convert_multi_nl:
                body_text = turn_multi_newlines_to_dividers(body_text)

            # Normalize dividers before trailing section cutting so all section breaks are recognized
            body_text = normalize_dividers(body_text)
            body_text = re.sub(r"(?:\n\s*\*\*\*\s*)+$", "", body_text).strip()

            # Step: Trailing sections cut option
            ch_label = f"Chapter {ch_info['num']}"
            if trailing_mode == "interactive":
                body_text = cut_trailing_sections_interactive(body_text, ch_label, trailing_session_state)
            elif trailing_mode == "bulk":
                body_text = cut_trailing_sections_bulk(body_text, bulk_cut_count)

            # Step: Fix hard-wrapped paragraph splits option
            if should_unwrap:
                body_text = unwrap_hard_wraps(body_text)

            # Compute sequential chapter number if appending
            raw_chap_num = ch_info["num"]
            if safe_target_book and base_chap >= 1 and raw_chap_num <= base_chap:
                computed_chap_num = base_chap + i + 1
            else:
                computed_chap_num = raw_chap_num

            if computed_chap_num != raw_chap_num:
                if re.search(r"\bchapter\s+\d+\b", chapter_header, re.IGNORECASE):
                    chapter_header = re.sub(r"(?i)\bchapter\s+\d+\b", f"Chapter {computed_chap_num}", chapter_header)
                elif ch_info["title"]:
                    chapter_header = f"Chapter {computed_chap_num}: {ch_info['title']}"
                else:
                    chapter_header = f"Chapter {computed_chap_num}"

            # Formatting Rule:
            # Make sure there is a '***' after first line (chapter header) and at end of file.
            # Never 2 '***' in a row.
            formatted_chapter = f"{chapter_header}\n\n***\n\n{body_text.strip()}\n\n***"
            formatted_chapter = normalize_dividers(formatted_chapter)

            # Naming convention: 0[#]_Chapter_[#]_[chapter title if available]
            chap_num_str = str(computed_chap_num)
            track_prefix = f"{track_counter:02d}"

            if ch_info["title"]:
                safe_title_part = sanitize_filename(ch_info["title"])
                chapter_file_base = f"{track_prefix}_Chapter_{chap_num_str}_{safe_title_part}.txt"
            else:
                chapter_file_base = f"{track_prefix}_Chapter_{chap_num_str}.txt"

            chapter_path = os.path.join(book_output_dir, chapter_file_base)
            with open(chapter_path, "w", encoding="utf-8") as out_f:
                out_f.write(formatted_chapter)

            created_files.append(chapter_path)
            track_counter += 1
            print(f"  [+] Parsed {chapter_header} -> {chapter_file_base}")

    # =========================================================================
    # SCENARIO B: NO CHAPTERS FOUND (SINGLE WORK)
    # =========================================================================
    else:
        print("[*] No chapters found in input file.")

        # Step: Make replacements using replacement json
        processed_text = apply_word_replacements(cleaned_raw_text, replacements)

        # Cuts in EXACT specified order:
        after_author = processed_text

        # Cut 1: If note pointer found, cut from author line (or first line) up to and including lines above
        after_author = cut_note_pointer_and_lines_above(after_author)

        # Cut 2: If 'Chapter Summary' or 'Summary' on a line, cut it and next text line
        after_author = cut_summary_and_next_text_line(after_author)

        # End Notes cutting is now handled via interactive review or bulk mode

        # Strip any dangling trailing divider
        after_author = re.sub(r"(?:\n\s*(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|(?:[—–]\s*){3,})\s*)+$", "", after_author).strip()

        # Step: Turn more than one new line in a row into '***' option (before unwrap and before trailing cut)
        if should_convert_multi_nl:
            after_author = turn_multi_newlines_to_dividers(after_author)

        # Normalize dividers before trailing section cutting
        after_author = normalize_dividers(after_author)
        after_author = re.sub(r"(?:\n\s*\*\*\*\s*)+$", "", after_author).strip()

        # Step: Trailing sections cut option
        if trailing_mode == "interactive":
            after_author = cut_trailing_sections_interactive(after_author, work_title)
        elif trailing_mode == "bulk":
            after_author = cut_trailing_sections_bulk(after_author, bulk_cut_count)

        # Step: Fix hard-wrapped paragraph splits option
        if should_unwrap:
            after_author = unwrap_hard_wraps(after_author)

        if safe_target_book:
            computed_chap_num = (base_chap + 1) if base_chap >= 1 else 1
            track_prefix = f"{base_track:02d}"
            raw_title = clean_file_base[:-4] if clean_file_base.lower().endswith(".txt") else clean_file_base
            raw_title = re.sub(r"^\d+_", "", raw_title).strip()
            if raw_title and sanitize_filename(raw_title).lower() != safe_work_title.lower():
                ch_title = raw_title.replace("_", " ")
                chapter_header = f"Chapter {computed_chap_num}: {ch_title}"
                chapter_file_base = f"{track_prefix}_Chapter_{computed_chap_num}_{sanitize_filename(ch_title)}.txt"
            else:
                chapter_header = f"Chapter {computed_chap_num}"
                chapter_file_base = f"{track_prefix}_Chapter_{computed_chap_num}.txt"

            final_content = f"{chapter_header}\n\n***\n\n{after_author.strip()}\n\n***"
            final_content = normalize_dividers(final_content)
            output_filename = os.path.join(book_output_dir, chapter_file_base)
            with open(output_filename, "w", encoding="utf-8") as out_f:
                out_f.write(final_content)
            created_files.append(output_filename)
            print(f"  [+] Saved new chapter to: {output_filename}")
        else:
            # Formatting Rule:
            # Make sure there is a '***' after author (or first line) and at the end of each file.
            # Never 2 '***' in a row.
            if author_line:
                final_content = f"{work_title}\n\n{author_line}\n\n***\n\n{after_author.strip()}\n\n***"
            else:
                final_content = f"{work_title}\n\n***\n\n{after_author.strip()}\n\n***"

            final_content = normalize_dividers(final_content)

            # Clean, safe, short output filename
            output_filename = os.path.join(book_output_dir, clean_file_base)
            with open(output_filename, "w", encoding="utf-8") as out_f:
                out_f.write(final_content)

            created_files.append(output_filename)
            print(f"  [+] Saved single work to: {output_filename}")

    # Write Manifest
    if safe_target_book and existing_info and existing_info.get("existing_manifest_files"):
        update_manifest_file(created_files, existing_files=existing_info["existing_manifest_files"])
    else:
        with open(MANIFEST_FILE, "w", encoding="utf-8") as mf:
            for cf in created_files:
                mf.write(f"{cf}\n")

    print("\n" + "=" * 60)
    print(f"[+] Successfully parsed {len(created_files)} files into '{book_output_dir}'!")
    print(f"[+] Updated manifest file '{MANIFEST_FILE}'.")
    print("=" * 60)
    return created_files


def select_input_file(custom_path=None):
    """Lists files in raw_txt and lets the user pick one, or enter a path."""
    if custom_path:
        if os.path.exists(custom_path):
            return custom_path
        print(f"[!] Warning: Specified file '{custom_path}' not found. Falling back to menu.")

    txt_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    txt_files = [f for f in txt_files if os.path.getsize(f) > 0]

    if not txt_files:
        print(f"[X] No .txt files found in '{RAW_DIR}'. Please put your text file in '{RAW_DIR}'.")
        sys.exit(1)

    print("\nAvailable files in 'raw_txt':")
    for idx, path in enumerate(txt_files, 1):
        print(f"  [{idx}] {os.path.basename(path)}")

    choice = input(f"\nSelect file number [1-{len(txt_files)}] (default 1): ").strip()
    try:
        idx = int(choice) - 1 if choice else 0
        if 0 <= idx < len(txt_files):
            return txt_files[idx]
    except ValueError:
        pass

    return txt_files[0]


def main():
    print("=" * 60)
    print("         Advanced Book & Chapter Text Parser         ")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Advanced Book & Chapter Text Parser")
    parser.add_argument("input_file", nargs="?", default=None, help="Path to input .txt file")
    parser.add_argument("--append-to", "--append", "--book", dest="append_to_book", default=None, help="Append chapter(s) to an existing book name")
    parser.add_argument("--new-book", action="store_true", default=False, help="Create a new book without prompting for destination")
    parser.add_argument("--chapters", "--chaps", dest="select_chapters", default=None, help="Specific chapter numbers or ranges to process/append (e.g. '5', '5-8', '5+').")
    parser.add_argument("--only-new", action="store_true", default=None, help="When appending an updated full work, automatically extract only newly posted chapters")
    parser.add_argument("--all-chapters", action="store_true", default=False, help="Force extracting all chapters even if previous chapters exist")
    parser.add_argument("--overwrite", dest="overwrite_existing", action="store_true", default=None, help="Overwrite existing chapters when re-parsing")
    parser.add_argument("--no-overwrite", dest="overwrite_existing", action="store_false", help="Keep existing chapters when duplicates are found")
    parser.add_argument("--multi-nl", dest="multi_nl", action="store_true", default=None, help="Convert multiple consecutive newlines to dividers (***)")
    parser.add_argument("--no-multi-nl", dest="multi_nl", action="store_false", help="Keep multiple consecutive newlines unchanged")
    parser.add_argument("--unwrap", dest="unwrap", action="store_true", default=None, help="Unwrap hard-wrapped single-line paragraph splits")
    parser.add_argument("--no-unwrap", dest="unwrap", action="store_false", help="Do not unwrap hard-wrapped lines")
    parser.add_argument("--trailing", choices=["none", "interactive", "bulk"], default=None, help="Trailing section cut mode")
    parser.add_argument("--bulk-cut", type=int, default=1, help="Number of trailing sections to cut per chapter in bulk mode")
    parser.add_argument("--no-tag", dest="run_tagger", action="store_false", default=None, help="Skip dialogue tagging")
    parser.add_argument("--tag", dest="run_tagger", action="store_true", help="Automatically trigger dialogue tagging")
    parser.add_argument("--tag-mode", "--format", dest="tag_mode", choices=["fast", "prose", "hybrid", "script"], default=None, help="Dialogue tagging engine")
    parser.add_argument("--model", "-m", dest="ollama_model", type=str, default=None, help="Ollama model name for hybrid tagging")
    parser.add_argument("--ollama-url", dest="ollama_url", type=str, default=None, help="Ollama server URL for hybrid tagging")
    parser.add_argument("--emotions", "--detect-tone", dest="detect_tone", action="store_true", default=None, help="Detect dialogue tone & emotions")
    parser.add_argument("--no-emotions", "--no-detect-tone", dest="detect_tone", action="store_false", help="Disable dialogue emotion/tone detection")
    parser.add_argument("--review-tags", "--review", dest="review_tags", action="store_true", default=None, help="Interactively review and edit dialogue tags after tagging")
    parser.add_argument("--no-review-tags", dest="review_tags", action="store_false", help="Skip interactive dialogue tag review")
    parser.add_argument("--list-tags", "--show-tags", action="store_true", default=False, help="List dialogue tags after tagging")
    parser.add_argument("--set-tag", "--change-tag", "--edit-tag", dest="set_tags", action="append", default=[], help="Change tag(s), e.g. '1=Alice' or '2,4-6=Bob:whisper'")
    parser.add_argument("--untag", dest="untag_specs", action="append", default=[], help="Remove dialogue tags so Narrator reads them")
    parser.add_argument("--remap-tag", "--rename-tag", dest="remap_tags", action="append", default=[], help="Bulk remap a speaker tag across chapters")
    parser.add_argument("--map-alias", dest="map_aliases", action="append", default=[], help="Map tag alias in tag_mappings.json")
    parser.add_argument("--interactive", action="store_true", default=False, help="Prompt interactively for unspecified options")
    parser.add_argument("--yes", "-y", "--non-interactive", dest="non_interactive", action="store_true", default=False, help="Run non-interactively with defaults for unspecified options")
    args = parser.parse_args()

    has_cli_flags = len(sys.argv) > 1
    is_interactive = args.interactive or (not has_cli_flags and not args.non_interactive)

    input_file = select_input_file(args.input_file)
    replacements = load_replacements()

    existing_books = get_existing_books(OUTPUT_DIR, TAGGED_DIR, FINISHED_DIR)
    append_to_book = None

    if args.append_to_book:
        resolved = resolve_book_name(args.append_to_book, existing_books)
        append_to_book = resolved if resolved else sanitize_filename(args.append_to_book)
        print(f"[*] Target book (from argument): '{append_to_book}'")
    elif not args.new_book and is_interactive:
        print("\nTarget book destination:")
        print("  [1] Create a new book [Default]")
        print("  [2] Add chapter(s) to an existing book (select from list or enter title)")
        dest_choice = input("Select option [1 or 2] (default 1): ").strip()
        if dest_choice == "2":
            append_to_book = interactive_select_existing_book(existing_books)
            print(f"[*] Appending to existing book: '{append_to_book}'")

    # Prompt 1: Turn more than one new line in a row to '***'
    if args.multi_nl is not None:
        should_convert_multi_nl = args.multi_nl
    elif is_interactive:
        multi_nl_choice = input("\nDo you want to turn more than one new line in a row into '***'? [y/n] (default 'n'): ").strip().lower()
        should_convert_multi_nl = multi_nl_choice == "y"
    else:
        should_convert_multi_nl = False

    # Prompt 2: Hard wrap healing
    if args.unwrap is not None:
        should_unwrap = args.unwrap
    elif is_interactive:
        unwrap_choice = input("\nDo you want to fix hard-wrapped single-line paragraph splits? [y/n] (default 'n'): ").strip().lower()
        should_unwrap = unwrap_choice == "y"
    else:
        should_unwrap = False

    # Prompt 3: Trailing sections cutting
    if args.trailing:
        trailing_mode = args.trailing
        bulk_cut_count = args.bulk_cut
    elif is_interactive:
        print("\nTrailing sections cutting (e.g. author notes at the end of chapters):")
        print("  [0] Keep all sections (no cut)")
        print("  [1] Review chapter-by-chapter (shows excerpt before cutting)")
        print("  [2] Bulk cut (cut the same number from all chapters)")
        trailing_choice = input("Select option [0, 1, or 2] (default 0): ").strip()

        trailing_mode = "none"
        bulk_cut_count = 0

        if trailing_choice == "1":
            trailing_mode = "interactive"
        elif trailing_choice == "2":
            trailing_mode = "bulk"
            count_input = input("How many trailing sections to cut from each chapter? (1 or 2, default 1): ").strip()
            try:
                bulk_cut_count = int(count_input) if count_input else 1
            except ValueError:
                bulk_cut_count = 1
    else:
        trailing_mode = "none"
        bulk_cut_count = 0

    only_new = False if args.all_chapters else (True if args.only_new else None)

    created_files = parse_book_file(
        input_file_path=input_file,
        replacements=replacements,
        should_convert_multi_nl=should_convert_multi_nl,
        should_unwrap=should_unwrap,
        trailing_mode=trailing_mode,
        bulk_cut_count=bulk_cut_count,
        append_to_book=append_to_book,
        only_new=only_new,
        select_chapters=args.select_chapters,
        interactive=is_interactive,
        overwrite_existing=args.overwrite_existing,
    )

    implied_tag = bool(
        args.tag_mode is not None
        or args.detect_tone is not None
        or args.review_tags is True
        or args.list_tags
        or args.set_tags
        or args.untag_specs
        or args.remap_tags
        or args.map_aliases
    )
    should_run_tagger = args.run_tagger if args.run_tagger is not None else (True if implied_tag else None)

    if created_files and should_run_tagger is not False:
        if should_run_tagger is None:
            if is_interactive:
                print("\n" + "=" * 60)
                print("                 AI Dialogue Tagging Option")
                print("=" * 60)
                tag_choice = input("Do you want to run the Dialogue Tagger on these chapters now? [y/n] (default 'n'): ").strip().lower()
                should_run_tagger = (tag_choice == "y")
            else:
                should_run_tagger = False

        if should_run_tagger:
            if args.tag_mode:
                chosen_format = "prose" if args.tag_mode == "fast" else args.tag_mode
            elif is_interactive:
                print("\nSelect dialogue tagging mode:")
                print("  [1] Fast Smart Parser (Instant, 100% quote coverage, zero GPU wait) [Recommended]")
                print("  [2] Hybrid LLM (Smart Parser + Local GPU LLM verification for ambiguous lines)")
                print("  [3] Script Form (Screenplay / Script format: [Name]: dialogue)")
                fmt_input = input("Enter choice [1, 2, or 3] (default 1): ").strip()
                if fmt_input == "2":
                    chosen_format = "hybrid"
                elif fmt_input == "3":
                    chosen_format = "script"
                else:
                    chosen_format = "prose"
            else:
                chosen_format = "prose"

            if args.detect_tone is not None:
                detect_tone = args.detect_tone
            elif is_interactive:
                tone_choice = input("Detect dialogue tone & emotions (whisper, shout, angry, etc.)? [y/n] (default 'y'): ").strip().lower()
                detect_tone = (tone_choice != "n")
            else:
                detect_tone = True

            try:
                import llm_dialogue_tagger
                tag_kwargs = {
                    "file_paths": created_files,
                    "dialogue_format": chosen_format,
                    "detect_tone": detect_tone,
                }
                if args.ollama_model:
                    tag_kwargs["model"] = args.ollama_model
                if args.ollama_url:
                    tag_kwargs["url"] = args.ollama_url
                tagged_files = llm_dialogue_tagger.batch_tag_files(**tag_kwargs)
                if tagged_files:
                    if args.list_tags or args.set_tags or args.untag_specs or args.remap_tags or args.map_aliases:
                        llm_dialogue_tagger.apply_cli_tag_operations(
                            tagged_files,
                            list_tags=args.list_tags,
                            set_tags=args.set_tags,
                            untag_specs=args.untag_specs,
                            remap_tags=args.remap_tags,
                            map_aliases=args.map_aliases,
                        )
                    if args.review_tags is True:
                        llm_dialogue_tagger.interactive_review_dialogue_tags(tagged_files, force_open=True)
                    elif args.review_tags is None and is_interactive:
                        llm_dialogue_tagger.interactive_review_dialogue_tags(tagged_files, force_open=False)

                    update_manifest_file(tagged_files)
                    print(f"\n[✓] Updated manifest '{MANIFEST_FILE}' to point to {len(tagged_files)} tagged chapter(s).")
                    print("[✓] Voice mappings synchronized in 'tag_mappings.json' and 'characters.json'.")
                    print("\nYou can now run the audiobook converter directly:")
                    print("  python local_hybrid_audioconverter.py")
            except Exception as e:
                print(f"[!] Error during dialogue tagging: {e}")


if __name__ == "__main__":
    main()
