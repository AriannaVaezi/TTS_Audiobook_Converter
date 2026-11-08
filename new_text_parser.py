import os
import re
import json
import glob
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==================== CONFIGURATION ====================
RAW_DIR = "raw_txt"
OUTPUT_DIR = "parsed_chapters"
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
    Interactively reviews trailing sections separated by '***' from the end of a chapter.
    Shows an excerpt before confirming the cut.
    Supports:
      [y] Cut section
      [n] Keep section (default)
      [a] Cut this & automatically cut 1 trailing section from all remaining chapters
      [s] Skip / keep all remaining chapters
    """
    if session_state is None:
        session_state = {}

    if session_state.get("auto_cut_all"):
        return cut_trailing_sections_bulk(body_text, 1)
    if session_state.get("auto_keep_all"):
        return body_text

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
    """Cuts N trailing sections from the end of a body separated by '***'."""
    if cut_count <= 0:
        return body_text
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

    # Pre-clean AO3 preamble / summary / notes for single works
    if is_ao3 and work_title:
        title_escaped = re.escape(work_title)
        matches = list(re.finditer(rf"(?im)^\s*{title_escaped}\s*$", body_text))
        if len(matches) >= 2:
            last_title_match = matches[-1]
            body_text = body_text[last_title_match.end():].strip()

    # Pre-clean Ellipsus header and socials
    if is_ellipsus:
        first_line_end = body_text.find("\n")
        if first_line_end != -1:
            body_text = body_text[first_line_end + 1:].strip()
        body_text = re.sub(r"(?im)^\s*Socials\s*:.*$\n*", "", body_text)

    # Cut end notes / afterword / footer comments across all formats
    body_text = re.sub(r"(?im)^\s*(?:Afterword|Chapter\s+End\s+Notes|End\s+Notes)\b.*", "", body_text, flags=re.DOTALL)
    body_text = re.sub(r"(?im)Please drop by the Archive and comment.*", "", body_text, flags=re.DOTALL)
    body_text = re.sub(r"(?im)You need an Ellipsus account to comment.*", "", body_text, flags=re.DOTALL)

    return work_title, author, clean_file_base, body_text


def parse_book_file(
    input_file_path,
    replacements,
    should_convert_multi_nl=False,
    should_unwrap=False,
    trailing_mode="none",
    bulk_cut_count=0,
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
    chapter_pattern = r"(?im)^\s*(Chapter\s+(\d+)(?:\s*[:\-]\s*(.*))?)\s*$"
    all_matches = list(re.finditer(chapter_pattern, raw_text))

    valid_splits = []
    for match in all_matches:
        start_idx = match.start(1)
        full_line = match.group(1).strip()

        # Ignore table of contents lines with trailing dots/spaces and page numbers (e.g. Chapter 1 ..... 15)
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    created_files = []

    # =========================================================================
    # SCENARIO A: CHAPTERS FOUND
    # =========================================================================
    if valid_splits:
        print(f"[*] Found {len(valid_splits)} chapters.")

        # --- 1. Intro Processing ---
        if author_line:
            intro_content = f"{work_title}\n\n{author_line}\n\n***"
        else:
            intro_content = f"{work_title}\n\n***"

        intro_content = apply_word_replacements(intro_content, replacements)
        intro_content = normalize_dividers(intro_content)

        intro_filename = os.path.join(OUTPUT_DIR, f"00_intro_{safe_work_title}.txt")
        with open(intro_filename, "w", encoding="utf-8") as out_f:
            out_f.write(intro_content)
        created_files.append(intro_filename)
        print(f"  [+] Created Intro: {intro_filename}")

        # --- 2. Chapter by Chapter Processing ---
        trailing_session_state = {}
        track_counter = 1
        for i, ch_info in enumerate(valid_splits):
            start_pos = ch_info["start"]
            end_pos = valid_splits[i + 1]["start"] if i + 1 < len(valid_splits) else len(raw_text)

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

            # Cut 2: If 'Chapter End Notes' or 'End Notes' found, cut that line and everything following
            body_text = cut_end_notes(body_text)

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

            # Formatting Rule:
            # Make sure there is a '***' after first line (chapter header) and at end of file.
            # Never 2 '***' in a row.
            formatted_chapter = f"{chapter_header}\n\n***\n\n{body_text.strip()}\n\n***"
            formatted_chapter = normalize_dividers(formatted_chapter)

            # Naming convention: 0[#]_Chapter_[#]_[chapter title if available]
            chap_num_str = str(ch_info["num"])
            track_prefix = f"{track_counter:02d}"

            if ch_info["title"]:
                safe_title_part = sanitize_filename(ch_info["title"])
                chapter_file_base = f"{track_prefix}_Chapter_{chap_num_str}_{safe_title_part}.txt"
            else:
                chapter_file_base = f"{track_prefix}_Chapter_{chap_num_str}.txt"

            chapter_path = os.path.join(OUTPUT_DIR, chapter_file_base)
            with open(chapter_path, "w", encoding="utf-8") as out_f:
                out_f.write(formatted_chapter)

            created_files.append(chapter_path)
            track_counter += 1
            print(f"  [+] Parsed {ch_info['header']} -> {chapter_file_base}")

    # =========================================================================
    # SCENARIO B: NO CHAPTERS FOUND (SINGLE WORK)
    # =========================================================================
    else:
        print("[*] No chapters found. Treating as a single continuous work.")

        # Step: Make replacements using replacement json
        processed_text = apply_word_replacements(cleaned_raw_text, replacements)

        # Cuts in EXACT specified order:
        after_author = processed_text

        # Cut 1: If note pointer found, cut from author line (or first line) up to and including lines above
        after_author = cut_note_pointer_and_lines_above(after_author)

        # Cut 2: If 'Chapter Summary' or 'Summary' on a line, cut it and next text line
        after_author = cut_summary_and_next_text_line(after_author)

        # Cut 3: If 'Afterword', 'Chapter End Notes', or 'End Notes' found, cut it and everything following
        after_author = cut_end_notes(after_author)

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

        # Formatting Rule:
        # Make sure there is a '***' after author (or first line) and at the end of each file.
        # Never 2 '***' in a row.
        if author_line:
            final_content = f"{work_title}\n\n{author_line}\n\n***\n\n{after_author.strip()}\n\n***"
        else:
            final_content = f"{work_title}\n\n***\n\n{after_author.strip()}\n\n***"

        final_content = normalize_dividers(final_content)

        # Clean, safe, short output filename
        output_filename = os.path.join(OUTPUT_DIR, clean_file_base)
        with open(output_filename, "w", encoding="utf-8") as out_f:
            out_f.write(final_content)

        created_files.append(output_filename)
        print(f"  [+] Saved single work to: {output_filename}")

    # Write Manifest
    with open(MANIFEST_FILE, "w", encoding="utf-8") as mf:
        for cf in created_files:
            mf.write(f"{cf}\n")

    print("\n" + "=" * 60)
    print(f"[+] Successfully parsed {len(created_files)} files into '{OUTPUT_DIR}'!")
    print(f"[+] Updated manifest file '{MANIFEST_FILE}'.")
    print("=" * 60)
    return created_files


def select_input_file():
    """Lists files in raw_txt and lets the user pick one, or enter a path."""
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
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

    input_file = select_input_file()
    replacements = load_replacements()

    # Prompt 1: Turn more than one new line in a row to '***'
    multi_nl_choice = input("\nDo you want to turn more than one new line in a row into '***'? [y/n] (default 'n'): ").strip().lower()
    should_convert_multi_nl = multi_nl_choice == "y"

    # Prompt 2: Hard wrap healing
    unwrap_choice = input("\nDo you want to fix hard-wrapped single-line paragraph splits? [y/n] (default 'n'): ").strip().lower()
    should_unwrap = unwrap_choice == "y"

    # Prompt 3: Trailing sections cutting
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

    created_files = parse_book_file(
        input_file_path=input_file,
        replacements=replacements,
        should_convert_multi_nl=should_convert_multi_nl,
        should_unwrap=should_unwrap,
        trailing_mode=trailing_mode,
        bulk_cut_count=bulk_cut_count,
    )

    if created_files:
        print("\n" + "=" * 60)
        print("                 AI Dialogue Tagging Option")
        print("=" * 60)
        tag_choice = input("Do you want to run the Dialogue Tagger on these chapters now? [y/n] (default 'n'): ").strip().lower()
        if tag_choice == "y":
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

            try:
                import llm_dialogue_tagger
                tagged_files = llm_dialogue_tagger.batch_tag_files(
                    file_paths=created_files,
                    dialogue_format=chosen_format
                )
                if tagged_files:
                    with open(MANIFEST_FILE, "w", encoding="utf-8") as mf:
                        for tf in tagged_files:
                            mf.write(f"{tf}\n")
                    print(f"\n[✓] Updated manifest '{MANIFEST_FILE}' to point to {len(tagged_files)} tagged chapter(s).")
                    print("[✓] Voice mappings synchronized in 'tag_mappings.json' and 'characters.json'.")
                    print("\nYou can now run the audiobook converter directly:")
                    print("  python local_hybrid_audioconverter.py")
            except Exception as e:
                print(f"[!] Error during dialogue tagging: {e}")


if __name__ == "__main__":
    main()
