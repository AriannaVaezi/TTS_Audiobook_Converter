"""
run_pipeline.py
===============
Unified End-to-End Audiobook Pipeline Runner

Automates the complete audiobook creation workflow in a single command:
  1. Ingestion: Accepts a web URL, ebook (.epub, .pdf, .mobi), or path to raw text.
  2. Parsing: Cleans headers, detects chapters, and splits into parsed sections.
  3. Dialogue Tagging: Automatically detects speakers and emotional delivery using
     the Fast Smart Parser (instant & 100% quote coverage).
  4. Audio Conversion: Synthesizes high-fidelity audio running CPU Kokoro and GPU F5-TTS
     simultaneously in parallel with live progress bars, timestamps, and auto-archiving.

Usage:
  python run_pipeline.py <url_or_path> [options]
  python run_pipeline.py  (Interactive menu)

Examples:
  python run_pipeline.py https://archiveofourown.org/works/12345
  python run_pipeline.py raw_txt/story.txt
  python run_pipeline.py "C:\\Books\\sample.epub"
  python run_pipeline.py --no-emotions raw_txt/sample.txt
"""

import os
import re
import sys
import glob
import shutil
import argparse
import urllib.parse

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Core pipeline imports
import url_to_txt
import ebook_to_txt
import new_text_parser as ntp
import llm_dialogue_tagger as ldt
import local_hybrid_audioconverter as lha

RAW_TXT_DIR = "raw_txt"
MANIFEST_FILE = "chapters.txt"
FINISHED_DIR = "finished"


def print_banner():
    print("=" * 72)
    print("       UNIFIED HYBRID AUDIOBOOK PIPELINE (CPU Kokoro + GPU F5)       ")
    print("=" * 72)


def is_url(target: str) -> bool:
    """Checks whether the given string is a valid web URL."""
    parsed = urllib.parse.urlparse(target.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def ingest_input_source(source: str, reader_mode: bool = True) -> str:
    """
    Ingests any input source (URL, ebook, or text file) and returns the path
    to the corresponding .txt file inside raw_txt/.
    """
    os.makedirs(RAW_TXT_DIR, exist_ok=True)
    source = source.strip().strip('"').strip("'")

    # Case 1: Web URL
    if is_url(source):
        print(f"\n[Step 1/4] Ingesting web URL via Playwright...")
        print(f"  URL: {source}")
        raw_txt_path = url_to_txt.scrape_url_to_txt(
            url=source,
            reader_mode=reader_mode,
            headless=True,
        )
        if not raw_txt_path or not os.path.exists(raw_txt_path):
            raise RuntimeError(f"Web scraping failed for URL: '{source}'")
        return raw_txt_path

    # Case 2: Local File Path
    if not os.path.exists(source):
        # Check if the filename exists inside raw_txt/ directly
        candidate = os.path.join(RAW_TXT_DIR, source)
        if os.path.exists(candidate):
            source = candidate
        else:
            raise FileNotFoundError(f"Input file or URL not found: '{source}'")

    ext = os.path.splitext(source)[1].lower()

    # Case 2a: Ebook format (.epub, .pdf, .mobi, .azw3)
    if ext in (".epub", ".pdf", ".mobi", ".azw3"):
        print(f"\n[Step 1/4] Converting ebook format '{ext}' to clean text...")
        raw_txt_path = ebook_to_txt.convert_file(source, apply_clean=True)
        if not raw_txt_path or not os.path.exists(raw_txt_path):
            raise RuntimeError(f"Ebook conversion failed for file: '{source}'")
        return raw_txt_path

    # Case 2b: Plain text file (.txt, .md)
    if ext in (".txt", ".md", ".text"):
        abs_source = os.path.abspath(source)
        abs_raw = os.path.abspath(RAW_TXT_DIR)
        # If already in raw_txt/, return directly
        if abs_source.startswith(abs_raw):
            print(f"\n[Step 1/4] Using raw text file: '{source}'")
            return source

        # Otherwise copy into raw_txt/ to keep workspace organized
        dest_filename = os.path.basename(source)
        dest_path = os.path.join(RAW_TXT_DIR, dest_filename)
        counter = 1
        stem, ext_part = os.path.splitext(dest_filename)
        while os.path.exists(dest_path) and os.path.abspath(dest_path) != abs_source:
            dest_path = os.path.join(RAW_TXT_DIR, f"{stem}_{counter}{ext_part}")
            counter += 1

        if os.path.abspath(dest_path) != abs_source:
            shutil.copyfile(source, dest_path)
            print(f"\n[Step 1/4] Imported '{source}' -> '{dest_path}'")
        else:
            print(f"\n[Step 1/4] Using raw text file: '{dest_path}'")
        return dest_path

    raise ValueError(f"Unsupported file type: '{ext}'. Supported formats: URL, .txt, .epub, .pdf, .mobi")


def interactive_select_source() -> str:
    """Prompts the user to enter a URL, file path, or choose from existing raw_txt files."""
    print("\nInput Source:")
    print("  Enter a web URL, ebook path (.epub/.pdf/.mobi), or text file path (.txt).")
    raw_files = [
        f for f in glob.glob(os.path.join(RAW_TXT_DIR, "*.txt"))
        if os.path.isfile(f) and not os.path.basename(f).startswith(".")
    ]

    if raw_files:
        print(f"  (Or press Enter to choose from {len(raw_files)} existing file(s) in '{RAW_TXT_DIR}/')")

    user_input = input("> ").strip()
    if user_input:
        return user_input

    if not raw_files:
        print(f"[!] No existing files found in '{RAW_TXT_DIR}/'.")
        return ""

    print(f"\nExisting files in '{RAW_TXT_DIR}/':")
    for idx, fpath in enumerate(raw_files, 1):
        bname = os.path.basename(fpath)
        sz_kb = os.path.getsize(fpath) / 1024
        print(f"  [{idx:2d}] {bname:<45} ({sz_kb:6.1f} KB)")

    sel = input(f"\nSelect file number [1-{len(raw_files)}] (default 1): ").strip()
    try:
        chosen_idx = int(sel) - 1 if sel else 0
        if 0 <= chosen_idx < len(raw_files):
            return raw_files[chosen_idx]
    except ValueError:
        pass
    return raw_files[0]


def run_full_pipeline(
    source: str,
    tag_dialogue: bool = True,
    tag_mode: str = "fast",
    enable_emotions: bool = True,
    trailing_mode: str = "none",
    bulk_cut_count: int = 0,
    should_unwrap: bool = False,
    should_convert_multi_nl: bool = False,
    skip_audio: bool = False,
    interactive: bool = False,
    normalize_volume: bool = None,
    force_rerecord_menu: bool = False,
    random_narrator: bool = None,
    narrator_gender: str = "any",
    narrator_accent: str = "any",
    narrator_voices: str = "2-4",
    save_random_narrator: bool = False,
    append_to_book: str = None,
    only_new: bool = None,
    select_chapters: str = None,
    reader_mode: bool = True,
    overwrite_existing: bool = None,
    ollama_model: str = None,
    ollama_url: str = None,
    review_tags: bool = None,
    list_tags: bool = False,
    set_tags: list = None,
    untag_specs: list = None,
    remap_tags: list = None,
    map_aliases: list = None,
    narrator_voice: str = None,
    set_volumes: list = None,
    adjust_volumes: bool = False,
    existing_action: str = None,
    rerecord_sections: str = None,
    list_chunks: bool = False,
    rerecord_chunks: list = None,
    set_chunk_voices: list = None,
    set_chunk_volumes: list = None,
    reset_chunks: list = None,
):
    """
    Executes all stages of the audiobook generator:
      Stage 1: Ingest URL / Ebook / Text
      Stage 2: Parse into chapters and write manifest
      Stage 3: Tag dialogue & emotions (Smart Parser) + optional tag review/edits
      Stage 4: Convert to audio (CPU Kokoro + GPU F5-TTS)
    """
    print_banner()

    # ---------------- STAGE 1: INGESTION ----------------
    raw_txt_path = ingest_input_source(source, reader_mode=reader_mode)
    if not os.path.exists(raw_txt_path):
        print(f"[X] Error: Could not resolve raw text path: {raw_txt_path}")
        return False

    # ---------------- STAGE 2: PARSING ----------------
    print(f"\n[Step 2/4] Parsing book text into formatted chapters...")
    replacements = ntp.load_replacements()

    created_chapter_files = ntp.parse_book_file(
        input_file_path=raw_txt_path,
        replacements=replacements,
        should_convert_multi_nl=should_convert_multi_nl,
        should_unwrap=should_unwrap,
        trailing_mode=trailing_mode,
        bulk_cut_count=bulk_cut_count,
        append_to_book=append_to_book,
        only_new=only_new,
        select_chapters=select_chapters,
        interactive=interactive,
        overwrite_existing=overwrite_existing,
    )

    if not created_chapter_files:
        print("[X] Parsing produced no chapter files. Aborting pipeline.")
        return False

    print(f"[✓] Successfully created {len(created_chapter_files)} chapter(s) in 'parsed_chapters/'.")

    # ---------------- STAGE 3: DIALOGUE TAGGING ----------------
    final_manifest_files = created_chapter_files
    if tag_dialogue:
        print(f"\n[Step 3/4] Running Dialogue Tagger ({tag_mode.upper()} mode, emotions={'ON' if enable_emotions else 'OFF'})...")
        diag_format = "hybrid" if tag_mode == "hybrid" else ("script" if tag_mode == "script" else "prose")

        tag_kwargs = {
            "file_paths": created_chapter_files,
            "dialogue_format": diag_format,
            "detect_tone": enable_emotions,
        }
        if ollama_model:
            tag_kwargs["model"] = ollama_model
        if ollama_url:
            tag_kwargs["url"] = ollama_url

        tagged_files = ldt.batch_tag_files(**tag_kwargs)

        if tagged_files:
            final_manifest_files = ntp.update_manifest_file(tagged_files)
            print(f"[✓] Dialogue tagging complete! Updated '{MANIFEST_FILE}' with {len(tagged_files)} tagged chapter(s) (preserved all chapters).")
        else:
            print("[!] Tagging produced no files; falling back to untagged parsed chapters.")
    else:
        print(f"\n[Step 3/4] Skipping dialogue tagging as requested. Manifest points to parsed chapters.")
        final_manifest_files = ntp.update_manifest_file(created_chapter_files)

    has_cli_tag_ops = bool(list_tags or set_tags or untag_specs or remap_tags or map_aliases)
    if has_cli_tag_ops:
        ldt.apply_cli_tag_operations(
            final_manifest_files,
            list_tags=list_tags,
            set_tags=set_tags,
            untag_specs=untag_specs,
            remap_tags=remap_tags,
            map_aliases=map_aliases,
        )

    if review_tags is True:
        ldt.interactive_review_dialogue_tags(final_manifest_files, force_open=True)
    elif review_tags is None and interactive and tag_dialogue:
        ldt.interactive_review_dialogue_tags(final_manifest_files, force_open=False)

    # ---------------- STAGE 4: AUDIO CONVERSION ----------------
    if skip_audio:
        print(f"\n[Step 4/4] Audio generation skipped (--skip-audio flag passed).")
        print(f"[*] Manifest is ready at '{MANIFEST_FILE}'. Run 'python local_hybrid_audioconverter.py' to generate audio later.")
        return True

    print(f"\n[Step 4/4] Starting Simultaneous Hybrid Audio Generation...")
    print(f"  • General Narration : CPU Kokoro (Neural Speech)")
    print(f"  • Dialogue / Clones : GPU F5-TTS (Flow-Matching)")
    print(f"  • Emotion Delivery  : {'ENABLED' if enable_emotions else 'DISABLED (Pure Baseline)'}")
    vol_desc = "Auto-Balanced" if normalize_volume is not False else "Raw (Per-Voice Multipliers Only)"
    print(f"  • Volume Mode       : {vol_desc}")

    success = lha.convert_audiobook(
        manifest_file=MANIFEST_FILE,
        enable_emotions=enable_emotions,
        interactive=interactive,
        normalize_volume=normalize_volume,
        force_rerecord_menu=force_rerecord_menu,
        random_narrator=random_narrator,
        narrator_gender=narrator_gender,
        narrator_accent=narrator_accent,
        narrator_voices=narrator_voices,
        save_random_narrator=save_random_narrator,
        narrator_voice=narrator_voice,
        set_volumes=set_volumes,
        adjust_volumes=adjust_volumes,
        review_tags=False,
        existing_action=existing_action,
        rerecord_sections=rerecord_sections,
        list_chunks=list_chunks,
        rerecord_chunks=rerecord_chunks,
        set_chunk_voices=set_chunk_voices,
        set_chunk_volumes=set_chunk_volumes,
        reset_chunks=reset_chunks,
    )

    if success:
        print("\n" + "=" * 72)
        print(" [✓] PIPELINE COMPLETE: Your audiobook is ready in the 'finished/' folder! ")
        print("=" * 72)
    else:
        print("\n[!] Audio generation encountered an error or was paused.")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Unified End-to-End Audiobook Pipeline Runner (Ingest -> Parse -> Tag -> Convert)."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="URL of web fiction or path to local file (.txt, .epub, .pdf, .mobi).",
    )
    parser.add_argument(
        "--reader",
        dest="reader_mode",
        action="store_true",
        default=True,
        help="Enable Reader Mode when scraping web URLs (default: True).",
    )
    parser.add_argument(
        "--no-reader",
        dest="reader_mode",
        action="store_false",
        help="Disable Reader Mode when scraping web URLs.",
    )
    parser.add_argument(
        "--tag",
        dest="tag_dialogue",
        action="store_true",
        default=None,
        help="Enable automatic dialogue tagging (default: enabled).",
    )
    parser.add_argument(
        "--no-tag",
        dest="tag_dialogue",
        action="store_false",
        help="Skip automatic dialogue tagging (renders clean narrator voice throughout).",
    )
    parser.add_argument(
        "--tag-mode",
        choices=["fast", "prose", "hybrid", "script"],
        default="fast",
        help="Dialogue tagging engine: 'fast'/'prose' (smart parser, default), 'hybrid' (LLM verification), 'script' (screenplay).",
    )
    parser.add_argument(
        "--model",
        dest="ollama_model",
        type=str,
        default=None,
        help="Ollama model name when --tag-mode=hybrid.",
    )
    parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        type=str,
        default=None,
        help="Ollama server URL when --tag-mode=hybrid.",
    )
    parser.add_argument(
        "--emotions",
        dest="enable_emotions",
        action="store_true",
        default=None,
        help="Enable acoustic emotion/tone modulation (default: enabled).",
    )
    parser.add_argument(
        "--no-emotions",
        dest="enable_emotions",
        action="store_false",
        help="Disable acoustic emotion/tone modulation (pure clean baseline voice profiles).",
    )
    parser.add_argument(
        "--review-tags", "--review",
        dest="review_tags",
        action="store_true",
        default=None,
        help="Interactively review and edit dialogue tags before audio conversion (when tags exist).",
    )
    parser.add_argument(
        "--no-review-tags",
        dest="review_tags",
        action="store_false",
        help="Skip interactive dialogue tag review prompt.",
    )
    parser.add_argument(
        "--list-tags", "--show-tags",
        action="store_true",
        default=False,
        help="Print numbered dialogue tags and context after tagging.",
    )
    parser.add_argument(
        "--set-tag", "--change-tag", "--edit-tag",
        dest="set_tags",
        action="append",
        default=[],
        help="Change dialogue tag(s), e.g. '1=Alice' or '2,4-6=Bob:whisper'.",
    )
    parser.add_argument(
        "--untag",
        dest="untag_specs",
        action="append",
        default=[],
        help="Remove dialogue tag(s) so Narrator reads them, e.g. '3,5'.",
    )
    parser.add_argument(
        "--remap-tag", "--rename-tag",
        dest="remap_tags",
        action="append",
        default=[],
        help="Bulk remap speaker tag across chapters, e.g. 'OldName=NewName'.",
    )
    parser.add_argument(
        "--map-alias",
        dest="map_aliases",
        action="append",
        default=[],
        help="Map tag alias in tag_mappings.json, e.g. 'bobby=Bob'.",
    )
    parser.add_argument(
        "--trailing",
        dest="trailing_mode",
        choices=["none", "interactive", "bulk"],
        default="none",
        help="Trailing sections cutting mode for author notes (default: 'none').",
    )
    parser.add_argument(
        "--bulk-cut",
        type=int,
        default=0,
        help="Number of trailing sections to cut from each chapter when --trailing=bulk.",
    )
    parser.add_argument(
        "--unwrap",
        dest="unwrap",
        action="store_true",
        default=False,
        help="Heal hard-wrapped lines into continuous paragraphs.",
    )
    parser.add_argument(
        "--no-unwrap",
        dest="unwrap",
        action="store_false",
        help="Do not unwrap hard-wrapped lines.",
    )
    parser.add_argument(
        "--multi-nl",
        dest="multi_nl",
        action="store_true",
        default=False,
        help="Convert multiple consecutive blank lines into '***' scene divider pauses.",
    )
    parser.add_argument(
        "--no-multi-nl",
        dest="multi_nl",
        action="store_false",
        help="Keep multiple consecutive blank lines as regular breaks.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        default=False,
        help="Only perform ingestion, parsing, and tagging without rendering audio.",
    )
    parser.add_argument(
        "--volume-balance", "--auto-balance",
        dest="normalize_volume",
        action="store_true",
        default=None,
        help="Enable automatic active-speech loudness balancing (default).",
    )
    parser.add_argument(
        "--no-volume-balance", "--raw-volume",
        dest="normalize_volume",
        action="store_false",
        help="Disable automatic active-speech loudness balancing (keeps explicit per-voice 'volume' multipliers).",
    )
    parser.add_argument(
        "--set-volume", "--volume",
        dest="set_volumes",
        action="append",
        default=[],
        help="Update character/variant volume in characters.json, e.g. 'Alice=1.2'.",
    )
    parser.add_argument(
        "--adjust-volumes",
        action="store_true",
        default=False,
        help="Open interactive voice volume editor before rendering.",
    )
    parser.add_argument(
        "--rerecord",
        action="store_true",
        default=False,
        help="Open chunk-level re-record / voice switching menu before rendering audio.",
    )
    parser.add_argument(
        "--narrator-voice",
        type=str,
        default=None,
        help="Explicit Kokoro voice or blend string for narrator.",
    )
    parser.add_argument(
        "--random-narrator",
        dest="random_narrator",
        action="store_true",
        default=None,
        help="Randomize the narrator voice using random Kokoro TTS voices.",
    )
    parser.add_argument(
        "--keep-narrator", "--no-random-narrator",
        dest="random_narrator",
        action="store_false",
        help="Keep existing narrator voice from characters.json without prompting.",
    )
    parser.add_argument(
        "--narrator-gender",
        choices=["female", "male", "mostly-female", "mostly-male", "any"],
        default=None,
        help="Gender preference for randomized Kokoro narrator.",
    )
    parser.add_argument(
        "--narrator-accent",
        choices=["american", "british", "any"],
        default=None,
        help="Regional accent preference for randomized Kokoro narrator.",
    )
    parser.add_argument(
        "--narrator-voices",
        type=str,
        default=None,
        help="Number or range of Kokoro voices to mix (e.g. '3' or '2-4').",
    )
    parser.add_argument(
        "--save-narrator",
        dest="save_narrator",
        action="store_true",
        default=False,
        help="Save randomized narrator voice into characters.json.",
    )
    parser.add_argument(
        "--no-save-narrator",
        dest="save_narrator",
        action="store_false",
        help="Use randomized narrator for this run only.",
    )
    parser.add_argument(
        "--append-to",
        "--append",
        "--add-to",
        dest="append_to_book",
        default=None,
        help="Append new chapter(s) to an existing book (preserves previous chapters and tracks).",
    )
    parser.add_argument(
        "--new-book",
        action="store_true",
        default=False,
        help="Create a new book without prompting for destination.",
    )
    parser.add_argument(
        "--chapters",
        "--chaps",
        dest="select_chapters",
        default=None,
        help="Specific chapter numbers or ranges to process/append (e.g. '5', '5-8', '5+').",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        default=None,
        help="When appending an updated full work, automatically extract only newly posted chapters.",
    )
    parser.add_argument(
        "--all-chapters",
        action="store_true",
        default=False,
        help="Force extracting all chapters even if previous chapters exist.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite_existing",
        action="store_true",
        default=None,
        help="Overwrite existing chapters when re-parsing duplicates.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite_existing",
        action="store_false",
        help="Keep existing chapters when duplicates are found.",
    )
    parser.add_argument(
        "--existing-action", "--on-existing",
        dest="existing_action",
        choices=["resume", "rerecord-all", "rerecord-sections", "rerecord-chunks"],
        default=None,
        help="Action when existing audio is detected.",
    )
    parser.add_argument(
        "--resume",
        dest="existing_action",
        action="store_const",
        const="resume",
        help="Resume existing audio without prompting.",
    )
    parser.add_argument(
        "--rerecord-all",
        dest="existing_action",
        action="store_const",
        const="rerecord-all",
        help="Clear and re-record all chapters from scratch.",
    )
    parser.add_argument(
        "--rerecord-sections", "--sections",
        dest="rerecord_sections",
        type=str,
        default=None,
        help="Specific sections to re-record or target for chunk operations.",
    )
    parser.add_argument(
        "--list-chunks", "--show-chunks",
        action="store_true",
        default=False,
        help="Print numbered audio chunks before rendering.",
    )
    parser.add_argument(
        "--rerecord-chunks",
        dest="rerecord_chunks",
        action="append",
        default=[],
        help="Re-record specific chunks, e.g. '2,5-7'.",
    )
    parser.add_argument(
        "--set-chunk-voice", "--chunk-voice",
        dest="set_chunk_voices",
        action="append",
        default=[],
        help="Override chunk voice, e.g. '2,3=Bob:whisper@1.1'.",
    )
    parser.add_argument(
        "--set-chunk-volume", "--chunk-volume",
        dest="set_chunk_volumes",
        action="append",
        default=[],
        help="Override chunk volume multiplier, e.g. '2,3=1.25'.",
    )
    parser.add_argument(
        "--reset-chunks",
        dest="reset_chunks",
        action="append",
        default=[],
        help="Reset chunk voice/volume overrides.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=False,
        help="Prompt interactively for unspecified options.",
    )
    parser.add_argument(
        "--yes", "-y", "--non-interactive",
        dest="non_interactive",
        action="store_true",
        default=False,
        help="Run non-interactively using defaults for all unspecified options.",
    )

    args = parser.parse_args()
    if args.random_narrator is True or any(
        val is not None for val in (args.narrator_gender, args.narrator_accent, args.narrator_voices)
    ):
        use_random_narrator = True
    elif args.random_narrator is False:
        use_random_narrator = False
    else:
        use_random_narrator = None

    source = args.source
    is_interactive = bool(args.interactive)
    if not source:
        if not args.non_interactive:
            is_interactive = True
        print_banner()
        source = interactive_select_source()
        if not source:
            print("[!] No input source provided. Exiting.")
            sys.exit(0)

    if is_interactive:
        print("\nPipeline Configuration (Press Enter for recommended defaults):")
        if args.tag_dialogue is None:
            tag_ans = input("  Tag dialogue with character voices? [Y/n] (default Y): ").strip().lower()
            tag_dialogue = (tag_ans != "n")
        else:
            tag_dialogue = args.tag_dialogue

        if args.enable_emotions is None:
            emo_ans = input("  Enable tone & emotion modulation? [Y/n] (default Y): ").strip().lower()
            enable_emotions = (emo_ans != "n")
        else:
            enable_emotions = args.enable_emotions
    else:
        tag_dialogue = True if args.tag_dialogue is None else args.tag_dialogue
        enable_emotions = True if args.enable_emotions is None else args.enable_emotions

    existing_books = ntp.get_existing_books(ntp.OUTPUT_DIR, ldt.DEFAULT_OUTPUT_DIR, FINISHED_DIR)
    append_to_book = None
    if args.append_to_book:
        resolved = ntp.resolve_book_name(args.append_to_book, existing_books)
        append_to_book = resolved if resolved else ntp.sanitize_filename(args.append_to_book)
        print(f"[*] Appending to existing book: '{append_to_book}'")
    elif is_interactive and not args.new_book:
        print("\nTarget Book Destination:")
        print("  [1] Create a new book [Default]")
        print("  [2] Add chapter(s) to an existing book (select from list or enter title)")
        dest_sel = input("Select destination [1 or 2] (default 1): ").strip()
        if dest_sel == "2":
            append_to_book = ntp.interactive_select_existing_book(existing_books)
            print(f"[*] Appending to existing book: '{append_to_book}'")

    only_new = False if args.all_chapters else (True if args.only_new else None)

    run_full_pipeline(
        source=source,
        tag_dialogue=tag_dialogue,
        tag_mode=args.tag_mode,
        enable_emotions=enable_emotions,
        trailing_mode=args.trailing_mode,
        bulk_cut_count=args.bulk_cut,
        should_unwrap=args.unwrap,
        should_convert_multi_nl=args.multi_nl,
        skip_audio=args.skip_audio,
        interactive=is_interactive,
        normalize_volume=args.normalize_volume,
        force_rerecord_menu=args.rerecord,
        random_narrator=use_random_narrator,
        narrator_gender=args.narrator_gender or "any",
        narrator_accent=args.narrator_accent or "any",
        narrator_voices=args.narrator_voices or "2-4",
        save_random_narrator=args.save_narrator,
        append_to_book=append_to_book,
        only_new=only_new,
        select_chapters=args.select_chapters,
        reader_mode=args.reader_mode,
        overwrite_existing=args.overwrite_existing,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        review_tags=args.review_tags,
        list_tags=args.list_tags,
        set_tags=args.set_tags,
        untag_specs=args.untag_specs,
        remap_tags=args.remap_tags,
        map_aliases=args.map_aliases,
        narrator_voice=args.narrator_voice,
        set_volumes=args.set_volumes,
        adjust_volumes=args.adjust_volumes,
        existing_action=args.existing_action,
        rerecord_sections=args.rerecord_sections,
        list_chunks=args.list_chunks,
        rerecord_chunks=args.rerecord_chunks,
        set_chunk_voices=args.set_chunk_voices,
        set_chunk_volumes=args.set_chunk_volumes,
        reset_chunks=args.reset_chunks,
    )


if __name__ == "__main__":
    main()
