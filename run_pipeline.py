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
    normalize_volume: bool = True,
    force_rerecord_menu: bool = False,
    random_narrator: bool = False,
    narrator_gender: str = "any",
    narrator_accent: str = "any",
    narrator_voices: str = "2-4",
    save_random_narrator: bool = False,
):
    """
    Executes all stages of the audiobook generator:
      Stage 1: Ingest URL / Ebook / Text
      Stage 2: Parse into chapters and write manifest
      Stage 3: Tag dialogue & emotions (Smart Parser)
      Stage 4: Convert to audio (CPU Kokoro + GPU F5-TTS)
    """
    print_banner()

    # ---------------- STAGE 1: INGESTION ----------------
    raw_txt_path = ingest_input_source(source, reader_mode=True)
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
    )

    if not created_chapter_files:
        print("[X] Parsing produced no chapter files. Aborting pipeline.")
        return False

    print(f"[✓] Successfully created {len(created_chapter_files)} chapter(s) in 'parsed_chapters/'.")

    # ---------------- STAGE 3: DIALOGUE TAGGING ----------------
    final_manifest_files = created_chapter_files
    if tag_dialogue:
        print(f"\n[Step 3/4] Running Dialogue Tagger ({tag_mode.upper()} mode, emotions={'ON' if enable_emotions else 'OFF'})...")
        diag_format = "script" if tag_mode == "script" else "prose"
        use_llm = (tag_mode == "hybrid")

        tagged_files = ldt.batch_tag_files(
            file_paths=created_chapter_files,
            dialogue_format=diag_format,
            detect_tone=enable_emotions,
        )

        if tagged_files:
            final_manifest_files = tagged_files
            with open(MANIFEST_FILE, "w", encoding="utf-8") as mf:
                for tf in tagged_files:
                    mf.write(tf + "\n")
            print(f"[✓] Dialogue tagging complete! Updated '{MANIFEST_FILE}' with {len(tagged_files)} tagged chapter(s).")
        else:
            print("[!] Tagging produced no files; falling back to untagged parsed chapters.")
    else:
        print(f"\n[Step 3/4] Skipping dialogue tagging as requested. Manifest points to parsed chapters.")
        with open(MANIFEST_FILE, "w", encoding="utf-8") as mf:
            for cf in created_chapter_files:
                mf.write(cf + "\n")

    # ---------------- STAGE 4: AUDIO CONVERSION ----------------
    if skip_audio:
        print(f"\n[Step 4/4] Audio generation skipped (--skip-audio flag passed).")
        print(f"[*] Manifest is ready at '{MANIFEST_FILE}'. Run 'python local_hybrid_audioconverter.py' to generate audio later.")
        return True

    print(f"\n[Step 4/4] Starting Simultaneous Hybrid Audio Generation...")
    print(f"  • General Narration : CPU Kokoro (Neural Speech)")
    print(f"  • Dialogue / Clones : GPU F5-TTS (Flow-Matching)")
    print(f"  • Emotion Delivery  : {'ENABLED' if enable_emotions else 'DISABLED (Pure Baseline)'}")
    print(f"  • Volume Mode       : {'Auto-Balanced' if normalize_volume else 'Raw (Per-Voice Multipliers Only)'}")

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
        "--no-tag",
        dest="tag_dialogue",
        action="store_false",
        default=True,
        help="Skip automatic dialogue tagging (renders clean narrator voice throughout).",
    )
    parser.add_argument(
        "--tag-mode",
        choices=["fast", "hybrid", "script"],
        default="fast",
        help="Dialogue tagging engine: 'fast' (Instant smart parser, default), 'hybrid' (LLM verification), 'script' (screenplay).",
    )
    parser.add_argument(
        "--no-emotions",
        dest="enable_emotions",
        action="store_false",
        default=True,
        help="Disable acoustic emotion/tone modulation (pure clean baseline voice profiles).",
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
        action="store_true",
        default=False,
        help="Heal hard-wrapped lines into continuous paragraphs.",
    )
    parser.add_argument(
        "--multi-nl",
        action="store_true",
        default=False,
        help="Convert multiple consecutive blank lines into '***' scene divider pauses.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        default=False,
        help="Only perform ingestion, parsing, and tagging without rendering audio.",
    )
    parser.add_argument(
        "--no-volume-balance",
        dest="normalize_volume",
        action="store_false",
        default=True,
        help="Disable automatic active-speech loudness balancing (keeps explicit per-voice 'volume' multipliers).",
    )
    parser.add_argument(
        "--rerecord",
        action="store_true",
        default=False,
        help="Open chunk-level re-record / voice switching menu before rendering audio.",
    )
    parser.add_argument(
        "--random-narrator",
        action="store_true",
        default=False,
        help="Randomize the narrator voice using random Kokoro TTS voices.",
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
        action="store_true",
        default=False,
        help="Save randomized narrator voice into characters.json.",
    )

    args = parser.parse_args()
    use_random_narrator = args.random_narrator or any(
        val is not None for val in (args.narrator_gender, args.narrator_accent, args.narrator_voices)
    )

    source = args.source
    is_interactive = False
    if not source:
        is_interactive = True
        print_banner()
        source = interactive_select_source()
        if not source:
            print("[!] No input source provided. Exiting.")
            sys.exit(0)

        # Quick interactive preference checks with smart defaults
        print("\nPipeline Configuration (Press Enter for recommended defaults):")
        tag_ans = input("  Tag dialogue with character voices? [Y/n] (default Y): ").strip().lower()
        tag_dialogue = (tag_ans != "n")

        emo_ans = input("  Enable tone & emotion modulation? [Y/n] (default Y): ").strip().lower()
        enable_emotions = (emo_ans != "n")
    else:
        tag_dialogue = args.tag_dialogue
        enable_emotions = args.enable_emotions

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
        interactive=is_interactive or args.rerecord,
        normalize_volume=args.normalize_volume,
        force_rerecord_menu=args.rerecord,
        random_narrator=use_random_narrator,
        narrator_gender=args.narrator_gender or "any",
        narrator_accent=args.narrator_accent or "any",
        narrator_voices=args.narrator_voices or "2-4",
        save_random_narrator=args.save_narrator,
    )


if __name__ == "__main__":
    main()
