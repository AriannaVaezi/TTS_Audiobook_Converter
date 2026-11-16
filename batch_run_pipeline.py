"""
batch_run_pipeline.py
=====================
Runs the unified audiobook pipeline (run_pipeline.py) sequentially across:
  - A batch text file of URLs, .epub/.txt/.pdf paths, or book names (one per line)
  - A folder of .epub / .txt / .pdf files (e.g. raw_epub/ or raw_txt/)
  - One or more explicit URLs / file paths passed on the command line

Supports all CLI options of run_pipeline.py (--no-tags, --new-book, --keep-narrator,
--random-narrator, --visible, --login, --cookies, --epub, --skip-audio, etc.).

Usage:
  python batch_run_pipeline.py url_file\\urls.txt --no-tags --new-book
  python batch_run_pipeline.py raw_epub --no-tags --new-book --keep-narrator
  python batch_run_pipeline.py urls.txt --visible --format epub --no-tags --new-book
"""

import argparse
import os
import sys

# Ensure UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import run_pipeline

SUPPORTED_BOOK_EXTS = (".epub", ".txt", ".pdf", ".mobi", ".azw3")


def _looks_like_source_entry(line: str) -> bool:
    """Heuristically distinguishes a URL / file path / filename line from story prose."""
    cleaned = line.strip().strip('"').strip("'")
    if not cleaned:
        return False
    if run_pipeline.is_url(cleaned):
        return True
    if os.path.exists(cleaned):
        return True
    ext = os.path.splitext(cleaned)[1].lower()
    if ext in SUPPORTED_BOOK_EXTS:
        return True
    if "/" in cleaned or "\\" in cleaned:
        return True
    return False


def _read_batch_list_file(list_file_path: str) -> list[str]:
    """Reads non-empty, non-comment target lines from a batch list file."""
    items = []
    with open(list_file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            items.append(stripped.strip('"').strip("'"))
    return items


def resolve_batch_items(inputs: list[str]) -> list[str]:
    """
    Expands CLI positional inputs (batch list files, folders, URLs, or book paths/stems)
    into an ordered list of source targets for run_pipeline.
    """
    if not inputs:
        inputs = ["urls.txt"]

    resolved: list[str] = []
    seen: set[str] = set()

    def _add(target: str):
        norm = target.strip()
        if norm and norm not in seen:
            seen.add(norm)
            resolved.append(norm)

    abs_raw_txt = os.path.abspath(run_pipeline.RAW_TXT_DIR)

    for raw_inp in inputs:
        item = raw_inp.strip().strip('"').strip("'")
        if not item:
            continue

        if run_pipeline.is_url(item):
            _add(item)
            continue

        if os.path.isdir(item):
            dir_files = sorted(
                os.path.join(item, fname)
                for fname in os.listdir(item)
                if os.path.isfile(os.path.join(item, fname))
                and not fname.startswith(".")
                and fname.lower().endswith(SUPPORTED_BOOK_EXTS)
            )
            for fpath in dir_files:
                _add(fpath)
            continue

        if os.path.isfile(item):
            ext = os.path.splitext(item)[1].lower()
            if ext in (".epub", ".pdf", ".mobi", ".azw3"):
                _add(item)
                continue

            # Inside raw_txt/, treat prose .txt files as a single book unless lines look like URLs/paths
            abs_item = os.path.abspath(item)
            in_raw_txt = abs_item == abs_raw_txt or abs_item.startswith(abs_raw_txt + os.sep)
            entries = _read_batch_list_file(item)
            if in_raw_txt and entries and not any(_looks_like_source_entry(e) for e in entries[:5]):
                _add(item)
            else:
                for entry in entries:
                    _add(entry)
            continue

        # Fallback: resolve stem/filename inside raw_txt/ or raw_epub/
        matched = run_pipeline._resolve_local_source_path(item)
        _add(matched)

    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch runner for run_pipeline.py across a URL/file list or folder of books (.epub, .txt, .pdf)."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["urls.txt"],
        help="Path(s) to a batch .txt list file (e.g. url_file\\urls.txt), a folder (e.g. raw_epub), or individual URLs/files.",
    )
    run_pipeline.add_pipeline_arguments(parser, include_source=False)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        items = resolve_batch_items(args.inputs)
    except FileNotFoundError as e:
        print(f"[X] {e}")
        return 1

    if not items:
        print("[!] No valid URLs or book files found to process.")
        return 1

    print("=" * 72)
    print(f"       BATCH AUDIOBOOK PIPELINE RUNNER ({len(items)} item(s))")
    print("=" * 72)
    print(f"  Tag dialogue : {'OFF' if args.tag_dialogue is False else 'ON'}")
    print(f"  New book     : {'YES' if args.new_book else ('APPEND: ' + args.append_to_book if args.append_to_book else 'Default')}")
    print(f"  Skip audio   : {'YES' if args.skip_audio else 'NO'}")
    print("=" * 72 + "\n")

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    first_url_login_used = False

    for idx, item in enumerate(items, 1):
        print(f"\n[Batch {idx}/{len(items)}] Starting: {item}")
        print("-" * 72)
        use_login = False
        if args.interactive_login and run_pipeline.is_url(item) and not first_url_login_used:
            use_login = True
            first_url_login_used = True

        try:
            ok = run_pipeline.run_from_args(
                args,
                source_override=item,
                interactive_login_override=use_login,
            )
            if ok:
                succeeded.append(item)
                print(f"[+] Batch item {idx}/{len(items)} finished: {item}")
            else:
                failed.append((item, "Pipeline returned False / aborted"))
                print(f"[-] Batch item {idx}/{len(items)} did not complete: {item}")
        except Exception as e:
            failed.append((item, str(e)))
            print(f"[X] Error processing '{item}': {e}")

    print("\n" + "=" * 72)
    print("Batch Pipeline Summary")
    print("=" * 72)
    print(f"  Successful : {len(succeeded)}/{len(items)}")
    print(f"  Failed     : {len(failed)}/{len(items)}")
    if failed:
        print("\nFailed items:")
        for target, reason in failed:
            print(f"  - {target} ({reason})")
    print("=" * 72)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
