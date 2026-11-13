# Hybrid Multi-Voice Audiobook Generator

An automated text-to-speech (TTS) pipeline designed for creating audiobooks with dynamic, multi-voice narration. It combines the speed and clarity of **Kokoro TTS running 100% locally on CPU** for general narration with the voice-cloning capabilities of **F5-TTS running on GPU** for specific characters and dialogue.

---

## Key Features

* **Hybrid Simultaneous TTS Engine (`local_hybrid_audioconverter.py`)**:
  * **Kokoro TTS (100% Local CPU)**: Fast, natural, high-fidelity neural narration for general book text and secondary characters (no Docker, no server, and no cloud API required; supports 50/50 and custom weighted voice blending).
  * **F5-TTS (Flow-Matching on GPU)**: Zero-shot voice cloning for custom characters using a clean 5–10 second reference audio sample.
  * **Simultaneous Parallel Rendering**: CPU Kokoro and GPU F5-TTS render concurrently on separate threads for maximum generation throughput.
* **Unified All-In-One Pipeline (`run_pipeline.py`)**:
  * Automates the full process in a single command: supply a web URL, ebook (`.epub`, `.pdf`, `.mobi`), or text file, and it handles ingestion, chapter parsing, dialogue tagging, and audio conversion end-to-end.
* **Intelligent Text Parser (`new_text_parser.py`)**:
  * Automatically detects chapters (`Chapter 1`, `Chapter 1: Title`).
  * Extracts clean title and author metadata.
  * Strips fanfiction/AO3 metadata, author notes, chapter summaries, and end notes.
  * Optionally heals manual hard-wrapped line breaks while preserving dialogue and section dividers.
  * Interactive or bulk trailing section removal (with live text excerpts).
  * **Seamless Chapter Appending**: Add new chapters to existing books with continuous sequential track prefixes and chapter numbers without recreating intros or wiping existing manifests.
* **Smart Dialogue & Emotion Tagger (`llm_dialogue_tagger.py`)**:
  * **Fast Smart Parser (Instant & 100% Quote Coverage)**: Grammar-aware speech verb detection and conversational turn-taking tags entire books in seconds with zero GPU wait.
  * **Dialogue Emotion Routing**: Detects conversational tones (whisper, shout, calm, sad, etc.) and routes them to character emotion reference clips with zero synthetic distortion.
* **Robust Audio Generation & Production**:
  * **Per-Voice Volume Control & Alt Voices**: Configure explicit `"volume"` multipliers and named `"alts"` / `"alt_voices"` in `characters.json`, toggle auto-balanced vs. raw volume at startup (or edit volumes interactively), and trigger alt voices via `[Character:alt_name]`.
  * **Chunk-Level Re-Record & Voice Switching**: Scans sections at launch to resume, re-record sections, or pick exact chunk numbers to re-record, switch to another character, switch to the same character's emotion/alt voice, or tune chunk volume without losing untouched chunks.
  * **Automatic Archiving**: Completed audiobook folders are automatically moved to a dedicated `finished/` directory.
  * **Timestamp Tracking**: Generates an absolute timestamp log (`_timestamps.txt`) and master production voice roster (`_voice_roster.txt`).
  * **Automatic Packaging**: Combines all finished chapter MP3s into a ZIP archive.

---

## Project Structure

```text
TTS_Audiobook_Converter/
├── run_pipeline.py                 # ⭐ UNIFIED PIPELINE: Ingest -> Parse -> Tag -> Convert (All-in-One)
│
├── Ingestion Tools:
│   ├── url_to_txt.py               # Playwright web scraper for web fiction & chapters (.txt or .epub)
│   ├── batch_url_to_txt.py         # Batch URL scraper from a list of web links (.txt or .epub)
│   └── ebook_to_txt.py             # Ebook converter for EPUB, MOBI, PDF, AZW3 to raw text
│
├── Processing Tools:
│   ├── new_text_parser.py          # Cleans and splits raw text into formatted chapter files
│   ├── llm_dialogue_tagger.py      # Dialogue tagger & tone/emotion detector (Fast Smart Parser & LLM)
│   └── local_hybrid_audioconverter.py # Simultaneous CPU Kokoro + GPU F5-TTS audio generator
│
├── Workspace & Storage:
│   ├── raw_txt/                    # Source story/book text files
│   ├── parsed_chapters/            # Chapters split and cleaned by new_text_parser.py
│   ├── tagged_chapters/            # Chapters tagged with character and emotion markers
│   ├── wavs/                       # Custom reference voice WAV samples for F5-TTS cloning
│   ├── finished/                   # Automatically archived completed audiobooks & ZIP packages
│   ├── test_code/                  # Diagnostic and audio verification test scripts
│   └── old_code/                   # Archived legacy scripts (e.g. deprecated hybrid_audioconverter.py)
│
├── Configuration:
│   ├── characters.json             # Character-to-voice mapping configuration (auto-generated)
│   ├── tag_mappings.json           # Character nickname and alias mapping dictionary (auto-generated)
│   ├── replacements.json           # Phonetic word pronunciation replacement dictionary (auto-generated)
│   ├── chapters.txt                # Chapter manifest file processed by the audio generator
│   └── requirements.txt            # Python dependencies
└── README.md
```

---

## Prerequisites

1. **Python 3.10 or 3.11** *(Recommended; avoid Python 3.13+ for PyTorch/F5-TTS compatibility)*.
2. **FFmpeg**: Required for audio slicing, concatenation, and encoding.
   * [Download FFmpeg](https://ffmpeg.org/download.html) and ensure `ffmpeg` is added to your system `PATH`.
3. **NVIDIA GPU with CUDA** *(Recommended for fast F5-TTS voice cloning; Kokoro runs 100% locally on CPU)*.

---

## Installation

### Option 1: One-Click Setup (Windows)
Double-click `setup_env.bat` in the project root. This will:
1. Create a dedicated Python 3.11 virtual environment (`.venv`).
2. Automatically install all dependencies from `requirements.txt`.

### Option 2: Manual Setup
Open PowerShell or your terminal in this directory:

```powershell
# 1. Create a Python 3.11 virtual environment
uv venv --seed --python 3.11 .venv
# (or: python -m venv .venv)

# 2. Activate the virtual environment
.\.venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt --index-strategy unsafe-best-match
# (or: pip install -r requirements.txt)
```

---

## Quick Start: The Unified Pipeline (`run_pipeline.py`)

The fastest way to generate an audiobook is with the unified runner. Point it to any web link, ebook, or local text file:

```powershell
# From a web fiction / chapter URL:
python run_pipeline.py "https://archiveofourown.org/works/12345"

# From a local text file:
python run_pipeline.py raw_txt/my_story.txt

# From an EPUB or PDF ebook:
python run_pipeline.py "C:\Books\my_book.epub"

# Append new chapter(s) to an existing book (renders only new chapters & updates the full audiobook):
python run_pipeline.py "https://archiveofourown.org/works/12345/chapters/67890" --append-to "My_Story_Title"
python run_pipeline.py raw_txt/chapter_5.txt --append-to "My_Story_Title"
# Passing an updated full work (with chapters 1-5) will automatically extract only the new Chapter 5:
python run_pipeline.py "https://archiveofourown.org/works/12345" --append-to "My_Story_Title" --only-new
# Append specific chapter numbers or ranges to any custom or differently named book:
python run_pipeline.py raw_txt/story.txt --append-to "Custom_Book_Title" --chapters "5-7"

# Interactive Menu (prompts to create new book or add to existing book):
python run_pipeline.py
```

The pipeline automatically handles:
1. **Ingestion**: Scrapes the URL or converts the ebook into `raw_txt/`.
2. **Parsing**: Cleans headers/author notes and splits into `parsed_chapters/`.
3. **Tagging**: Fast Smart Parser tags character quotes and emotional delivery into `tagged_chapters/`.
4. **Conversion**: Local CPU Kokoro and GPU F5-TTS synthesize audio in parallel, merge MP3s, generate timestamps, and move the finished audiobook into `finished/`.

---

## Advanced Modular Workflow

For full control over each stage, you can also run the modular tools independently:

### Step 1: Ingest Source Material
* **Web URL**:
  ```powershell
  python url_to_txt.py "https://example.com/story/chapter1" --reader
  python url_to_txt.py "https://example.com/story/chapter1" --reader --epub
  ```
* **Ebook File**:
  ```powershell
  python ebook_to_txt.py "C:\Books\sample.epub"
  ```
Clean text is automatically saved into `raw_txt/` (and EPUBs into `raw_epub/`).

### Step 2: Parse and Clean Chapters (`new_text_parser.py`)
```powershell
python new_text_parser.py
# Or append to an existing book directly via CLI:
python new_text_parser.py raw_txt/chapter_6.txt --append-to "My_Book_Title"
```
* **Interactive Prompts**:
  1. **File Selection**: Choose which file from `raw_txt/` to process.
  2. **Book Destination**: Choose between creating a new book or adding chapter(s) to an existing book.
  3. **Multi-Newline Dividers**: Convert consecutive blank lines into `***` scene pauses.
  4. **Hard-Wrap Healing**: Merge lines split by manual line breaks into natural paragraphs.
  5. **Trailing Section Review**: Review trailing sections chapter-by-chapter (with live excerpts) to remove author notes.
* **Adding Chapters to an Existing Book**:
  * **Select Any Book Destination**: In interactive mode, select from a numbered list of existing books in your workspace, or choose `[0]` to type any custom/alternate book title. Or use `--append-to "Custom_Title"` via CLI.
  * **Selective Chapter Appending (`--chapters`)**: Pass specific chapter numbers or ranges via `--chapters "5"`, `--chapters "5-8"`, or `--chapters "5+"`, or pick them interactively from the chapter selection menu.
  * **Seamless Continuous Numbering**: New chapters are automatically numbered with continuous track prefixes (`05_`, `06_`) and chapter headers without recreating `00_intro`.
  * **Updated Full Work Support (`--only-new`)**: If an author posts a new chapter and you pass the full updated story (e.g. Chapters 1–5), the parser detects that Chapters 1–4 already exist in your book, prompts to add only the new Chapter 5, and extracts only Chapter 5.
  * `chapters.txt` retains all previous chapters while appending newly added chapters.
  * When generating audio, `local_hybrid_audioconverter.py` automatically detects existing `_complete.mp3` chapter files and skips re-rendering them, generating only newly appended chapters and updating the complete audiobook package and ZIP.
* Parsed chapters are saved to `parsed_chapters/<BookTitle>/` and registered in `chapters.txt`.

### Step 3: Configure Character Voices (`characters.json`)
The voice configuration file maps character names to specific TTS engines and voice profiles:

```json
{
  "narrator": {
    "engine": "kokoro",
    "voice": "af_bella(0.7)+af_nicole(0.3)",
    "pitch": 1.0,
    "speed": 1.0,
    "volume": 1.0,
    "alts": {
      "storyteller": "bm_george"
    },
    "description": "Default narrator on CPU for untagged story text."
  },
  "detective": {
    "engine": "kokoro",
    "voice": "am_michael",
    "pitch": 1.0,
    "speed": 1.0,
    "volume": 1.0
  },
  "villain": {
    "engine": "f5tts",
    "ref_audio": "wavs/villain_sample.wav",
    "ref_text": "The exact words spoken in the villain reference clip.",
    "speed": 1.0,
    "volume": 1.05,
    "steps": 16,
    "emotions": {
      "shout": "wavs/villain_shouting.wav",
      "whisper": "wavs/villain_whisper.wav"
    },
    "alts": {
      "calm_disguise": {
        "ref_audio": "wavs/villain_alt.wav",
        "ref_text": "An alternate calm vocal take for this character.",
        "volume": 0.95
      }
    }
  }
}
```

#### Voice Configuration Guidelines:
* **`narrator`**: The default fallback. All general story text without dialogue tags uses this voice.
* **`volume`**: Optional loudness multiplier (`1.0` default; `<1.0` quieter, `>1.0` louder). Works per-character and per-emotion/alt entry. Can also be edited interactively at startup.
* **Kokoro Voice Blending (CPU)**:
  * Single voice: `"af_bella"`
  * Equal blend: `"af_bella+af_nicole"`
  * Weighted blend: `"af_bella(0.6)+af_nicole(0.2)+am_michael(0.2)"`
* **F5-TTS Voice Cloning (GPU)**:
  * `ref_audio`: Path to a clean **5–10 second** WAV audio clip of the target voice.
  * `ref_text`: The **exact verbatim transcript** of what is spoken in `ref_audio`.
  * `steps`: Flow-matching diffusion steps (default: `16` for fast, crisp audio; `32` for high steps).
  * `emotions`: Optional dictionary mapping emotions (`shout`, `whisper`, `sad`) to dedicated reference clips (disabled when `--no-emotions` is set).
  * `alts` (or `alt_voices`): Optional dictionary mapping custom alternate voices (usable via `[Villain:calm_disguise]` or the chunk re-record menu; remains active even when `--no-emotions` is used).

### Step 4: Tag Dialogue (`llm_dialogue_tagger.py`)
```powershell
python llm_dialogue_tagger.py
```
* **[1] Fast Smart Parser (Instant & 100% Quote Coverage - Recommended)**:
  * Grammar-aware speech verb attribution and conversational turn-taking.
  * Tags multi-quote lines accurately (`[Hero] "Wait!" [/Hero] he said. [Hero] "Look there." [/Hero]`).
  * Tags thousands of words in **less than 0.1 seconds**.
* **[2] Hybrid LLM**: Fast context parser with local GPU micro-prompts (`qwen2.5:7b` via Ollama) for ambiguous multi-character scenes.
* **Auto-Sync**: Automatically updates `tag_mappings.json` (for nicknames/aliases) and `characters.json` (for voice setup).

### Step 5: Generate the Audiobook (`local_hybrid_audioconverter.py`)
```powershell
python local_hybrid_audioconverter.py
```
Options:
* `--emotions`: Enables dialogue tone/emotion delivery (default).
* `--no-emotions`: Renders pure baseline voices (custom `alts` voices still remain active).
* `--no-volume-balance`: Disables automatic loudness balancing and only applies explicit `"volume"` multipliers.
* `--rerecord`: Opens the interactive **Chunk Re-Record & Voice Switcher** directly on launch.

Interactive Features:
* **Voice Volume Options**: Choose between Auto-balanced (`volume` multipliers applied), Unbalanced / Raw (`volume` multipliers only), or open the live Volume Adjustment Editor to update `characters.json`.
* **Chunk-Level Re-Recording (`[4]` or `--rerecord`)**: Pick specific chunk numbers (`2`, `5-7`, `all`) inside any chapter to re-record, switch to another character, switch to one of the same character's emotion/alt voices, adjust chunk volume, or reset chunks while keeping all other chunks untouched (`chunk_overrides.json`). F5-TTS dialogue uses compact sentence-grouped chunks (~140 chars) for targeted re-recording without isolating brief one-word replies.

The generator will:
1. Render Kokoro chunks on CPU and F5-TTS chunks on GPU concurrently.
2. Display real-time chunk progress bars for each chapter.
3. Stitch chunks into completed chapter MP3s (`01_Chapter_1_complete.mp3`).
4. Generate absolute timestamp logs (`01_Chapter_1_timestamps.txt`) and master production voice rosters (`_voice_roster.txt`).
5. Package the MP3s into a ZIP archive and archive the entire audiobook into `finished/`.

---

## Pronunciation Replacements (`replacements.json`)

To fix phonetic pronunciations (e.g. abbreviations, titles, or fictional terms), add them to `replacements.json`:

```json
{
  "Dr.": "Doctor",
  "Mr.": "Mister",
  "etc.": "et cetera"
}
```
`new_text_parser.py` automatically applies these substitutions during the parsing stage.

---

## License

This project code is licensed under the MIT License. Book text and audio recordings belong to their respective authors and creators.
