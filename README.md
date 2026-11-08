# Hybrid Multi-Voice Audiobook Generator

An automated text-to-speech (TTS) pipeline designed for creating audiobooks with dynamic, multi-voice narration. It combines the speed and clarity of **Kokoro TTS** for general narration with the voice-cloning capabilities of **F5-TTS** for specific characters and dialogue.

---

## Key Features

* **Hybrid TTS Engine**:
  * **Kokoro TTS (API)**: Fast, natural, high-fidelity narration for general book text and secondary characters (supports 50/50 and weighted voice blending).
  * **F5-TTS (Flow Matching)**: Zero-shot voice cloning for custom characters using a simple 5–10 second audio sample.
* **Intelligent Text Parser (`new_text_parser.py`)**:
  * Automatically detects chapters (`Chapter 1`, `Chapter 1: Title`).
  * Extracts the title and author.
  * Strips fanfiction/AO3 metadata, author notes, chapter summaries, and end notes.
  * Optionally heals manual hard-wrapped line breaks while preserving dialogue and section dividers.
  * Interactive or bulk trailing section removal (with live text excerpts).
* **Robust Audio Generation (`hybrid_audioconverter.py`)**:
  * **Upfront Re-record Options**: Scans all sections at launch and lets you selectively re-record specific chapters (or the entire book) without having to babysit the process.
  * **Automatic Archiving**: Completed audiobook folders are automatically moved to a dedicated `finished/` directory.
  * **Resume Support**: Completed chapters and previously rendered chunks are skipped automatically.
  * **Smart Sentence Chunking**: Monologues rendered by F5-TTS are split on sentence boundaries under 300 characters for high flow-matching fidelity, then spliced together seamlessly.
  * **Section Dividers**: Inserts a clean 2-second silent pause wherever `***` appears.
  * **Rubberband Pitch Shifting**: Post-production pitch adjustment per character or globally without altering playback speed.
  * **Timestamp Tracking**: Generates an absolute timestamp log (`_timestamps.txt`) showing start and end times for each character chunk.
  * **Automatic Packaging**: Combines all finished chapter MP3s into a ZIP archive.

---

## Project Structure

```text
├── raw_txt/                    # Place raw story/book .txt files here
│   └── .gitkeep
├── new_text_parser.py          # Cleans and splits raw text into formatted chapters
├── llm_dialogue_tagger.py      # Local GPU LLM dialogue tagger & voice mapper (Ollama/Qwen 2.5)
├── local_hybrid_audioconverter.py # Simultaneous CPU Kokoro + GPU F5-TTS generator (No Docker required)
├── hybrid_audioconverter.py    # Legacy Docker/API-based generator
├── requirements.txt            # Python dependencies (with CUDA PyTorch index)
├── setup_env.bat               # One-click virtual environment setup script (Windows)
├── characters.json             # Character-to-voice mapping configuration (auto-generated)
├── replacements.json           # Word pronunciation replacement dictionary (auto-generated)
└── README.md
```

---

## Prerequisites

1. **Python 3.10 or 3.11** *(Recommended; avoid Python 3.13+ for PyTorch/F5-TTS compatibility)*.
2. **FFmpeg**: Required for audio slicing, concatenation, and rubberband pitch adjustment.
   * [Download FFmpeg](https://ffmpeg.org/download.html) and ensure `ffmpeg` is added to your system `PATH`.
3. **Kokoro TTS Server**:
   * A local Kokoro API running on `http://localhost:8880` (e.g. via Docker, Kokoro-FastAPI, or OpenedAI-Speech).
4. **NVIDIA GPU with CUDA** *(Recommended)*:
   * Greatly accelerates F5-TTS voice cloning inference.

---

## Installation

### Option 1: One-Click Setup (Windows)
Double-click `setup_env.bat` in the project root. This will:
1. Create a dedicated Python 3.11 virtual environment (`.venv`) using `uv` or standard Python.
2. Automatically install all dependencies from `requirements.txt`.

### Option 2: Manual Setup
Open PowerShell or your terminal in this directory:

```powershell
# 1. Create a Python 3.11 virtual environment (with pip seeded)
uv venv --seed --python 3.11 .venv
# (or: python -m venv .venv)

# 2. Activate the virtual environment
.\.venv\Scripts\activate

# 3. Install dependencies
# Using uv (fastest):
uv pip install -r requirements.txt --index-strategy unsafe-best-match
# (or standard pip):
pip install -r requirements.txt
```

---

## Step-by-Step Workflow

### Step 1: Add Raw Books
Place your source `.txt` files into the `raw_txt/` directory.

### Step 2: Parse and Clean the Text
Run the parser to clean metadata, remove notes, and split into chapters:

```powershell
python new_text_parser.py
```

* **Interactive Prompts**:
  1. **File Selection**: Choose which file from `raw_txt/` to process.
  2. **Multi-Newline Dividers**: Choose `y` to turn more than one new line in a row (multiple blank lines representing scene transitions) into `***` dividers.
  3. **Hard-Wrap Healing**: Choose `y` to merge lines split by manual wrapping into natural paragraphs.
  4. **Trailing Section Review**: Review trailing sections chapter-by-chapter (with live excerpts) to remove author notes.
* Parsed chapters will be saved to `parsed_chapters/` and registered in `chapters.txt`.

---

### Step 3: Configure Character Voices (`characters.json`)
The first time you run the converter, a default `characters.json` file is created. You can customize the voice roster:

```json
{
  "narrator": {
    "engine": "kokoro",
    "voice": "af_bella(0.7)+af_nicole(0.3)",
    "pitch": 1.0,
    "description": "Default narrator for all untagged text."
  },
  "detective": {
    "engine": "kokoro",
    "voice": "am_michael",
    "pitch": 0.95
  },
  "villain": {
    "engine": "f5tts",
    "ref_audio": "voices/villain_voice.wav",
    "ref_text": "The exact words spoken in the villain reference clip.",
    "speed": 1.0,
    "steps": 32
  }
}
```

#### Voice Configuration Rules:
* **`narrator`**: The default fallback. All untagged paragraphs use this voice.
* **Kokoro Voice Blending**:
  * Equal blend: `"af_bella+af_nicole"`
  * Weighted blend: `"af_bella(0.6)+af_nicole(0.2)+am_michael(0.2)"`
* **F5-TTS Voice Cloning**:
  * `ref_audio`: Path to a clean 5–10 second WAV file of the target voice.
  * `ref_text`: The exact transcript of what is spoken in the reference audio.
  * `steps`: Flow matching steps (default: `32` for best quality; `16` for faster generation).

---

### Step 4: Tag Character Dialogue in Your Chapters

#### Option A: Automatic Dialogue Tagging (Fast Smart Engine & Hybrid LLM)
Run the dialogue tagger from `new_text_parser.py` or standalone via `llm_dialogue_tagger.py`:

```powershell
python llm_dialogue_tagger.py
```

Offers 3 optimized modes:
1. **[1] Fast Smart Parser (Instant & 100% Quote Coverage - Recommended)**:
   * Uses grammar-aware narrative parsing, speech verb detection, and scene turn-taking.
   * **100% Quote Coverage**: Never misses quotes—multi-quote lines (e.g. `"Wait a moment," John said. "Look over there."`) have **both** quotes tagged accurately (`[John] "Wait a moment," [/John] John said. [John] "Look over there." [/John]`).
   * **Blazing Fast**: Tags thousands of words in **less than 0.1 seconds** (no GPU wait).
2. **[2] Hybrid LLM (Fast Context + GPU LLM Verification)**:
   * Combines the fast context parser with local GPU micro-prompts (`qwen2.5:7b` via Ollama) for ambiguous multi-character scenes.
   * 10x-50x faster than full JSON generation while guaranteeing 100% quote coverage.
3. **[3] Script Form**:
   * Formats screenplays/scripts (`Alice: Hello` -> `[Alice] "Hello" [/Alice]`).
* **Auto-Sync**: Automatically updates `tag_mappings.json` (for nicknames/aliases) and `characters.json` (for voice configuration).

#### Option B: Manual Tagging
You can also manually add tags directly into the text inside `parsed_chapters/`:

##### 1. Inline or Single-Paragraph Tag:
```text
The storm grew louder as the stranger kicked open the tavern door.

[villain] "No one leaves this room alive," he growled into the darkness.

The patrons froze in terror, nobody daring to make a sound.
```

##### 2. Multi-Paragraph Block Tag:
```text
[villain]
"For three long years I waited in that dungeon.
Every single day, the cold seeped into my bones.
Now it is your turn to suffer."
[/villain]

The detective stood unmoving, his hand slowly reaching for his coat.
```

* Any text without tags automatically defaults back to the `"narrator"` voice.

---

### Step 5: Generate the Audiobook
Ensure your Kokoro server is running at `http://localhost:8880`, then start the converter:

```powershell
python hybrid_audioconverter.py
```

The script will:
1. Render each paragraph chunk (`chunk_001.wav`, `chunk_002.wav`, etc.).
2. Concatenate the chunks into completed chapter MP3s (`01_Chapter_1_complete.mp3`).
3. Generate detailed timestamps (`01_Chapter_1_timestamps.txt`).
4. Package the finished MP3s into a single `.zip` file inside `[BookTitle]_hybrid_audiobook_output/`.

---

## Pronunciation Replacements (`replacements.json`)

To fix phonetic pronunciations (e.g. character names, abbreviations, or fictional terms), add them to `replacements.json`:

```json
{
  "Dr.": "Doctor",
  "Mr.": "Mister",
}
```
`new_text_parser.py` automatically applies these substitutions during the parsing stage.

---

## License

This project code is licensed under the MIT License. Book text and audio recordings belong to their respective authors and creators.
