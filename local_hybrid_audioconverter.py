import os
import re
import sys
import time
import json
import queue
import shutil
import zipfile
import threading
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ==================== TORCHAUDIO COMPATIBILITY FIX ====================
# In torchaudio >= 2.9 (installed in modern PyTorch environments), torchaudio.load
# attempts to use 'torchcodec' by default, which is missing or unsupported on Windows.
# This monkeypatch redirects torchaudio.load to soundfile so F5-TTS audio loading works reliably.
def _soundfile_load(uri, *args, **kwargs):
    data, sr = sf.read(uri, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = _soundfile_load

from kokoro import KPipeline
from f5_tts.api import F5TTS

# ==================== CONFIGURATION ====================
MANIFEST_FILE = "chapters.txt"
CHARACTERS_FILE = "characters.json"
TAG_MAPPINGS_FILE = "tag_mappings.json"
FINISHED_DIR = "finished"
MAX_CHARS_PER_KOKORO_CHUNK = 1200
MAX_CHARS_PER_F5_CHUNK = 300  # F5-TTS works best with shorter sentence chunks
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
# ========================================================

# Thread-safe console printing
PRINT_LOCK = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe stdout printing to prevent overlapping worker log lines."""
    with PRINT_LOCK:
        print(*args, **kwargs)
        sys.stdout.flush()


def ensure_tag_mappings_file():
    """Creates a default tag_mappings.json template if it doesn't already exist."""
    if not os.path.exists(TAG_MAPPINGS_FILE):
        default_mappings = {
            "narrator": "narrator",
            "detective": "detective",
            "villain": "villain",
            "hero": "Hero"
        }
        try:
            with open(TAG_MAPPINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(default_mappings, f, indent=4)
        except Exception:
            pass


def load_tag_mappings():
    """
    Loads tag-to-character alias mappings from tag_mappings.json.
    Maps lowercase tag string -> Canonical character name (in lowercase for lookup).
    """
    ensure_tag_mappings_file()
    if not os.path.exists(TAG_MAPPINGS_FILE):
        return {}
    try:
        with open(TAG_MAPPINGS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}
    except Exception as e:
        safe_print(f"[!] Warning: Could not read '{TAG_MAPPINGS_FILE}': {e}.")
        return {}


def ensure_characters_file():
    """Creates a default characters.json template if it doesn't already exist."""
    if not os.path.exists(CHARACTERS_FILE):
        default_config = {
            "narrator": {
                "engine": "kokoro",
                "voice": "af_bella",
                "pitch": 1.0,
                "speed": 1.0,
                "description": "Default narrator voice on CPU for general story text."
            },
            "detective": {
                "engine": "kokoro",
                "voice": "am_michael",
                "pitch": 0.95,
                "speed": 1.0,
                "description": "Example male character using Kokoro on CPU."
            },
            "villain": {
                "engine": "f5tts",
                "ref_audio": "wavs/output_custom_0.wav",
                "ref_text": "Kokoro is running locally with your custom weighted voice blend. I need this audio clip for audio later",
                "speed": 1.0,
                "pitch": 1.0,
                "steps": 32,
                "description": "Example character using F5-TTS zero-shot voice cloning on GPU."
            }
        }
        with open(CHARACTERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        safe_print(f"[*] Created default character template '{CHARACTERS_FILE}'.")


def load_characters():
    """Loads character-to-voice mappings from characters.json with case-insensitive matching and alias support."""
    ensure_characters_file()
    try:
        with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
            raw_chars = json.load(f)
    except Exception as e:
        safe_print(f"[!] Warning: Could not read '{CHARACTERS_FILE}': {e}. Using default narrator.")
        raw_chars = {
            "narrator": {
                "engine": "kokoro",
                "voice": "af_bella",
                "pitch": 1.0,
                "speed": 1.0
            }
        }

    # Normalize keys to lowercase for case-insensitive lookup, while preserving display name
    chars = {}
    for name, cfg in raw_chars.items():
        if isinstance(cfg, dict):
            cfg_copy = dict(cfg)
            cfg_copy["display_name"] = name
            chars[name.lower()] = cfg_copy
        elif isinstance(cfg, str):
            chars[name.lower()] = {"alias": cfg.lower(), "display_name": name}

    # Resolve aliases (e.g. "narrator": "Hero" or "narrator": {"alias": "Hero"})
    for name, cfg in list(chars.items()):
        alias_target = cfg.get("alias") or cfg.get("character")
        if alias_target and alias_target.lower() in chars:
            resolved = dict(chars[alias_target.lower()])
            resolved["display_name"] = cfg["display_name"]
            chars[name] = resolved

    return chars


def is_section_divider(text):
    """Checks if a paragraph is a section divider like ***, ---, or * * *."""
    cleaned = text.strip().replace(" ", "")
    return bool(re.match(r"^[\*\-\_]{3,}$", cleaned))


def generate_silence_file(output_dir):
    """Creates a 2-second silent WAV file for section dividers."""
    silence_path = os.path.join(output_dir, "silence_2s.wav")
    if not os.path.exists(silence_path) or os.path.getsize(silence_path) == 0:
        safe_print("Generating 2-second pause audio for section dividers...")
        cmd = f'ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono -t 2 "{silence_path}"'
        os.system(cmd)
    return silence_path


def apply_pitch_shift(input_path, output_path, pitch_factor):
    """Shifts audio pitch cleanly using rubberband filter."""
    cmd = (
        f'ffmpeg -y -i "{input_path}" -af '
        f'"rubberband=pitch={pitch_factor}" '
        f'"{output_path}"'
    )
    os.system(cmd)


def get_wav_duration(wav_path):
    """
    Returns exact audio duration in seconds using soundfile (with ffprobe and size fallbacks).
    Prevents the wave.open 2^31-1 frame header bug and guarantees exact alignment.
    """
    # 1. Primary: soundfile header inspection
    try:
        info = sf.info(wav_path)
        if 0 < info.duration < 36000:
            return float(info.duration)
    except Exception:
        pass

    # 2. Secondary: ffprobe
    try:
        import subprocess
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{wav_path}"'
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        val = float(out)
        if 0 < val < 36000:
            return val
    except Exception:
        pass

    # 3. Fallback: file size calculation (16-bit PCM @ 24kHz mono = 48,000 bytes/sec)
    try:
        file_size = os.path.getsize(wav_path)
        audio_bytes = max(0, file_size - 44)
        return audio_bytes / 48000.0
    except Exception:
        return 0.0


def format_time(seconds):
    """Formats seconds into HH:MM:SS format."""
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(int(mins), 60)
    return f"{hours:02d}:{mins:02d}:{int(secs):02d}"


# ==================== PARSING & HYBRID CHUNKING ====================

def split_text_into_tagged_blocks(text, default_character="narrator", characters_map=None, tag_mappings=None):
    """
    Splits text into chunks assigned to specific characters.
    Recognizes:
      1. Section dividers: ***, ---, * * *
      2. Block tags:
           [villain]
           Dialogue or paragraphs here
           [/villain]
      3. Paragraph prefix tags:
           [villain] "I will destroy everything," he laughed.
      4. Default text without tags -> assigned to default_character (narrator).
    """
    paragraphs = text.split("\n")
    blocks = []

    def resolve_char(name):
        if not name:
            return default_character
        t_clean = name.strip().lower()
        if tag_mappings and t_clean in tag_mappings:
            return tag_mappings[t_clean].lower()
        return t_clean

    current_char = resolve_char(default_character)
    in_explicit_block = False

    i = 0
    while i < len(paragraphs):
        raw_line = paragraphs[i].strip()
        i += 1

        if not raw_line:
            continue

        # 1. Section Dividers
        if is_section_divider(raw_line):
            blocks.append({
                "type": "DIVIDER",
                "character": "DIVIDER",
                "text": "DIVIDER"
            })
            continue

        # 2. Check for closing block tag: [/character]
        close_match = re.match(r"^\[\s*/\s*([a-zA-Z0-9_\-]+)\s*\]$", raw_line)
        if close_match:
            in_explicit_block = False
            current_char = resolve_char(default_character)
            continue

        # 3. Check for opening block tag: [character] alone on a line
        open_block_match = re.match(r"^\[\s*([a-zA-Z0-9_\-]+)\s*\]$", raw_line)
        if open_block_match:
            in_explicit_block = True
            current_char = resolve_char(open_block_match.group(1))
            continue

        # 4. Check for embedded inline tags: e.g. Narration [Hero] "Dialogue" [/Hero] Narration
        inline_tag_pattern = re.compile(r"\[\s*([a-zA-Z0-9_\-]+)\s*\](.*?)\[\s*/\s*\1\s*\]")
        if not in_explicit_block and inline_tag_pattern.search(raw_line):
            pos = 0
            for match in inline_tag_pattern.finditer(raw_line):
                start, end = match.span()
                before = raw_line[pos:start].strip()
                if before:
                    blocks.append({
                        "type": "VOICE",
                        "character": current_char,
                        "text": before
                    })
                char_name = resolve_char(match.group(1))
                dialogue = match.group(2).strip()
                if dialogue:
                    blocks.append({
                        "type": "VOICE",
                        "character": char_name,
                        "text": dialogue
                    })
                pos = end
            after = raw_line[pos:].strip()
            if after:
                blocks.append({
                    "type": "VOICE",
                    "character": current_char,
                    "text": after
                })
            continue

        # 5. Check for single prefix tag without closing: [character] Rest of text...
        prefix_match = re.match(r"^\[\s*([a-zA-Z0-9_\-]+)\s*\]\s*(.*)$", raw_line)
        if prefix_match and not in_explicit_block:
            char_name = resolve_char(prefix_match.group(1))
            content = prefix_match.group(2).strip()
            if content:
                blocks.append({
                    "type": "VOICE",
                    "character": char_name,
                    "text": content
                })
            continue

        # 6. Regular text inside or outside an explicit block
        blocks.append({
            "type": "VOICE",
            "character": current_char,
            "text": raw_line
        })

    # Group contiguous blocks of the same character up to maximum character limits
    merged_chunks = []
    for b in blocks:
        if b["type"] == "DIVIDER":
            merged_chunks.append(b)
            continue

        char = b["character"]
        text_piece = b["text"]

        if merged_chunks and merged_chunks[-1]["type"] == "VOICE" and merged_chunks[-1]["character"] == char:
            prev_len = len(merged_chunks[-1]["text"])
            char_cfg = (characters_map or {}).get(char.lower(), {})
            is_f5 = char_cfg.get("engine", "").lower() in ["f5tts", "f5-tts"] or "f5" in char.lower()
            max_limit = MAX_CHARS_PER_F5_CHUNK if is_f5 else MAX_CHARS_PER_KOKORO_CHUNK
            if prev_len + len(text_piece) + 2 <= max_limit:
                merged_chunks[-1]["text"] += "\n\n" + text_piece
                continue

        merged_chunks.append({
            "type": "VOICE",
            "character": char,
            "text": text_piece
        })

    return merged_chunks


def split_sentences(text, max_len=250):
    """Splits long text on sentence boundaries for optimal F5-TTS flow matching."""
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_len:
            current = (current + " " + s).strip() if current else s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks or [text.strip()]


# ==================== TTS ENGINES (CPU & GPU) ====================

GLOBAL_KOKORO_PIPELINE = None
GLOBAL_F5_ENGINE = None
VOICE_CACHE = {}
VOICE_LOCK = threading.Lock()


def get_kokoro_pipeline():
    """Initializes and returns the Kokoro pipeline explicitly on CPU."""
    global GLOBAL_KOKORO_PIPELINE
    if GLOBAL_KOKORO_PIPELINE is None:
        safe_print(f"[*] Initializing local Kokoro TTS pipeline on CPU (repo: {KOKORO_REPO_ID})...")
        GLOBAL_KOKORO_PIPELINE = KPipeline(lang_code="a", repo_id=KOKORO_REPO_ID, device="cpu")
    return GLOBAL_KOKORO_PIPELINE


def get_f5tts_engine():
    """Initializes and returns the F5-TTS engine explicitly on GPU (CUDA)."""
    global GLOBAL_F5_ENGINE
    if GLOBAL_F5_ENGINE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_desc = f"GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU (GPU not available)"
        safe_print(f"[*] Initializing F5-TTS Engine on {device_desc}...")
        try:
            GLOBAL_F5_ENGINE = F5TTS(model="F5TTS_v1_Base", device=device)
        except TypeError:
            try:
                GLOBAL_F5_ENGINE = F5TTS(model_type="F5-TTS", device=device)
            except TypeError:
                GLOBAL_F5_ENGINE = F5TTS(device=device)
    return GLOBAL_F5_ENGINE


def parse_and_load_kokoro_voice(pipeline, voice_str):
    """
    Parses and loads single or blended voice tensors with caching.
    Supported formats:
      - 'af_bella'
      - 'af_bella+af_nicole'
      - 'af_bella(0.7)+af_nicole(0.3)'
      - 'af_kore*0.3 + af_nicole*0.3 + af_bella*0.1 + am_michael*0.2 + am_onyx*0.1'
      - 'af_bella, am_michael'
    """
    with VOICE_LOCK:
        if voice_str in VOICE_CACHE:
            return VOICE_CACHE[voice_str]

        raw_parts = re.split(r'[\+,]', voice_str)
        items = []
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            m1 = re.match(r'^([a-zA-Z0-9_]+)\s*\(\s*([0-9.]+)\s*\)$', part)
            m2 = re.match(r'^([a-zA-Z0-9_]+)\s*\*\s*([0-9.]+)$', part)
            m3 = re.match(r'^([0-9.]+)\s*\*\s*([a-zA-Z0-9_]+)$', part)
            if m1:
                items.append((m1.group(1), float(m1.group(2))))
            elif m2:
                items.append((m2.group(1), float(m2.group(2))))
            elif m3:
                items.append((m3.group(2), float(m3.group(1))))
            else:
                items.append((part, None))

        if not items:
            items = [("af_bella", 1.0)]

        if len(items) == 1 and items[0][1] is None:
            voice_tensor = pipeline.load_voice(items[0][0])
        else:
            if any(w is None for _, w in items):
                eq_w = 1.0 / len(items)
                items = [(name, eq_w) for name, _ in items]
            else:
                total_w = sum(w for _, w in items)
                if total_w <= 0:
                    total_w = 1.0
                items = [(name, w / total_w) for name, w in items]

            tensors = [pipeline.load_voice(name) * w for name, w in items]
            voice_tensor = sum(tensors)

        VOICE_CACHE[voice_str] = voice_tensor
        return voice_tensor


def render_kokoro_chunk_local(text, char_cfg, output_wav):
    """Renders a text chunk using the local CPU Kokoro pipeline."""
    pipeline = get_kokoro_pipeline()
    voice_str = char_cfg.get("voice", "af_bella")
    speed = float(char_cfg.get("speed", 1.0))

    voice_tensor = parse_and_load_kokoro_voice(pipeline, voice_str)

    audio_pieces = []
    for _, _, audio in pipeline(text, voice=voice_tensor, speed=speed):
        if audio is not None:
            audio_pieces.append(audio)

    if not audio_pieces:
        raise RuntimeError(f"Kokoro failed to generate audio for text: '{text[:60]}...'")

    full_tensor = torch.cat(audio_pieces) if len(audio_pieces) > 1 else audio_pieces[0]
    audio_np = full_tensor.numpy() if isinstance(full_tensor, torch.Tensor) else full_tensor

    sf.write(output_wav, audio_np, 24000)


def render_f5tts_chunk_local(text, char_cfg, output_wav, temp_dir, chunk_index):
    """Renders a text chunk using local GPU F5-TTS with sentence chunking & stitching."""
    f5 = get_f5tts_engine()

    ref_audio = char_cfg.get("ref_audio", "ref_sample.wav")
    ref_text = char_cfg.get("ref_text", "")
    speed = float(char_cfg.get("speed", 1.0))
    steps = int(char_cfg.get("steps", 32))

    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"F5-TTS reference audio '{ref_audio}' does not exist.")

    sentences = split_sentences(text, max_len=MAX_CHARS_PER_F5_CHUNK)

    if len(sentences) == 1:
        wav, sr, _ = f5.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=sentences[0],
            file_wave=output_wav,
            speed=speed,
            nfe_step=steps
        )
        if not os.path.exists(output_wav):
            sf.write(output_wav, wav, sr)
    else:
        audio_segments = []
        target_sr = 24000
        # 0.18s natural pause between sentences
        pause_samples = int(0.18 * target_sr)
        pause = np.zeros(pause_samples, dtype=np.float32)

        for s_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            wav, sr, _ = f5.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=sentence,
                speed=speed,
                nfe_step=steps
            )
            target_sr = sr
            if audio_segments:
                audio_segments.append(pause)
            audio_segments.append(wav)

        full_wav = np.concatenate(audio_segments) if audio_segments else np.zeros(target_sr, dtype=np.float32)
        sf.write(output_wav, full_wav, target_sr)


# ==================== PARALLEL CHAPTER PROCESSING ====================

def process_hybrid_chapter_simultaneous(txt_filename, characters_map, master_output_dir, tag_mappings=None):
    """
    Processes a chapter by running F5-TTS on GPU and Kokoro on CPU simultaneously.
    """
    if not os.path.exists(txt_filename):
        safe_print(f"  [!] Warning: Could not find chapter file '{txt_filename}'. Skipping.")
        return False

    base_name = os.path.splitext(os.path.basename(txt_filename))[0]
    chapter_output_dir = os.path.join(master_output_dir, "chunks_" + base_name)
    final_mp3 = os.path.join(master_output_dir, f"{base_name}_complete.mp3")
    timestamp_file = os.path.join(master_output_dir, f"{base_name}_timestamps.txt")

    if os.path.exists(final_mp3) and os.path.getsize(final_mp3) > 0:
        safe_print(f"\n--> Section '{base_name}' MP3 already exists in output folder. Skipping.")
        return True

    os.makedirs(chapter_output_dir, exist_ok=True)

    with open(txt_filename, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = split_text_into_tagged_blocks(raw_text, default_character="narrator", characters_map=characters_map, tag_mappings=tag_mappings)
    safe_print(f"\n--> Processing Simultaneous Hybrid Chapter: '{txt_filename}' ({len(chunks)} blocks)")

    # Prepare shared data structures
    kokoro_queue = queue.Queue()
    f5_queue = queue.Queue()
    completed_chunks = {}
    worker_errors = []

    chapter_silence_file = None
    all_tasks = []

    # 1. Pre-scan all chunks and route
    for i, chunk in enumerate(chunks):
        chunk_idx = i + 1
        chunk_filename = os.path.join(chapter_output_dir, f"chunk_{chunk_idx:03d}.wav")
        c_type = chunk["type"]
        char_name = chunk["character"]
        canonical_char = tag_mappings.get(char_name.lower(), char_name.lower()) if tag_mappings else char_name.lower()
        char_cfg = characters_map.get(canonical_char, characters_map.get("narrator", {}))
        engine = char_cfg.get("engine", "kokoro").lower()
        pitch_factor = float(char_cfg.get("pitch", 1.0))
        display_name = char_cfg.get("display_name", canonical_char.title())

        if c_type == "DIVIDER":
            voice_label = "2s Silence (Section Divider)"
            engine_label = "SILENCE"
        elif engine in ["f5tts", "f5-tts"]:
            ref_audio_name = os.path.basename(char_cfg.get("ref_audio", "sample.wav"))
            voice_label = f"f5tts [ref: {ref_audio_name}]"
            engine_label = "F5-TTS (GPU)"
        else:
            kokoro_voice = char_cfg.get("voice", "af_bella")
            if pitch_factor != 1.0:
                voice_label = f"{kokoro_voice} (pitch: {pitch_factor})"
            else:
                voice_label = kokoro_voice
            engine_label = "KOKORO (CPU)"

        task_info = {
            "index": chunk_idx,
            "total": len(chunks),
            "type": c_type,
            "character": display_name,
            "char_cfg": char_cfg,
            "engine": engine,
            "engine_label": engine_label,
            "voice_label": voice_label,
            "pitch_factor": pitch_factor,
            "chunk_filename": chunk_filename,
            "text": chunk.get("text", "")
        }
        all_tasks.append(task_info)

        # Resume support: skip if already rendered
        if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 0:
            safe_print(f"    [Chunk {chunk_idx:03d}/{len(chunks)}] Found existing on disk. Resuming...")
            completed_chunks[chunk_idx] = {
                "index": chunk_idx,
                "filename": f"chunk_{chunk_idx:03d}.wav",
                "character": char_name,
                "engine": engine_label,
                "voice": voice_label,
                "duration": get_wav_duration(chunk_filename)
            }
            continue

        # Immediate handling for Section Dividers
        if c_type == "DIVIDER":
            safe_print(f"    [Chunk {chunk_idx:03d}/{len(chunks)}] Section Divider (2s Pause)")
            if not chapter_silence_file:
                chapter_silence_file = generate_silence_file(master_output_dir)

            with open(chapter_silence_file, "rb") as sf_in:
                silence_data = sf_in.read()
            with open(chunk_filename, "wb") as out_f:
                out_f.write(silence_data)

            completed_chunks[chunk_idx] = {
                "index": chunk_idx,
                "filename": f"chunk_{chunk_idx:03d}.wav",
                "character": "DIVIDER",
                "engine": "SILENCE",
                "voice": voice_label,
                "duration": get_wav_duration(chunk_filename)
            }
            continue

        # Route to respective engine queue
        if engine in ["f5tts", "f5-tts"]:
            f5_queue.put(task_info)
        else:
            kokoro_queue.put(task_info)

    # 2. Define concurrent worker functions
    def kokoro_worker():
        while True:
            try:
                task = kokoro_queue.get_nowait()
            except queue.Empty:
                break

            idx = task["index"]
            total = task["total"]
            char = task["character"]
            cfg = task["char_cfg"]
            text = task["text"]
            chunk_file = task["chunk_filename"]
            pitch = task["pitch_factor"]
            v_label = task["voice_label"]
            e_label = task["engine_label"]

            safe_print(f"    [Chunk {idx:03d}/{total}] [CPU-Kokoro] [{char.upper()}] (Voice: {v_label}) ({len(text)} chars)...")
            temp_raw = os.path.join(chapter_output_dir, f"temp_raw_{idx:03d}.wav")

            t0 = time.time()
            try:
                render_kokoro_chunk_local(text, cfg, temp_raw)
                if pitch != 1.0:
                    apply_pitch_shift(temp_raw, chunk_file, pitch)
                    if os.path.exists(temp_raw):
                        os.remove(temp_raw)
                else:
                    os.replace(temp_raw, chunk_file)

                dur = get_wav_duration(chunk_file)
                elapsed = time.time() - t0
                safe_print(f"    [OK] [Chunk {idx:03d}/{total}] [CPU-Kokoro] Done in {elapsed:.2f}s (Audio: {dur:.2f}s)")

                completed_chunks[idx] = {
                    "index": idx,
                    "filename": f"chunk_{idx:03d}.wav",
                    "character": char,
                    "engine": e_label,
                    "voice": v_label,
                    "duration": dur
                }
            except Exception as e:
                safe_print(f"    [X] Error rendering Kokoro chunk {idx} ({char}): {e}")
                worker_errors.append((idx, char, e))
                if os.path.exists(temp_raw):
                    try: os.remove(temp_raw)
                    except Exception: pass
            finally:
                kokoro_queue.task_done()

    def f5_worker():
        while True:
            try:
                task = f5_queue.get_nowait()
            except queue.Empty:
                break

            idx = task["index"]
            total = task["total"]
            char = task["character"]
            cfg = task["char_cfg"]
            text = task["text"]
            chunk_file = task["chunk_filename"]
            pitch = task["pitch_factor"]
            v_label = task["voice_label"]
            e_label = task["engine_label"]

            safe_print(f"    [Chunk {idx:03d}/{total}] [GPU-F5TTS]  [{char.upper()}] (Voice: {v_label}) ({len(text)} chars)...")
            temp_raw = os.path.join(chapter_output_dir, f"temp_raw_{idx:03d}.wav")

            t0 = time.time()
            try:
                render_f5tts_chunk_local(text, cfg, temp_raw, chapter_output_dir, idx)
                if pitch != 1.0:
                    apply_pitch_shift(temp_raw, chunk_file, pitch)
                    if os.path.exists(temp_raw):
                        os.remove(temp_raw)
                else:
                    os.replace(temp_raw, chunk_file)

                dur = get_wav_duration(chunk_file)
                elapsed = time.time() - t0
                safe_print(f"    [OK] [Chunk {idx:03d}/{total}] [GPU-F5TTS]  Done in {elapsed:.2f}s (Audio: {dur:.2f}s)")

                completed_chunks[idx] = {
                    "index": idx,
                    "filename": f"chunk_{idx:03d}.wav",
                    "character": char,
                    "engine": e_label,
                    "voice": v_label,
                    "duration": dur
                }
            except Exception as e:
                safe_print(f"    [X] Error rendering F5-TTS chunk {idx} ({char}): {e}")
                worker_errors.append((idx, char, e))
                if os.path.exists(temp_raw):
                    try: os.remove(temp_raw)
                    except Exception: pass
            finally:
                f5_queue.task_done()

    # 3. Start simultaneous threads
    pending_kokoro = kokoro_queue.qsize()
    pending_f5 = f5_queue.qsize()

    safe_print(f"[*] Queued tasks: {pending_kokoro} Kokoro chunks (CPU), {pending_f5} F5-TTS chunks (GPU).")

    threads = []
    # GPU F5-TTS Worker (1 thread targeting NVIDIA GPU)
    if pending_f5 > 0:
        f5_t = threading.Thread(target=f5_worker, name="F5TTS-GPU-Worker")
        f5_t.start()
        threads.append(f5_t)

    # CPU Kokoro Worker(s) (1-2 threads targeting CPU)
    if pending_kokoro > 0:
        num_kokoro_workers = 2 if os.cpu_count() and os.cpu_count() >= 8 else 1
        for w_i in range(num_kokoro_workers):
            k_t = threading.Thread(target=kokoro_worker, name=f"Kokoro-CPU-Worker-{w_i+1}")
            k_t.start()
            threads.append(k_t)

    # Wait for all workers to complete
    for t in threads:
        t.join()

    # Check for any failures
    if worker_errors:
        safe_print(f"[!] Encountered {len(worker_errors)} error(s) during chapter rendering.")
        for err_idx, err_char, err_ex in worker_errors:
            safe_print(f"    - Chunk {err_idx:03d} ({err_char}): {err_ex}")
        return False

    # 4. Verify all chunks generated and build sequential metadata
    saved_files = []
    chunk_metadata = []

    for idx in range(1, len(chunks) + 1):
        c_file = os.path.join(chapter_output_dir, f"chunk_{idx:03d}.wav")
        if not os.path.exists(c_file) or os.path.getsize(c_file) == 0:
            safe_print(f"[X] Missing or empty chunk audio file: '{c_file}'")
            return False

        saved_files.append(c_file)
        if idx in completed_chunks:
            chunk_metadata.append(completed_chunks[idx])
        else:
            chunk_metadata.append({
                "index": idx,
                "filename": f"chunk_{idx:03d}.wav",
                "character": "unknown",
                "engine": "unknown",
                "voice": "unknown",
                "duration": get_wav_duration(c_file)
            })

    # 5. Stitch into master MP3
    list_file = os.path.join(chapter_output_dir, "file_list.txt")
    with open(list_file, "w", encoding="utf-8") as lf:
        for sf_path in saved_files:
            lf.write(f"file '{os.path.basename(sf_path)}'\n")

    safe_print(f"    [*] Merging {len(saved_files)} chunks into '{final_mp3}'...")
    cmd = (
        f'ffmpeg -y -f concat -safe 0 -i "{list_file}" -codec:a libmp3lame'
        f' -qscale:a 2 "{final_mp3}"'
    )

    if os.system(cmd) == 0:
        safe_print(f"    [OK] Success! Saved '{final_mp3}'.")

        # Write absolute timestamps
        current_time = 0.0
        with open(timestamp_file, "w", encoding="utf-8") as tf:
            tf.write("======================================================================\n")
            tf.write(f" ABSOLUTE TIMESTAMP LOG: {base_name}\n")
            tf.write(f" Output File: {os.path.basename(final_mp3)}\n")
            tf.write("======================================================================\n")
            used_characters = {
                meta["character"]
                for meta in chunk_metadata
                if meta["character"] != "DIVIDER"
            }
            tf.write("VOICE ROSTER USED:\n")
            for c_name, c_info in characters_map.items():
                if c_name not in used_characters:
                    continue
                c_eng = c_info.get("engine", "kokoro").upper()
                if c_eng == "KOKORO":
                    v_str = c_info.get("voice", "af_bella")
                    p_val = float(c_info.get("pitch", 1.0))
                    if p_val != 1.0:
                        v_str += f" (pitch: {p_val})"
                    tf.write(f"  - [{c_name}] (KOKORO - CPU): {v_str}\n")
                else:
                    ref_f = os.path.basename(c_info.get("ref_audio", "sample.wav"))
                    tf.write(f"  - [{c_name}] (F5-TTS - GPU): {ref_f}\n")
            tf.write("======================================================================\n\n")

            for meta in chunk_metadata:
                start_str = format_time(current_time)
                end_time = current_time + meta["duration"]
                end_str = format_time(end_time)

                if meta["character"] == "DIVIDER":
                    tf.write(
                        f"Chunk {meta['index']:03d} ({meta['filename']}) | "
                        f"[{start_str} - {end_str}] | Type: Section Divider (2s Pause)\n"
                    )
                else:
                    tf.write(
                        f"Chunk {meta['index']:03d} ({meta['filename']}) | "
                        f"[{start_str} - {end_str}] | "
                        f"Char: {meta['character']:<12} | Voice: {meta['voice']}\n"
                    )
                current_time = end_time

            tf.write("\n======================================================================\n")
            tf.write(f" Total Audio Duration: {format_time(current_time)}\n")
            tf.write("======================================================================\n")

        return True
    else:
        safe_print(f"    [X] FFmpeg merge failed for {txt_filename}.")
        return False


def parse_selection_indices(input_str, max_idx):
    """
    Parses comma-separated numbers and ranges (e.g. '1, 3', '2-4') into a set of 1-based indices.
    """
    selected = set()
    parts = input_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) == 2:
                try:
                    start, end = int(bounds[0].strip()), int(bounds[1].strip())
                    for n in range(min(start, end), max(start, end) + 1):
                        if 1 <= n <= max_idx:
                            selected.add(n)
                except ValueError:
                    pass
        else:
            try:
                n = int(part)
                if 1 <= n <= max_idx:
                    selected.add(n)
            except ValueError:
                pass
    return sorted(list(selected))


def clear_chapter_recording(master_output_dir, base_name):
    """Deletes existing chunk WAVs, complete MP3, and timestamps for a specific chapter."""
    final_mp3 = os.path.join(master_output_dir, f"{base_name}_complete.mp3")
    timestamp_file = os.path.join(master_output_dir, f"{base_name}_timestamps.txt")
    chunk_dir = os.path.join(master_output_dir, "chunks_" + base_name)

    if os.path.exists(final_mp3):
        try: os.remove(final_mp3)
        except Exception: pass
    if os.path.exists(timestamp_file):
        try: os.remove(timestamp_file)
        except Exception: pass
    if os.path.exists(chunk_dir):
        try: shutil.rmtree(chunk_dir)
        except Exception: pass


def main():
    safe_print("==================================================================")
    safe_print("  Simultaneous Hybrid Audiobook Generator (CPU Kokoro + GPU F5)  ")
    safe_print("==================================================================")

    if not os.path.exists(MANIFEST_FILE):
        safe_print(f"Error: Could not find manifest file '{MANIFEST_FILE}'.")
        safe_print(f"Tip: Run 'new_text_parser.py' first to format chapters into '{MANIFEST_FILE}'.")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
        chapter_files = [line.strip() for line in mf if line.strip() and not line.startswith("#")]

    if not chapter_files:
        safe_print(f"Error: Manifest file '{MANIFEST_FILE}' is empty.")
        return

    # Load character profiles and tag mappings
    characters = load_characters()
    tag_mappings = load_tag_mappings()
    safe_print(f"\nLoaded {len(characters)} character profiles from '{CHARACTERS_FILE}':")
    for name, cfg in characters.items():
        disp_name = cfg.get("display_name", name)
        engine = cfg.get("engine", "kokoro").upper()
        if engine == "KOKORO":
            safe_print(f"  - [{disp_name}]: KOKORO (CPU) -> {cfg.get('voice', '')}")
        else:
            safe_print(f"  - [{disp_name}]: F5-TTS (GPU) -> {cfg.get('ref_audio', '')}")

    if tag_mappings:
        safe_print(f"\nLoaded {len(tag_mappings)} tag/alias mapping(s) from '{TAG_MAPPINGS_FILE}':")
        for tag, canon in sorted(tag_mappings.items()):
            if tag != canon:
                safe_print(f"  - Tag [{tag}] maps to character [{canon}]")

    first_file_base = os.path.splitext(os.path.basename(chapter_files[0]))[0]
    cleaned_first = re.sub(r"^\d+_", "", first_file_base)
    book_title_base = re.sub(r"^intro_", "", cleaned_first)

    master_output_dir = f"{book_title_base}_hybrid_audiobook_output"

    # If already moved to finished in a previous run, bring back to workspace for editing
    finished_candidate = os.path.join(FINISHED_DIR, master_output_dir)
    if not os.path.exists(master_output_dir) and os.path.exists(finished_candidate):
        safe_print(f"\n[*] Found previously finished recording in '{FINISHED_DIR}/'. Loading into workspace...")
        shutil.move(finished_candidate, master_output_dir)

    os.makedirs(master_output_dir, exist_ok=True)

    # ==================== UPFRONT RE-RECORD SCANNER ====================
    scan_results = []
    has_existing = False
    for idx, txt_file in enumerate(chapter_files, 1):
        bname = os.path.splitext(os.path.basename(txt_file))[0]
        final_mp3 = os.path.join(master_output_dir, f"{bname}_complete.mp3")
        chunk_dir = os.path.join(master_output_dir, "chunks_" + bname)
        chunks_count = 0
        if os.path.exists(chunk_dir):
            chunks_count = len([f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".wav") and os.path.getsize(os.path.join(chunk_dir, f)) > 0])

        if os.path.exists(final_mp3) and os.path.getsize(final_mp3) > 0:
            status = "COMPLETED (MP3 exists)"
            has_existing = True
        elif chunks_count > 0:
            status = f"PARTIAL ({chunks_count} chunks rendered)"
            has_existing = True
        else:
            status = "NEW"

        scan_results.append({
            "idx": idx,
            "txt_file": txt_file,
            "base_name": bname,
            "status": status,
        })

    if has_existing:
        safe_print("\n" + "=" * 70)
        safe_print("                 Existing Audio Sections Detected")
        safe_print("=" * 70)
        for item in scan_results:
            safe_print(f"  [{item['idx']:02d}] {item['base_name']:<40} -> [{item['status']}]")
        safe_print("=" * 70)
        safe_print("Options:")
        safe_print("  [1] Resume / Keep existing (skip completed, finish partials) [Default]")
        safe_print("  [2] Re-record EVERYTHING from scratch")
        safe_print("  [3] Choose specific sections to re-record")
        choice = input("\nSelect how to proceed [1, 2, or 3] (default 1): ").strip()

        if choice == "2":
            safe_print("\n[*] Clearing ALL previous recordings. Re-recording entire book from scratch...")
            for item in scan_results:
                clear_chapter_recording(master_output_dir, item["base_name"])
            safe_print("[OK] All sections cleared.")
        elif choice == "3":
            safe_print("\nEnter section number(s) to re-record (e.g. '1, 3' or '2-4' or 'all'):")
            sel_input = input("> ").strip()
            if sel_input.lower() == "all":
                to_clear = list(range(1, len(chapter_files) + 1))
            else:
                to_clear = parse_selection_indices(sel_input, len(chapter_files))

            if to_clear:
                cleared_names = []
                for num in to_clear:
                    item = scan_results[num - 1]
                    clear_chapter_recording(master_output_dir, item["base_name"])
                    cleared_names.append(item["base_name"])
                safe_print(f"\n[OK] Scheduled {len(cleared_names)} section(s) for fresh re-recording:")
                for name in cleared_names:
                    safe_print(f"    - {name}")
                safe_print("[*] All other completed sections will be kept and skipped automatically.\n")
            else:
                safe_print("[-] No valid sections entered. Resuming standard progress.")

    # ==================== MAIN PROCESSING LOOP ====================
    for idx, txt_file in enumerate(chapter_files, 1):
        safe_print(f"\n[{idx}/{len(chapter_files)}] Processing chapter: {txt_file}")
        success = process_hybrid_chapter_simultaneous(txt_file, characters, master_output_dir, tag_mappings=tag_mappings)
        if not success:
            safe_print(f"[!] Pausing due to error in {txt_file}.")
            break

    # Package into ZIP archive if multiple chapters
    if len(chapter_files) > 1:
        audio_zip_filename = os.path.join(master_output_dir, f"{book_title_base}_hybrid_audiobook.zip")
        safe_print(f"\n[*] Packaging complete audio MP3s into '{audio_zip_filename}'...")
        with zipfile.ZipFile(audio_zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for txt_file in chapter_files:
                base_name = os.path.splitext(os.path.basename(txt_file))[0]
                final_mp3 = os.path.join(master_output_dir, f"{base_name}_complete.mp3")
                if os.path.exists(final_mp3):
                    zipf.write(final_mp3, arcname=os.path.basename(final_mp3))
        safe_print(f"[OK] Created '{audio_zip_filename}'!")

    # Check if all chapters in manifest are finished
    all_finished = all(
        os.path.exists(os.path.join(master_output_dir, f"{os.path.splitext(os.path.basename(f))[0]}_complete.mp3"))
        for f in chapter_files
    )

    if all_finished:
        os.makedirs(FINISHED_DIR, exist_ok=True)
        dest_folder = os.path.join(FINISHED_DIR, os.path.basename(master_output_dir))
        safe_print(f"\n[*] All chapters complete! Moving finished audiobook folder to '{FINISHED_DIR}/'...")
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
        shutil.move(master_output_dir, dest_folder)
        safe_print(f"[OK] Successfully archived in: '{dest_folder}'")

    safe_print("\n==================================================================")
    safe_print("       Simultaneous Hybrid Audiobook Processing Finished!         ")
    safe_print("==================================================================")


if __name__ == "__main__":
    main()
