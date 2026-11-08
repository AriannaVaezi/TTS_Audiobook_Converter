import os
import re
import time
import json
import wave
import glob
import shutil
import zipfile
import requests
import soundfile as sf
import torch
import torchaudio

# Compatibility patch for modern torchaudio (>=2.9) on Windows
def _soundfile_load(uri, *args, **kwargs):
    data, sr = sf.read(uri, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = _soundfile_load

# ==================== CONFIGURATION ====================
API_URL = "http://localhost:8880/v1/audio/speech"
MANIFEST_FILE = "chapters.txt"
CHARACTERS_FILE = "characters.json"
TAG_MAPPINGS_FILE = "tag_mappings.json"
FINISHED_DIR = "finished"
MAX_CHARS_PER_KOKORO_CHUNK = 1200
MAX_CHARS_PER_F5_CHUNK = 300  # F5-TTS works best with shorter sentence chunks
# ========================================================


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
        print(f"[!] Warning: Could not read '{TAG_MAPPINGS_FILE}': {e}.")
        return {}


def ensure_characters_file():
    """Creates a default characters.json template if it doesn't already exist."""
    if not os.path.exists(CHARACTERS_FILE):
        default_config = {
            "narrator": {
                "engine": "kokoro",
                "voice": "af_bella",
                "pitch": 1.0,
                "description": "Default narrator voice for general story text."
            },
            "detective": {
                "engine": "kokoro",
                "voice": "am_michael",
                "pitch": 0.95,
                "description": "Example male character using Kokoro preset."
            },
            "villain": {
                "engine": "f5tts",
                "ref_audio": "voices/villain_sample.wav",
                "ref_text": "You are too late. The shadow has already consumed this city.",
                "speed": 1.0,
                "pitch": 1.0,
                "steps": 32,
                "description": "Example character using F5-TTS zero-shot voice cloning."
            }
        }
        with open(CHARACTERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"[*] Created default character template '{CHARACTERS_FILE}'.")


def load_characters():
    """Loads character-to-voice mappings from characters.json with case-insensitive matching and alias support."""
    ensure_characters_file()
    try:
        with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
            raw_chars = json.load(f)
    except Exception as e:
        print(f"[!] Warning: Could not read '{CHARACTERS_FILE}': {e}. Using default narrator.")
        raw_chars = {
            "narrator": {
                "engine": "kokoro",
                "voice": "af_bella",
                "pitch": 1.0
            }
        }

    chars = {}
    for name, cfg in raw_chars.items():
        if isinstance(cfg, dict):
            cfg_copy = dict(cfg)
            cfg_copy["display_name"] = name
            chars[name.lower()] = cfg_copy
        elif isinstance(cfg, str):
            chars[name.lower()] = {"alias": cfg.lower(), "display_name": name}

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
        print("Generating 2-second pause audio for section dividers...")
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
    Prevents the wave.open 2^31-1 frame header bug and guarantees exact alignment with the finished MP3.
    """
    # 1. Primary: soundfile header inspection (exact down to microseconds)
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
        
        # Check if we can merge with the previous chunk
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


# ==================== TTS ENGINES ====================

# Global F5-TTS engine cached instance
F5_INSTANCE = None

def get_f5tts_engine():
    """Lazy loads F5-TTS only when a chunk actually needs it."""
    global F5_INSTANCE
    if F5_INSTANCE is None:
        try:
            import torch
            from f5_tts.api import F5TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[*] Initializing F5-TTS Engine on device: {device}...")
            try:
                # Newer F5-TTS versions use model="F5TTS_v1_Base" or default
                F5_INSTANCE = F5TTS(model="F5TTS_v1_Base", device=device)
            except TypeError:
                try:
                    # Older F5-TTS versions used model_type="F5-TTS"
                    F5_INSTANCE = F5TTS(model_type="F5-TTS", device=device)
                except TypeError:
                    F5_INSTANCE = F5TTS(device=device)
        except Exception as e:
            print(f"[X] Failed to load F5-TTS: {e}")
            raise e
    return F5_INSTANCE


def render_kokoro_chunk(text, voice_name, output_wav):
    """Renders a text chunk using the local Kokoro HTTP API."""
    payload = {
        "model": "kokoro",
        "voice": voice_name,
        "input": text,
        "response_format": "wav",
    }
    response = requests.post(API_URL, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Kokoro API error ({response.status_code}): {response.text}")

    with open(output_wav, "wb") as f:
        f.write(response.content)


def split_sentences(text, max_len=250):
    """Splits long text on sentence boundaries for optimal F5-TTS flow matching."""
    sentences = re.split(r'(?<=[.!?…])\s+', text)
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
    return chunks or [text]


def render_f5tts_chunk(text, char_cfg, output_wav, temp_dir):
    """Renders a text chunk using F5-TTS with sentence chunking & stitching."""
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
        import numpy as np
        audio_segments = []
        target_sr = 24000
        pause = np.zeros(int(0.18 * target_sr), dtype=np.float32)

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


# ==================== CHAPTER PROCESSING ====================

def process_hybrid_chapter(txt_filename, characters_map, master_output_dir, tag_mappings=None):
    """Processes a chapter, dynamically routing chunks to Kokoro or F5-TTS."""
    if not os.path.exists(txt_filename):
        print(f"  [!] Warning: Could not find chapter file '{txt_filename}'. Skipping.")
        return False

    base_name = os.path.splitext(os.path.basename(txt_filename))[0]
    chapter_output_dir = os.path.join(master_output_dir, "chunks_" + base_name)
    final_mp3 = os.path.join(master_output_dir, f"{base_name}_complete.mp3")
    timestamp_file = os.path.join(master_output_dir, f"{base_name}_timestamps.txt")

    if os.path.exists(final_mp3) and os.path.getsize(final_mp3) > 0:
        print(f"\n--> Section '{base_name}' MP3 already exists in output folder. Skipping.")
        return True

    os.makedirs(chapter_output_dir, exist_ok=True)

    with open(txt_filename, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunks = split_text_into_tagged_blocks(raw_text, default_character="narrator", characters_map=characters_map, tag_mappings=tag_mappings)
    print(f"\n--> Processing Hybrid Chapter: '{txt_filename}' ({len(chunks)} blocks)")

    saved_files = []
    chunk_metadata = []
    chapter_silence_file = None

    for i, chunk in enumerate(chunks):
        chunk_filename = os.path.join(chapter_output_dir, f"chunk_{i + 1:03d}.wav")
        c_type = chunk["type"]
        char_name = chunk["character"]

        # Resolve voice configuration and exact Kokoro / F5 voice name
        canonical_char = tag_mappings.get(char_name.lower(), char_name.lower()) if tag_mappings else char_name.lower()
        char_cfg = characters_map.get(canonical_char, characters_map.get("narrator", {}))
        engine = char_cfg.get("engine", "kokoro").lower()
        pitch_factor = float(char_cfg.get("pitch", 1.0))

        if c_type == "DIVIDER":
            voice_label = "2s Silence (Section Divider)"
            engine_label = "SILENCE"
        elif engine in ["f5tts", "f5-tts"]:
            ref_audio_name = os.path.basename(char_cfg.get("ref_audio", "sample.wav"))
            voice_label = f"f5tts [ref: {ref_audio_name}]"
            engine_label = "F5-TTS"
        else:
            # Kokoro engine: capture exact voice name and any blend formula
            kokoro_voice = char_cfg.get("voice", "af_bella")
            if pitch_factor != 1.0:
                voice_label = f"{kokoro_voice} (pitch: {pitch_factor})"
            else:
                voice_label = kokoro_voice
            engine_label = "KOKORO"

        # Resume support: skip if already rendered
        if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 0:
            print(f"    [Chunk {i + 1}/{len(chunks)}] Found existing on disk. Resuming...")
            saved_files.append(chunk_filename)
            duration = get_wav_duration(chunk_filename)
            chunk_metadata.append({
                "index": i + 1,
                "filename": f"chunk_{i + 1:03d}.wav",
                "character": char_name,
                "engine": engine_label,
                "voice": voice_label,
                "duration": duration,
            })
            continue

        # Handle Section Divider (2s silence)
        if c_type == "DIVIDER":
            print(f"    [Chunk {i + 1}/{len(chunks)}] Section Divider (2s Pause)")
            if not chapter_silence_file:
                chapter_silence_file = generate_silence_file(master_output_dir)

            with open(chapter_silence_file, "rb") as sf_in:
                silence_data = sf_in.read()
            with open(chunk_filename, "wb") as out_f:
                out_f.write(silence_data)

            saved_files.append(chunk_filename)
            duration = get_wav_duration(chunk_filename)
            chunk_metadata.append({
                "index": i + 1,
                "filename": f"chunk_{i + 1:03d}.wav",
                "character": "DIVIDER",
                "engine": "SILENCE",
                "voice": voice_label,
                "duration": duration,
            })
            continue

        # Voice generation
        text_to_speak = chunk["text"]
        print(f"    [Chunk {i + 1}/{len(chunks)}] [{char_name.upper()} via {engine_label}] (Voice: {voice_label}) ({len(text_to_speak)} chars)...")

        temp_raw = os.path.join(chapter_output_dir, f"temp_raw_{i + 1}.wav")

        try:
            if engine in ["f5tts", "f5-tts"]:
                render_f5tts_chunk(text_to_speak, char_cfg, temp_raw, chapter_output_dir)
            else:
                # Default: Kokoro
                render_kokoro_chunk(text_to_speak, kokoro_voice, temp_raw)

            # Apply rubberband pitch shift if specified
            if pitch_factor != 1.0:
                apply_pitch_shift(temp_raw, chunk_filename, pitch_factor)
                if os.path.exists(temp_raw):
                    os.remove(temp_raw)
            else:
                os.replace(temp_raw, chunk_filename)

            saved_files.append(chunk_filename)
            duration = get_wav_duration(chunk_filename)
            chunk_metadata.append({
                "index": i + 1,
                "filename": f"chunk_{i + 1:03d}.wav",
                "character": char_name,
                "engine": engine_label,
                "voice": voice_label,
                "duration": duration,
            })

            time.sleep(0.1)

        except Exception as e:
            print(f"    [X] Error rendering chunk {i + 1} with character '{char_name}': {e}")
            if os.path.exists(temp_raw):
                try: os.remove(temp_raw)
                except Exception: pass
            return False

    # Stitch into master MP3
    list_file = os.path.join(chapter_output_dir, "file_list.txt")
    with open(list_file, "w", encoding="utf-8") as lf:
        for sf_path in saved_files:
            lf.write(f"file '{os.path.basename(sf_path)}'\n")

    print(f"    [*] Merging {len(saved_files)} chunks into '{final_mp3}'...")
    cmd = (
        f'ffmpeg -y -f concat -safe 0 -i "{list_file}" -codec:a libmp3lame'
        f' -qscale:a 2 "{final_mp3}"'
    )

    if os.system(cmd) == 0:
        print(f"    [✓] Success! Saved '{final_mp3}'.")

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
                    tf.write(f"  • [{c_name}] (KOKORO): {v_str}\n")
                else:
                    ref_f = os.path.basename(c_info.get("ref_audio", "sample.wav"))
                    tf.write(f"  • [{c_name}] (F5-TTS): {ref_f}\n")
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
        print(f"    [X] FFmpeg merge failed for {txt_filename}.")
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
    print("========================================")
    print("   Hybrid Kokoro + F5-TTS Audiobook     ")
    print("========================================")

    if not os.path.exists(MANIFEST_FILE):
        print(f"Error: Could not find manifest file '{MANIFEST_FILE}'.")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
        chapter_files = [line.strip() for line in mf if line.strip() and not line.startswith("#")]

    if not chapter_files:
        print(f"Error: Manifest file '{MANIFEST_FILE}' is empty.")
        return

    # Load character profiles and tag mappings
    characters = load_characters()
    tag_mappings = load_tag_mappings()
    print(f"Loaded {len(characters)} character profiles from '{CHARACTERS_FILE}':")
    for name, cfg in characters.items():
        engine = cfg.get("engine", "kokoro").upper()
        details = cfg.get("voice", "") if engine == "KOKORO" else cfg.get("ref_audio", "")
        print(f"  • [{name}]: {engine} -> {details}")

    first_file_base = os.path.splitext(os.path.basename(chapter_files[0]))[0]
    cleaned_first = re.sub(r"^\d+_", "", first_file_base)
    book_title_base = re.sub(r"^intro_", "", cleaned_first)

    master_output_dir = f"{book_title_base}_hybrid_audiobook_output"

    # If already moved to finished in a previous run, bring back to workspace for editing
    finished_candidate = os.path.join(FINISHED_DIR, master_output_dir)
    if not os.path.exists(master_output_dir) and os.path.exists(finished_candidate):
        print(f"\n[*] Found previously finished recording in '{FINISHED_DIR}/'. Loading into workspace...")
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
            chunks_count = len([f for f in glob.glob(os.path.join(chunk_dir, "chunk_*.wav")) if os.path.getsize(f) > 0])

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
        print("\n" + "=" * 70)
        print("                 Existing Audio Sections Detected")
        print("=" * 70)
        for item in scan_results:
            print(f"  [{item['idx']:02d}] {item['base_name']:<40} -> [{item['status']}]")
        print("=" * 70)
        print("Options:")
        print("  [1] Resume / Keep existing (skip completed, finish partials) [Default]")
        print("  [2] Re-record EVERYTHING from scratch")
        print("  [3] Choose specific sections to re-record")
        choice = input("\nSelect how to proceed [1, 2, or 3] (default 1): ").strip()

        if choice == "2":
            print("\n[*] Clearing ALL previous recordings. Re-recording entire book from scratch...")
            for item in scan_results:
                clear_chapter_recording(master_output_dir, item["base_name"])
            print("[✓] All sections cleared.")
        elif choice == "3":
            print("\nEnter section number(s) to re-record (e.g. '1, 3' or '2-4' or 'all'):")
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
                print(f"\n[✓] Scheduled {len(cleared_names)} section(s) for fresh re-recording:")
                for name in cleared_names:
                    print(f"    - {name}")
                print("[*] All other completed sections will be kept and skipped automatically.\n")
            else:
                print("[-] No valid sections entered. Resuming standard progress.")

    # ==================== MAIN PROCESSING LOOP ====================
    for idx, txt_file in enumerate(chapter_files, 1):
        print(f"\n[{idx}/{len(chapter_files)}] Processing chapter: {txt_file}")
        success = process_hybrid_chapter(txt_file, characters, master_output_dir, tag_mappings=tag_mappings)
        if not success:
            print(f"[!] Pausing due to error in {txt_file}.")
            break

    # Package into ZIP archive if multiple chapters
    if len(chapter_files) > 1:
        audio_zip_filename = os.path.join(master_output_dir, f"{book_title_base}_hybrid_audiobook.zip")
        print(f"\n[*] Packaging complete audio MP3s into '{audio_zip_filename}'...")
        with zipfile.ZipFile(audio_zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for txt_file in chapter_files:
                base_name = os.path.splitext(os.path.basename(txt_file))[0]
                final_mp3 = os.path.join(master_output_dir, f"{base_name}_complete.mp3")
                if os.path.exists(final_mp3):
                    zipf.write(final_mp3, arcname=os.path.basename(final_mp3))
        print(f"[✓] Created '{audio_zip_filename}'!")

    # Check if all chapters in manifest are finished
    all_finished = all(
        os.path.exists(os.path.join(master_output_dir, f"{os.path.splitext(os.path.basename(f))[0]}_complete.mp3"))
        for f in chapter_files
    )

    if all_finished:
        os.makedirs(FINISHED_DIR, exist_ok=True)
        dest_folder = os.path.join(FINISHED_DIR, os.path.basename(master_output_dir))
        print(f"\n[*] All chapters complete! Moving finished audiobook folder to '{FINISHED_DIR}/'...")
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
        shutil.move(master_output_dir, dest_folder)
        print(f"[✓] Successfully archived in: '{dest_folder}'")

    print("\n========================================")
    print("   Hybrid Audiobook Processing Finished! ")
    print("========================================")


if __name__ == "__main__":
    main()
