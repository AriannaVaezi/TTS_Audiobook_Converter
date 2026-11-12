import os
import re
import sys
import time
import json
import queue
import subprocess
import shutil
import zipfile
import threading
import numpy as np
import soundfile as sf
import torch
if hasattr(torch, 'set_num_threads'):
    try: torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception: pass
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
MAX_CHARS_PER_F5_CHUNK = 100  # Compact F5 chunk size (~1-2 sentences) for easy targeted re-recording
MIN_CHARS_PER_F5_CHUNK = 25   # Avoid isolating tiny words ("Yes.", "No.", "Okay.") as standalone chunks
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
                "volume": 1.0,
                "description": "Default narrator voice on CPU for general story text."
            },
            "detective": {
                "engine": "kokoro",
                "voice": "am_michael",
                "pitch": 0.95,
                "speed": 1.0,
                "volume": 1.0,
                "description": "Example male character using Kokoro on CPU."
            },
            "villain": {
                "engine": "f5tts",
                "ref_audio": "wavs/output_custom_0.wav",
                "ref_text": "Kokoro is running locally with your custom weighted voice blend. I need this audio clip for audio later",
                "speed": 1.0,
                "pitch": 1.0,
                "steps": 32,
                "volume": 1.0,
                "emotions": {},
                "alts": {},
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


def normalize_chunk_volume(wav_path, emotion="", volume_scale=1.0, max_peak=0.88, auto_balance=True):
    """
    Balances active-speech loudness across Kokoro and F5-TTS chunks and applies per-voice
    volume multipliers cleanly without clipping.
    When auto_balance=False, preserves natural relative amplitude and applies volume_scale directly.
    """
    try:
        data, sr = sf.read(wav_path)
    except Exception:
        return
    if data.size == 0:
        return

    vol = float(volume_scale)
    if not auto_balance:
        if abs(vol - 1.0) < 1e-4:
            return
        scaled = data * max(0.05, min(4.0, vol))
        peak = float(np.max(np.abs(scaled)))
        if peak > 0.95:
            scaled = scaled * (0.95 / peak)
        sf.write(wav_path, scaled, sr)
        return

    mono = data.mean(axis=1) if data.ndim > 1 else data
    win = max(1, int(0.05 * sr))
    n_frames = len(mono) // win
    if n_frames > 0:
        frames = mono[: n_frames * win].reshape(n_frames, win)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        thresh = max(0.005, 0.15 * float(np.max(frame_rms)))
        active = frame_rms[frame_rms >= thresh]
        cur_rms = float(np.sqrt(np.mean(active ** 2))) if len(active) > 0 else float(np.sqrt(np.mean(mono ** 2)))
    else:
        cur_rms = float(np.sqrt(np.mean(mono ** 2)))

    if cur_rms < 1e-4:
        return

    emo = (emotion or "").strip().lower()
    if emo in ("whisper", "whispering"):
        target_rms = 0.052
    elif emo in ("shout", "shouting", "screaming", "yelling"):
        target_rms = 0.092
    else:
        target_rms = 0.075

    target_rms *= vol
    gain = max(0.35, min(3.0, target_rms / cur_rms))

    scaled = data * gain
    peak = float(np.max(np.abs(scaled)))
    if peak > max_peak:
        scaled = scaled * (max_peak / peak)

    sf.write(wav_path, scaled, sr)


WARMTH_EQ_FILTER = (
    "highpass=f=65,"
    "equalizer=f=175:t=q:w=0.9:g=3.5,"
    "equalizer=f=3100:t=q:w=1.0:g=-3.0,"
    "treble=g=-3.5:f=5500,"
    "lowpass=f=9800"
)


def apply_warm_eq(wav_path):
    """
    Applies a subtle studio warmth curve to remove thin/tinny F5-TTS Vocos resonance
    on headphones (adds chest body at 175 Hz, tames harsh 3.1 kHz & >5.5 kHz metallic hiss).
    """
    temp_eq = wav_path + ".eq.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-af", WARMTH_EQ_FILTER, temp_eq]
    try:
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and os.path.exists(temp_eq) and os.path.getsize(temp_eq) > 0:
            os.replace(temp_eq, wav_path)
    except Exception:
        pass
    finally:
        if os.path.exists(temp_eq):
            try:
                os.remove(temp_eq)
            except Exception:
                pass


def _finalize_rendered_chunk(temp_raw, chunk_file, task, normalize_volume=True):
    """Applies optional pitch shift, studio warmth EQ (for F5-TTS), and volume adjustment."""
    pitch = task["pitch_factor"]
    if pitch != 1.0:
        apply_pitch_shift(temp_raw, chunk_file, pitch)
        if os.path.exists(temp_raw):
            os.remove(temp_raw)
    else:
        os.replace(temp_raw, chunk_file)

    if task.get("apply_warmth", False):
        apply_warm_eq(chunk_file)

    normalize_chunk_volume(
        chunk_file,
        emotion=task.get("emotion", ""),
        volume_scale=task.get("volume_scale", 1.0),
        auto_balance=normalize_volume,
    )


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

def _collect_variants(char_cfg):
    """Returns (emotions_dict, alts_dict) with lowercase keys for case-insensitive lookup."""
    raw_emo = char_cfg.get("emotions", {}) or {}
    emotions = {str(k).strip().lower(): v for k, v in raw_emo.items()} if isinstance(raw_emo, dict) else {}

    alts = {}
    for field in ("alts", "alt_voices", "alt"):
        raw_alt = char_cfg.get(field, {})
        if isinstance(raw_alt, dict):
            for k, v in raw_alt.items():
                alts[str(k).strip().lower()] = v
    return emotions, alts


def get_character_variants(char_cfg):
    """Returns list of (variant_name, kind, entry_value) across emotions and alt voices."""
    emotions, alts = _collect_variants(char_cfg)
    variants = []
    for k, v in emotions.items():
        variants.append((k, "emotion", v))
    for k, v in alts.items():
        variants.append((k, "alt", v))
    return variants


def _is_alt_variant(char_name, variant_name, characters_map):
    """Checks if a tag suffix resolves to an explicit alt voice for the character."""
    if not variant_name or not characters_map:
        return False
    cfg = characters_map.get(char_name.lower(), {})
    _, alts = _collect_variants(cfg)
    return variant_name.strip().lower() in alts


def _apply_variant_entry(resolved, variant_key, variant_kind, matched_entry):
    """Applies a matched emotion or alt-voice override dict/string onto resolved voice config."""
    if isinstance(matched_entry, dict):
        for k, v in matched_entry.items():
            resolved[k] = v
        if "engine" not in matched_entry:
            if "ref_audio" in matched_entry and "voice" not in matched_entry:
                resolved["engine"] = "f5tts"
            elif "voice" in matched_entry and "ref_audio" not in matched_entry:
                resolved["engine"] = "kokoro"
    elif isinstance(matched_entry, str):
        eng = str(resolved.get("engine", "kokoro")).lower()
        looks_like_file = matched_entry.lower().endswith((".wav", ".mp3", ".flac", ".ogg")) or "/" in matched_entry or "\\" in matched_entry
        if eng not in ["f5tts", "f5-tts"] and not looks_like_file:
            resolved["voice"] = matched_entry
        else:
            resolved["ref_audio"] = matched_entry
            if looks_like_file:
                resolved["engine"] = "f5tts"
    resolved["active_emotion"] = variant_key
    resolved["active_variant_kind"] = variant_kind
    return resolved


def resolve_character_emotion_config(char_cfg, emotion="", enable_emotions=True, force_variant=False):
    """
    Resolves voice parameters applying emotion and/or alt-voice overrides.
    - Alt voices ('alts' / 'alt_voices') always apply when tagged or chosen manually.
    - Emotion voices ('emotions') apply when enable_emotions=True or force_variant=True.
    """
    resolved = dict(char_cfg)
    resolved["speed"] = float(char_cfg.get("speed", 1.0))
    resolved["pitch"] = float(char_cfg.get("pitch", 1.0))
    resolved["volume"] = float(char_cfg.get("volume", 1.0))

    if not emotion:
        return resolved

    variant = str(emotion).strip().lower()
    emotions, alts = _collect_variants(char_cfg)

    # Check alt voices first (always active even if acoustic emotions are off)
    if variant in alts:
        return _apply_variant_entry(resolved, variant, "alt", alts[variant])

    if not enable_emotions and not force_variant:
        return resolved

    EMOTION_ALIASES = {
        "shouting": "shout",
        "whispering": "whisper",
        "screaming": "shout",
        "yelling": "shout",
        "crying": "sad",
        "laughing": "happy",
    }
    lookup_keys = [variant, EMOTION_ALIASES.get(variant, variant)]
    for k in lookup_keys:
        if k in emotions:
            return _apply_variant_entry(resolved, variant, "emotion", emotions[k])

    return resolved


def write_production_voice_roster(master_output_dir, book_title_base, characters_map):
    """Writes a standalone master voice roster text file documenting all active voices."""
    roster_path = os.path.join(master_output_dir, f"{book_title_base}_voice_roster.txt")
    with open(roster_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write(f" PRODUCTION VOICE ROSTER: {book_title_base.replace('_', ' ')}\n")
        f.write("======================================================================\n\n")

        # 1. GPU F5-TTS Voices
        f5_chars = [c for c in characters_map.values() if c.get("engine", "").lower() in ["f5tts", "f5-tts"]]
        f.write(f"GPU ENGINE: F5-TTS (Flow-Matching Voice Cloning) [{len(f5_chars)} voice(s)]\n")
        f.write("-" * 70 + "\n")
        if f5_chars:
            for c in f5_chars:
                disp = c.get("display_name", "Unknown")
                ref = c.get("ref_audio", "N/A")
                steps = c.get("steps", 32)
                speed = c.get("speed", 1.0)
                pitch = c.get("pitch", 1.0)
                volume = c.get("volume", 1.0)
                desc = c.get("description", "")
                f.write(f"  • [{disp}]\n")
                f.write(f"      Ref Audio : {ref}\n")
                f.write(f"      Settings  : steps={steps}, speed={speed}, pitch={pitch}, volume={volume}\n")
                if desc:
                    f.write(f"      Notes     : {desc}\n")
                emotions, alts = _collect_variants(c)
                if emotions:
                    f.write("      Emotion Ref Clips:\n")
                    for emo_k, emo_v in emotions.items():
                        r_clip = emo_v.get("ref_audio") if isinstance(emo_v, dict) else emo_v
                        f.write(f"        - [{emo_k}]: {r_clip}\n")
                if alts:
                    f.write("      Alt Voices:\n")
                    for alt_k, alt_v in alts.items():
                        r_clip = (alt_v.get("ref_audio") or alt_v.get("voice")) if isinstance(alt_v, dict) else alt_v
                        f.write(f"        - [{alt_k}]: {r_clip}\n")
                f.write("\n")
        else:
            f.write("  (No F5-TTS characters configured)\n\n")

        # 2. CPU Kokoro Voices
        kokoro_chars = [c for c in characters_map.values() if c.get("engine", "").lower() not in ["f5tts", "f5-tts"]]
        f.write(f"CPU ENGINE: Kokoro TTS (Neural Speech Synthesis) [{len(kokoro_chars)} voice(s)]\n")
        f.write("-" * 70 + "\n")
        for c in kokoro_chars:
            disp = c.get("display_name", "Unknown")
            voice = c.get("voice", "af_bella")
            speed = c.get("speed", 1.0)
            pitch = c.get("pitch", 1.0)
            volume = c.get("volume", 1.0)
            desc = c.get("description", "")
            f.write(f"  • [{disp}]\n")
            f.write(f"      Voice     : {voice}\n")
            f.write(f"      Settings  : speed={speed}, pitch={pitch}, volume={volume}\n")
            if desc:
                f.write(f"      Notes     : {desc}\n")
            emotions, alts = _collect_variants(c)
            if emotions:
                f.write(f"      Emotions  : {', '.join(emotions.keys())}\n")
            if alts:
                f.write(f"      Alt Voices: {', '.join(alts.keys())}\n")
            f.write("\n")

        f.write("======================================================================\n")
    return roster_path


def split_text_into_tagged_blocks(text, default_character="narrator", characters_map=None, tag_mappings=None, enable_emotions=True):
    """
    Splits text into chunks assigned to specific characters and optional emotion/tone.
    Recognizes:
      1. Section dividers: ***, ---, * * *
      2. Block tags:
           [villain] or [villain:angry]
           Dialogue or paragraphs here
           [/villain]
      3. Inline tags:
           Narration [Hero:whisper] "Stay back," [/Hero] narration.
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

    def resolve_variant(char_key, raw_tag):
        if not raw_tag:
            return ""
        v = raw_tag.strip().lower()
        if enable_emotions or _is_alt_variant(char_key, v, characters_map):
            return v
        return ""

    current_char = resolve_char(default_character)
    current_emotion = ""
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
                "emotion": "",
                "text": "DIVIDER"
            })
            continue

        # 2. Check for closing block tag: [/character] or [/character:emotion]
        close_match = re.match(r"^\[\s*/\s*([a-zA-Z0-9_\-]+)(?::[a-zA-Z0-9_\-]+)?\s*\]$", raw_line)
        if close_match:
            in_explicit_block = False
            current_char = resolve_char(default_character)
            current_emotion = ""
            continue

        # 3. Check for opening block tag: [character] or [character:emotion] alone on a line
        open_block_match = re.match(r"^\[\s*([a-zA-Z0-9_\-]+)(?::([a-zA-Z0-9_\-]+))?\s*\]$", raw_line)
        if open_block_match:
            in_explicit_block = True
            current_char = resolve_char(open_block_match.group(1))
            current_emotion = resolve_variant(current_char, open_block_match.group(2))
            continue

        # 4. Check for embedded inline tags: e.g. Narration [Hero:whisper] "Dialogue" [/Hero] Narration
        inline_tag_pattern = re.compile(
            r"\[\s*([a-zA-Z0-9_\-]+)(?::([a-zA-Z0-9_\-]+))?\s*\](.*?)\[\s*/\s*\1(?::[a-zA-Z0-9_\-]+)?\s*\]"
        )
        if not in_explicit_block and inline_tag_pattern.search(raw_line):
            pos = 0
            for match in inline_tag_pattern.finditer(raw_line):
                start, end = match.span()
                before = raw_line[pos:start].strip()
                if before:
                    blocks.append({
                        "type": "VOICE",
                        "character": current_char,
                        "emotion": current_emotion,
                        "text": before
                    })
                char_name = resolve_char(match.group(1))
                emotion = resolve_variant(char_name, match.group(2))
                dialogue = match.group(3).strip()
                if dialogue:
                    blocks.append({
                        "type": "VOICE",
                        "character": char_name,
                        "emotion": emotion,
                        "text": dialogue
                    })
                pos = end
            after = raw_line[pos:].strip()
            if after:
                blocks.append({
                    "type": "VOICE",
                    "character": current_char,
                    "emotion": current_emotion,
                    "text": after
                })
            continue

        # 5. Check for single prefix tag without closing: [character:emotion] Rest of text...
        prefix_match = re.match(r"^\[\s*([a-zA-Z0-9_\-]+)(?::([a-zA-Z0-9_\-]+))?\s*\]\s*(.*)$", raw_line)
        if prefix_match and not in_explicit_block:
            char_name = resolve_char(prefix_match.group(1))
            emotion = resolve_variant(char_name, prefix_match.group(2))
            content = prefix_match.group(3).strip()
            if content:
                blocks.append({
                    "type": "VOICE",
                    "character": char_name,
                    "emotion": emotion,
                    "text": content
                })
            continue

        # 6. Regular text inside or outside an explicit block
        blocks.append({
            "type": "VOICE",
            "character": current_char,
            "emotion": current_emotion if in_explicit_block else "",
            "text": raw_line
        })

    def is_f5_block(char_name, emotion_name=""):
        char_cfg = (characters_map or {}).get(char_name.lower(), {})
        if emotion_name:
            char_cfg = resolve_character_emotion_config(char_cfg, emotion_name, enable_emotions=enable_emotions)
        return char_cfg.get("engine", "").lower() in ["f5tts", "f5-tts"] or "f5" in char_name.lower()

    # Split F5-TTS blocks into compact (~140 char) sentence-grouped chunks while keeping short
    # replies ("Yes.", "No.", "Okay.") grouped with adjacent lines; merge Kokoro up to 1200 chars.
    merged_chunks = []
    for b in blocks:
        if b["type"] == "DIVIDER":
            merged_chunks.append(b)
            continue

        char = b["character"]
        emotion = b.get("emotion", "")
        text_piece = b["text"]
        f5_block = is_f5_block(char, emotion)

        pieces = (
            split_sentences(text_piece, max_len=MAX_CHARS_PER_F5_CHUNK, min_chunk_len=MIN_CHARS_PER_F5_CHUNK)
            if f5_block
            else [text_piece]
        )

        for piece in pieces:
            if not piece:
                continue
            if (merged_chunks and merged_chunks[-1]["type"] == "VOICE"
                    and merged_chunks[-1]["character"] == char
                    and merged_chunks[-1].get("emotion", "") == emotion):
                prev_len = len(merged_chunks[-1]["text"])
                max_limit = MAX_CHARS_PER_F5_CHUNK if f5_block else MAX_CHARS_PER_KOKORO_CHUNK
                can_merge = (prev_len + len(piece) + 2 <= max_limit)
                if f5_block and not can_merge:
                    # Merge short trailing/leading words ("Yes.", "Okay.") unless chunk grows overly large
                    if (prev_len < MIN_CHARS_PER_F5_CHUNK or len(piece) < MIN_CHARS_PER_F5_CHUNK) and (prev_len + len(piece) + 2 <= MAX_CHARS_PER_F5_CHUNK + MIN_CHARS_PER_F5_CHUNK):
                        can_merge = True
                if can_merge:
                    merged_chunks[-1]["text"] += (" " if f5_block else "\n\n") + piece
                    continue

            merged_chunks.append({
                "type": "VOICE",
                "character": char,
                "emotion": emotion,
                "text": piece
            })

    return merged_chunks


_SENTENCE_ABBREVS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m."
}


def _tokenize_sentences(text):
    """Splits text into clean sentence strings while preserving abbreviations, mid-sentence ellipses, and quotes."""
    cleaned = text.strip()
    if not cleaned:
        return []

    raw_sentences = []
    start = 0
    for m in re.finditer(r'([.!?…]+["\'\)\]\u201d\u2019]*)\s+', cleaned):
        punct = m.group(1)
        before = cleaned[start:m.end(1)].rstrip()
        after = cleaned[m.end():]
        last_word = before.split()[-1].strip('"\'(\u201c').lower() if before.split() else ""
        if last_word in _SENTENCE_ABBREVS:
            continue
        # Don't split on an ellipsis/period followed immediately by a lowercase continuation word
        if ("..." in punct or "…" in punct or punct.startswith(".")) and "!" not in punct and "?" not in punct:
            if re.match(r'^["\'\(\u201c\u2018]*[a-z]', after):
                continue
        piece = before.strip()
        if piece:
            raw_sentences.append(piece)
        start = m.end()

    tail = cleaned[start:].strip()
    if tail:
        raw_sentences.append(tail)

    # Fold pure non-word punctuation fragments into neighboring sentence instead of emitting empty audio
    sentences = []
    for piece in raw_sentences:
        if not re.search(r'[A-Za-z0-9]', piece) and sentences:
            sentences[-1] = (sentences[-1] + " " + piece).strip()
        else:
            sentences.append(piece)
    return sentences or [cleaned]


def split_sentences(text, max_len=250, min_chunk_len=0):
    """
    Splits text on sentence boundaries.
    - When max_len is None or <= 0: returns one sentence per item.
    - When max_len > 0: groups consecutive sentences up to max_len characters, folding tiny
      sentences (< min_chunk_len, e.g. 'Yes.', 'No.', 'Okay.') with adjacent text cleanly.
    """
    sentences = _tokenize_sentences(text)
    if not max_len or max_len <= 0:
        return sentences

    chunks = []
    current = ""
    for s in sentences:
        if not s:
            continue
        if not current:
            current = s
        elif len(current) + len(s) + 1 <= max_len:
            current = current + " " + s
        elif min_chunk_len > 0 and (len(current) < min_chunk_len or len(s) < min_chunk_len) and (len(current) + len(s) + 1 <= max_len + min_chunk_len):
            current = current + " " + s
        else:
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


def _sanitize_f5_punctuation(text):
    """Normalizes exclamation punctuation to prevent F5-TTS exclamation screaming distortion."""
    if not text:
        return text
    # Convert !? / ?! to ?, and ! to ., then collapse accidental duplicates
    cleaned = re.sub(r"(?:!\?|\?!)+", "?", text)
    cleaned = re.sub(r"!+", ".", cleaned)
    cleaned = re.sub(r"(?<!\.)\.\.(?!\.)", ".", cleaned)
    return re.sub(r"([?!])\.+", r"\1", cleaned).strip()


def render_f5tts_chunk_local(text, char_cfg, output_wav, temp_dir, chunk_index):
    """Renders a text chunk using local GPU F5-TTS with clean punctuation & peak protection."""
    f5 = get_f5tts_engine()

    ref_audio = char_cfg.get("ref_audio", "ref_sample.wav")
    ref_text = _sanitize_f5_punctuation(char_cfg.get("ref_text", ""))
    speed = float(char_cfg.get("speed", 1.0))
    steps = int(char_cfg.get("steps", 32))
    active_emotion = str(char_cfg.get("active_emotion", "")).lower()

    # Lower CFG and target RMS slightly on intense emotions to prevent vocoder crunch on shouted/raspy refs
    is_intense = active_emotion in ("shout", "shouting", "screaming", "yelling", "angry")
    cfg_strength = float(char_cfg.get("cfg_strength", 1.3 if is_intense else 1.8))
    target_rms = float(char_cfg.get("target_rms", 0.08 if is_intense else 0.1))

    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"F5-TTS reference audio '{ref_audio}' does not exist.")

    # Guard check: Warn if reference audio is > 11.5s to prevent F5-TTS internal clipping & text bleed
    try:
        ref_dur = get_wav_duration(ref_audio)
        if ref_dur > 11.5:
            safe_print(f"  [!] Note: Reference audio '{ref_audio}' is {ref_dur:.1f}s (> 11.5s). F5-TTS clips audio at 12s which causes trailing reference text to bleed into output. Optimal duration is 5-10s.")
    except Exception:
        pass

    sentences = [s.strip() for s in split_sentences(text, max_len=MAX_CHARS_PER_F5_CHUNK) if s.strip()]
    audio_segments = []
    target_sr = 24000
    pause = np.zeros(int(0.18 * target_sr), dtype=np.float32)

    for sentence in sentences:
        clean_sentence = _sanitize_f5_punctuation(sentence)
        if not clean_sentence:
            continue
        wav, sr, _ = f5.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=clean_sentence,
            file_wave=None,
            speed=speed,
            nfe_step=steps,
            cfg_strength=cfg_strength,
            target_rms=target_rms,
            show_info=lambda *args, **kwargs: None,
            progress=None
        )
        target_sr = sr
        if audio_segments:
            audio_segments.append(pause)
        audio_segments.append(wav)

    full_wav = np.concatenate(audio_segments) if audio_segments else np.zeros(target_sr, dtype=np.float32)

    # Prevent PCM clipping distortion when loud reference clips overshoot 1.0
    peak = float(np.max(np.abs(full_wav))) if len(full_wav) else 0.0
    if peak > 0.88:
        full_wav = full_wav * (0.88 / peak)

    sf.write(output_wav, full_wav, target_sr)


# ==================== PARALLEL CHAPTER PROCESSING ====================

CHUNK_OVERRIDES_FILENAME = "chunk_overrides.json"


def load_chunk_overrides(chunk_dir):
    """Loads per-chunk voice/emotion/alt/volume overrides stored inside a chapter's chunks directory."""
    ov_path = os.path.join(chunk_dir, CHUNK_OVERRIDES_FILENAME)
    if not os.path.exists(ov_path):
        return {}
    try:
        with open(ov_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        return {}


def save_chunk_overrides(chunk_dir, overrides):
    """Persists per-chunk voice/emotion/alt/volume overrides inside a chapter's chunks directory."""
    os.makedirs(chunk_dir, exist_ok=True)
    ov_path = os.path.join(chunk_dir, CHUNK_OVERRIDES_FILENAME)
    with open(ov_path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=4)


def apply_chunk_override_action(
    txt_filename,
    master_output_dir,
    chunk_indices,
    new_char=None,
    new_variant=None,
    set_variant=False,
    new_volume=None,
    clear_override=False,
):
    """
    Invalidates selected chunks (and the merged chapter MP3/timestamps) so they are freshly
    re-recorded on the next render pass, optionally overriding or clearing character,
    emotion/alt voice, and/or volume multiplier persistently in chunk_overrides.json.
    """
    base_name = os.path.splitext(os.path.basename(txt_filename))[0]
    chunk_dir = os.path.join(master_output_dir, "chunks_" + base_name)
    final_mp3 = os.path.join(master_output_dir, f"{base_name}_complete.mp3")
    timestamp_file = os.path.join(master_output_dir, f"{base_name}_timestamps.txt")

    os.makedirs(chunk_dir, exist_ok=True)
    overrides = load_chunk_overrides(chunk_dir)
    ov_path = os.path.join(chunk_dir, CHUNK_OVERRIDES_FILENAME)

    updated = []
    for c_idx in chunk_indices:
        c_key = str(c_idx)
        wav_p = os.path.join(chunk_dir, f"chunk_{c_idx:03d}.wav")
        if os.path.exists(wav_p):
            try:
                os.remove(wav_p)
            except Exception:
                pass

        if clear_override:
            overrides.pop(c_key, None)
        elif new_char is not None or set_variant or new_volume is not None:
            entry = dict(overrides.get(c_key, {}))
            if new_char is not None:
                entry["character"] = new_char.lower()
            if set_variant:
                entry["variant"] = (new_variant or "").strip().lower()
            if new_volume is not None:
                entry["volume"] = float(new_volume)
            overrides[c_key] = entry
        updated.append(c_idx)

    if overrides:
        save_chunk_overrides(chunk_dir, overrides)
    elif clear_override and os.path.exists(ov_path):
        try:
            os.remove(ov_path)
        except Exception:
            pass

    # Remove only the merged chapter MP3 & timestamps so untouched chunks resume instantly
    for artifact in (final_mp3, timestamp_file):
        if os.path.exists(artifact):
            try:
                os.remove(artifact)
            except Exception:
                pass

    return updated


def process_hybrid_chapter_simultaneous(txt_filename, characters_map, master_output_dir, tag_mappings=None, enable_emotions=True, normalize_volume=True):
    """
    Processes a chapter by running F5-TTS on GPU and Kokoro on CPU simultaneously,
    featuring an interactive chunk progress bar, emotion/tone modulation, and
    case-insensitive master voice roster logging.
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

    chunks = split_text_into_tagged_blocks(raw_text, default_character="narrator", characters_map=characters_map, tag_mappings=tag_mappings, enable_emotions=enable_emotions)
    safe_print(f"\n--> Processing Simultaneous Hybrid Chapter: '{txt_filename}' ({len(chunks)} blocks)")

    # Prepare shared data structures
    kokoro_queue = queue.Queue()
    f5_queue = queue.Queue()
    completed_chunks = {}
    worker_errors = []

    chapter_silence_file = None
    all_tasks = []

    # Initialize tqdm progress bar
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    progress_bar = None
    if tqdm is not None:
        short_title = base_name[:22]
        progress_bar = tqdm(
            total=len(chunks),
            desc=f"Chapter Chunks [{short_title}]",
            unit="chunk",
            dynamic_ncols=True,
            leave=True
        )

    progress_lock = threading.Lock()

    def record_chunk_done(meta, desc_info=""):
        with progress_lock:
            completed_chunks[meta["index"]] = meta
            if progress_bar:
                if desc_info:
                    progress_bar.set_postfix_str(desc_info, refresh=False)
                progress_bar.update(1)

    def log_print(msg):
        if progress_bar:
            tqdm.write(msg)
        else:
            safe_print(msg)

    chunk_overrides = load_chunk_overrides(chapter_output_dir)

    # 1. Pre-scan all chunks and route
    for i, chunk in enumerate(chunks):
        chunk_idx = i + 1
        chunk_filename = os.path.join(chapter_output_dir, f"chunk_{chunk_idx:03d}.wav")
        c_type = chunk["type"]
        override = chunk_overrides.get(str(chunk_idx), {})
        char_name = override.get("character", chunk["character"])

        if "variant" in override:
            emotion = override["variant"]
            force_variant = True
        else:
            emotion = chunk.get("emotion", "")
            force_variant = False

        canonical_char = tag_mappings.get(char_name.lower(), char_name.lower()) if tag_mappings else char_name.lower()
        base_char_cfg = characters_map.get(canonical_char, characters_map.get("narrator", {}))
        char_cfg = resolve_character_emotion_config(
            base_char_cfg,
            emotion,
            enable_emotions=enable_emotions,
            force_variant=force_variant,
        )

        engine = char_cfg.get("engine", "kokoro").lower()
        pitch_factor = float(char_cfg.get("pitch", 1.0))
        speed_factor = float(char_cfg.get("speed", 1.0))
        volume_scale = float(override["volume"]) if "volume" in override else float(char_cfg.get("volume", 1.0))
        is_f5_engine = engine in ["f5tts", "f5-tts"]
        apply_warmth = bool(char_cfg.get("warmth", is_f5_engine))
        display_name = char_cfg.get("display_name", canonical_char.title())

        if c_type == "DIVIDER":
            voice_label = "2s Silence (Section Divider)"
            engine_label = "SILENCE"
        elif engine in ["f5tts", "f5-tts"]:
            ref_audio_name = os.path.basename(char_cfg.get("ref_audio", "sample.wav"))
            act_emo = char_cfg.get("active_emotion", "")
            act_kind = char_cfg.get("active_variant_kind", "")
            if act_emo:
                if act_kind == "alt":
                    voice_label = f"f5tts [alt: {act_emo} -> {ref_audio_name}]"
                else:
                    voice_label = f"f5tts [{act_emo}: {ref_audio_name}]"
            else:
                voice_label = f"f5tts [ref: {ref_audio_name}]"
            if abs(volume_scale - 1.0) > 1e-3:
                voice_label += f" (vol: {volume_scale:.2f}x)"
            engine_label = "F5-TTS (GPU)"
        else:
            kokoro_voice = char_cfg.get("voice", "af_bella")
            act_emo = char_cfg.get("active_emotion", "")
            act_kind = char_cfg.get("active_variant_kind", "")
            extras = []
            if act_emo:
                lbl = "alt" if act_kind == "alt" else "tone"
                extras.append(f"{lbl}: {act_emo}")
            elif emotion and (enable_emotions or force_variant):
                extras.append(f"tone: {emotion}")
            if pitch_factor != 1.0:
                extras.append(f"pitch: {pitch_factor}")
            if speed_factor != 1.0:
                extras.append(f"speed: {speed_factor}")
            if abs(volume_scale - 1.0) > 1e-3:
                extras.append(f"vol: {volume_scale:.2f}x")
            if extras:
                voice_label = f"{kokoro_voice} ({', '.join(extras)})"
            else:
                voice_label = kokoro_voice
            engine_label = "KOKORO (CPU)"

        task_info = {
            "index": chunk_idx,
            "total": len(chunks),
            "type": c_type,
            "character": display_name,
            "emotion": emotion,
            "char_cfg": char_cfg,
            "engine": engine,
            "engine_label": engine_label,
            "voice_label": voice_label,
            "pitch_factor": pitch_factor,
            "volume_scale": volume_scale,
            "apply_warmth": apply_warmth,
            "chunk_filename": chunk_filename,
            "text": chunk.get("text", "")
        }
        all_tasks.append(task_info)

        # Resume support: skip if already rendered
        if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 0:
            log_print(f"    [Chunk {chunk_idx:03d}/{len(chunks)}] Found existing on disk. Resuming...")
            meta = {
                "index": chunk_idx,
                "filename": f"chunk_{chunk_idx:03d}.wav",
                "character": display_name,
                "engine": engine_label,
                "voice": voice_label,
                "duration": get_wav_duration(chunk_filename)
            }
            record_chunk_done(meta, desc_info=f"Resume: {display_name}")
            continue

        # Immediate handling for Section Dividers
        if c_type == "DIVIDER":
            log_print(f"    [Chunk {chunk_idx:03d}/{len(chunks)}] Section Divider (2s Pause)")
            if not chapter_silence_file:
                chapter_silence_file = generate_silence_file(master_output_dir)

            with open(chapter_silence_file, "rb") as sf_in:
                silence_data = sf_in.read()
            with open(chunk_filename, "wb") as out_f:
                out_f.write(silence_data)

            meta = {
                "index": chunk_idx,
                "filename": f"chunk_{chunk_idx:03d}.wav",
                "character": "DIVIDER",
                "engine": "SILENCE",
                "voice": voice_label,
                "duration": get_wav_duration(chunk_filename)
            }
            record_chunk_done(meta, desc_info="Divider")
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

            log_print(f"    [Chunk {idx:03d}/{total}] [CPU-Kokoro] [{char.upper()}] (Voice: {v_label}) ({len(text)} chars)...")
            temp_raw = os.path.join(chapter_output_dir, f"temp_raw_{idx:03d}.wav")

            t0 = time.time()
            try:
                render_kokoro_chunk_local(text, cfg, temp_raw)
                _finalize_rendered_chunk(temp_raw, chunk_file, task, normalize_volume=normalize_volume)

                dur = get_wav_duration(chunk_file)
                elapsed = time.time() - t0
                log_print(f"    [OK] [Chunk {idx:03d}/{total}] [CPU-Kokoro] Done in {elapsed:.2f}s (Audio: {dur:.2f}s)")

                meta = {
                    "index": idx,
                    "filename": f"chunk_{idx:03d}.wav",
                    "character": char,
                    "engine": e_label,
                    "voice": v_label,
                    "duration": dur
                }
                record_chunk_done(meta, desc_info=f"Kokoro: {char}")
            except Exception as e:
                log_print(f"    [X] Error rendering Kokoro chunk {idx} ({char}): {e}")
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

            log_print(f"    [Chunk {idx:03d}/{total}] [GPU-F5TTS]  [{char.upper()}] (Voice: {v_label}) ({len(text)} chars)...")
            temp_raw = os.path.join(chapter_output_dir, f"temp_raw_{idx:03d}.wav")

            t0 = time.time()
            try:
                render_f5tts_chunk_local(text, cfg, temp_raw, chapter_output_dir, idx)
                _finalize_rendered_chunk(temp_raw, chunk_file, task, normalize_volume=normalize_volume)

                dur = get_wav_duration(chunk_file)
                elapsed = time.time() - t0
                log_print(f"    [OK] [Chunk {idx:03d}/{total}] [GPU-F5TTS]  Done in {elapsed:.2f}s (Audio: {dur:.2f}s)")

                meta = {
                    "index": idx,
                    "filename": f"chunk_{idx:03d}.wav",
                    "character": char,
                    "engine": e_label,
                    "voice": v_label,
                    "duration": dur
                }
                record_chunk_done(meta, desc_info=f"F5: {char}")
            except Exception as e:
                log_print(f"    [X] Error rendering F5-TTS chunk {idx} ({char}): {e}")
                worker_errors.append((idx, char, e))
                if os.path.exists(temp_raw):
                    try: os.remove(temp_raw)
                    except Exception: pass
            finally:
                f5_queue.task_done()

    # 3. Start simultaneous threads
    pending_kokoro = kokoro_queue.qsize()
    pending_f5 = f5_queue.qsize()

    log_print(f"[*] Queued tasks: {pending_kokoro} Kokoro chunks (CPU), {pending_f5} F5-TTS chunks (GPU).")

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

    # Close progress bar cleanly
    if progress_bar:
        progress_bar.close()

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
        f'ffmpeg -y -f concat -safe 0 -i "{list_file}" -ar 44100 -codec:a libmp3lame'
        f' -b:a 192k "{final_mp3}"'
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
            used_chars_lower = {
                str(meta["character"]).strip().lower()
                for meta in chunk_metadata
                if meta["character"] != "DIVIDER"
            }
            tf.write("VOICE ROSTER USED:\n")
            for c_name, c_info in characters_map.items():
                if c_name.strip().lower() not in used_chars_lower:
                    continue
                disp = c_info.get("display_name", c_name.title())
                c_eng = c_info.get("engine", "kokoro").upper()
                if c_eng in ["F5TTS", "F5-TTS"]:
                    ref_f = os.path.basename(c_info.get("ref_audio", "sample.wav"))
                    emotions = c_info.get("emotions", {})
                    emo_str = f" (emotions: {', '.join(emotions.keys())})" if emotions else ""
                    tf.write(f"  • [{disp}] (F5-TTS - GPU): ref={ref_f}{emo_str}\n")
                else:
                    v_str = c_info.get("voice", "af_bella")
                    p_val = float(c_info.get("pitch", 1.0))
                    s_val = float(c_info.get("speed", 1.0))
                    extras = []
                    if p_val != 1.0: extras.append(f"pitch: {p_val}")
                    if s_val != 1.0: extras.append(f"speed: {s_val}")
                    emotions = c_info.get("emotions", {})
                    if emotions: extras.append(f"emotions: {', '.join(emotions.keys())}")
                    if extras:
                        v_str += f" ({', '.join(extras)})"
                    tf.write(f"  • [{disp}] (KOKORO - CPU): {v_str}\n")
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


def save_characters_volume_update(char_key, new_volume, variant_name=None, characters_file=CHARACTERS_FILE):
    """Updates volume setting for a character or specific emotion/alt entry in characters.json."""
    ensure_characters_file()
    with open(characters_file, "r", encoding="utf-8") as f:
        raw_chars = json.load(f)

    matched_key = next((k for k in raw_chars if k.lower() == char_key.lower()), None)
    if not matched_key or not isinstance(raw_chars[matched_key], dict):
        return False

    if not variant_name:
        raw_chars[matched_key]["volume"] = float(new_volume)
    else:
        updated = False
        for section in ("emotions", "alts", "alt_voices", "alt"):
            sec_map = raw_chars[matched_key].get(section)
            if isinstance(sec_map, dict):
                v_key = next((vk for vk in sec_map if vk.lower() == variant_name.lower()), None)
                if v_key is not None:
                    if isinstance(sec_map[v_key], dict):
                        sec_map[v_key]["volume"] = float(new_volume)
                    elif isinstance(sec_map[v_key], str):
                        eng = str(raw_chars[matched_key].get("engine", "kokoro")).lower()
                        field = "ref_audio" if eng in ("f5tts", "f5-tts") or "." in sec_map[v_key] else "voice"
                        sec_map[v_key] = {field: sec_map[v_key], "volume": float(new_volume)}
                    updated = True
                    break
        if not updated:
            return False

    with open(characters_file, "w", encoding="utf-8") as f:
        json.dump(raw_chars, f, indent=4)
    return True


def interactive_adjust_volumes(characters):
    """Lets the user adjust character or emotion/alt volume multipliers cleanly in characters.json."""
    char_items = list(characters.items())
    while True:
        safe_print("\n" + "=" * 70)
        safe_print("                   Voice Volume Adjustment Editor")
        safe_print("=" * 70)
        for idx, (key, cfg) in enumerate(char_items, 1):
            disp = cfg.get("display_name", key)
            vol = float(cfg.get("volume", 1.0))
            variants = get_character_variants(cfg)
            var_str = f" | variants: {', '.join(v[0] for v in variants)}" if variants else ""
            safe_print(f"  [{idx:2d}] {disp:<18} (volume={vol:.2f}x){var_str}")
        safe_print("  [ 0] Done / Continue")

        sel = input("\nSelect character number to adjust [0-" + str(len(char_items)) + "] (default 0): ").strip()
        if not sel or sel == "0":
            break
        try:
            c_num = int(sel)
            if not (1 <= c_num <= len(char_items)):
                safe_print("[-] Invalid selection.")
                continue
        except ValueError:
            safe_print("[-] Invalid number.")
            continue

        c_key, c_cfg = char_items[c_num - 1]
        disp = c_cfg.get("display_name", c_key)
        variants = get_character_variants(c_cfg)

        target_variant = None
        if variants:
            safe_print(f"\nWhich voice for [{disp}]?")
            safe_print(f"  [0] Baseline ({disp}) [cur volume={float(c_cfg.get('volume', 1.0)):.2f}x]")
            for v_i, (v_name, v_kind, v_entry) in enumerate(variants, 1):
                v_vol = float(v_entry.get("volume", c_cfg.get("volume", 1.0))) if isinstance(v_entry, dict) else float(c_cfg.get("volume", 1.0))
                safe_print(f"  [{v_i}] {v_name} [{v_kind}] (cur volume={v_vol:.2f}x)")
            v_sel = input(f"Select [0-{len(variants)}] (default 0): ").strip()
            if v_sel and v_sel != "0":
                try:
                    vi = int(v_sel)
                    if 1 <= vi <= len(variants):
                        target_variant = variants[vi - 1][0]
                except ValueError:
                    pass

        vol_in = input("Enter new volume multiplier (e.g. 1.0 normal, 1.25 louder, 0.8 quieter): ").strip()
        try:
            new_vol = float(vol_in)
            if new_vol <= 0:
                safe_print("[-] Volume must be positive.")
                continue
        except ValueError:
            safe_print("[-] Invalid number.")
            continue

        if save_characters_volume_update(c_key, new_vol, variant_name=target_variant):
            label = f"{disp}:{target_variant}" if target_variant else disp
            safe_print(f"[OK] Saved volume={new_vol:.2f}x for [{label}] in '{CHARACTERS_FILE}'.")
            characters = load_characters()
            char_items = list(characters.items())
        else:
            safe_print("[!] Could not update character entry.")

    return characters


def _prompt_pick_character_and_variant(characters_map):
    """Interactive picker for choosing a character and optional emotion/alt variant."""
    char_items = list(characters_map.items())
    safe_print("\nAvailable Characters:")
    for idx, (c_key, cfg) in enumerate(char_items, 1):
        disp = cfg.get("display_name", c_key)
        eng = cfg.get("engine", "kokoro").upper()
        variants = get_character_variants(cfg)
        v_note = f" (variants: {', '.join(v[0] for v in variants)})" if variants else ""
        safe_print(f"  [{idx:2d}] {disp:<16} [{eng}]{v_note}")

    sel = input(f"Select character [1-{len(char_items)}]: ").strip()
    try:
        ci = int(sel)
        if not (1 <= ci <= len(char_items)):
            return None, None
    except ValueError:
        return None, None

    c_key, cfg = char_items[ci - 1]
    disp = cfg.get("display_name", c_key)
    variants = get_character_variants(cfg)
    chosen_variant = ""
    if variants:
        safe_print(f"\nSelect voice variant for [{disp}]:")
        safe_print("  [0] Baseline / Default")
        for v_i, (v_name, v_kind, _) in enumerate(variants, 1):
            safe_print(f"  [{v_i}] {v_name} [{v_kind}]")
        v_sel = input(f"Select variant [0-{len(variants)}] (default 0): ").strip()
        if v_sel and v_sel != "0":
            try:
                vi = int(v_sel)
                if 1 <= vi <= len(variants):
                    chosen_variant = variants[vi - 1][0]
            except ValueError:
                pass

    return c_key, chosen_variant


def _prompt_pick_same_char_variant(char_key, characters_map, tag_mappings):
    """Interactive picker for switching to an emotion/alt voice of the same character."""
    canon = tag_mappings.get(char_key.lower(), char_key.lower()) if tag_mappings else char_key.lower()
    cfg = characters_map.get(canon, characters_map.get("narrator", {}))
    disp = cfg.get("display_name", canon)
    variants = get_character_variants(cfg)
    if not variants:
        safe_print(f"  [!] No emotion or alt voices configured for [{disp}]. Using Baseline.")
        return ""

    safe_print(f"\nAvailable emotion / alt voices for [{disp}]:")
    safe_print("  [0] Baseline / Default")
    for v_i, (v_name, v_kind, v_entry) in enumerate(variants, 1):
        clip = ""
        if isinstance(v_entry, dict):
            clip = v_entry.get("ref_audio") or v_entry.get("voice") or ""
        elif isinstance(v_entry, str):
            clip = v_entry
        clip_lbl = f" ({os.path.basename(clip)})" if clip else ""
        safe_print(f"  [{v_i}] {v_name} [{v_kind}]{clip_lbl}")

    v_sel = input(f"Select voice [0-{len(variants)}] (default 0): ").strip()
    if not v_sel or v_sel == "0":
        return ""
    try:
        vi = int(v_sel)
        if 1 <= vi <= len(variants):
            return variants[vi - 1][0]
    except ValueError:
        pass
    return ""


def interactive_rerecord_chunks(scan_results, master_output_dir, characters_map, tag_mappings=None, enable_emotions=True):
    """
    Interactive chunk-level re-recording & voice switcher:
    Allows selecting specific chunks inside a section, changing their voice to another
    character or to one of the same character's emotion/alt voices, and adjusting chunk volume.
    """
    while True:
        if len(scan_results) == 1:
            sec_num = 1
        else:
            safe_print("\nSections:")
            for item in scan_results:
                safe_print(f"  [{item['idx']:02d}] {item['base_name']:<40} -> [{item['status']}]")
            sec_in = input(f"\nEnter section number to edit chunks [1-{len(scan_results)}] (or Enter to finish): ").strip()
            if not sec_in:
                break
            try:
                sec_num = int(sec_in)
                if not (1 <= sec_num <= len(scan_results)):
                    safe_print("[-] Invalid section number.")
                    continue
            except ValueError:
                safe_print("[-] Invalid number.")
                continue

        item = scan_results[sec_num - 1]
        txt_file = item["txt_file"]
        base_name = item["base_name"]
        if not os.path.exists(txt_file):
            safe_print(f"[!] Could not find text file '{txt_file}'.")
            break

        with open(txt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        blocks = split_text_into_tagged_blocks(
            raw_text,
            default_character="narrator",
            characters_map=characters_map,
            tag_mappings=tag_mappings,
            enable_emotions=enable_emotions,
        )
        chunk_dir = os.path.join(master_output_dir, "chunks_" + base_name)
        overrides = load_chunk_overrides(chunk_dir)

        safe_print("\n" + "=" * 72)
        safe_print(f" Chunks in '{base_name}' ({len(blocks)} chunks):")
        safe_print("=" * 72)
        effective_info = {}
        for idx_b, b in enumerate(blocks, 1):
            ov = overrides.get(str(idx_b), {})
            eff_char = ov.get("character", b["character"])
            eff_var = ov.get("variant", b.get("emotion", ""))
            eff_vol = ov.get("volume", None)
            effective_info[idx_b] = (b["type"], eff_char, eff_var)

            if b["type"] == "DIVIDER":
                safe_print(f"  [{idx_b:03d}] [DIVIDER]                  | (2s Silence Pause)")
            else:
                tag_lbl = f"{eff_char}:{eff_var}" if eff_var else eff_char
                if ov:
                    tag_lbl += "*"
                if eff_vol is not None:
                    tag_lbl += f" vol={eff_vol:.2f}x"
                preview = b["text"].replace("\n", " ").strip()
                if len(preview) > 52:
                    preview = preview[:52] + "..."
                safe_print(f"  [{idx_b:03d}] [{tag_lbl:<24}] | {preview}")
        safe_print("=" * 72)

        ch_in = input("Enter chunk number(s) to re-record (e.g. '2', '2, 5-7', 'all', or Enter to cancel): ").strip()
        if not ch_in:
            if len(scan_results) == 1:
                break
            continue

        if ch_in.lower() == "all":
            sel_indices = list(range(1, len(blocks) + 1))
        else:
            sel_indices = parse_selection_indices(ch_in, len(blocks))

        voice_indices = [ci for ci in sel_indices if effective_info[ci][0] != "DIVIDER"]
        if not voice_indices:
            safe_print("[-] No voice chunks selected.")
            if len(scan_results) == 1:
                break
            continue

        safe_print(f"\nSelected chunk(s): {', '.join(f'{c:03d}' for c in voice_indices)}")
        safe_print("Voice / Re-record Action:")
        safe_print("  [1] Re-record with current voice (no voice change) [Default]")
        safe_print("  [2] Change to an emotion / alt voice of the SAME character")
        safe_print("  [3] Switch chunk(s) to ANOTHER character (and optionally their emotion / alt voice)")
        safe_print("  [4] Adjust volume multiplier only for these chunk(s)")
        safe_print("  [5] Reset chunk(s) to original script tag & default volume")
        act = input("Select action [1, 2, 3, 4, or 5] (default 1): ").strip() or "1"

        def _prompt_optional_volume(required=False):
            prompt = (
                "Enter volume multiplier (e.g. 1.25 louder, 0.8 quieter): "
                if required
                else "Optional volume multiplier for these chunk(s) (e.g. 1.2, 0.8, Enter to keep): "
            )
            v_str = input(prompt).strip()
            if not v_str:
                return None
            try:
                parsed_v = float(v_str)
                if parsed_v > 0:
                    return parsed_v
                safe_print("[!] Volume must be positive; ignored.")
            except ValueError:
                safe_print("[!] Invalid volume ignored.")
            return None

        if act == "1":
            apply_chunk_override_action(txt_file, master_output_dir, voice_indices)
        elif act == "2":
            by_char = {}
            for ci in voice_indices:
                by_char.setdefault(effective_info[ci][1], []).append(ci)
            chosen_pairs = []
            for c_name, c_list in by_char.items():
                chosen_var = _prompt_pick_same_char_variant(c_name, characters_map, tag_mappings)
                chosen_pairs.append((c_list, chosen_var))
            new_vol = _prompt_optional_volume(required=False)
            for c_list, chosen_var in chosen_pairs:
                apply_chunk_override_action(
                    txt_file,
                    master_output_dir,
                    c_list,
                    new_variant=chosen_var,
                    set_variant=True,
                    new_volume=new_vol,
                )
        elif act == "3":
            target_char, target_var = _prompt_pick_character_and_variant(characters_map)
            if target_char is None:
                safe_print("[-] Cancelled character switch.")
                if len(scan_results) == 1:
                    break
                continue
            new_vol = _prompt_optional_volume(required=False)
            apply_chunk_override_action(
                txt_file,
                master_output_dir,
                voice_indices,
                new_char=target_char,
                new_variant=target_var,
                set_variant=True,
                new_volume=new_vol,
            )
        elif act == "4":
            new_vol = _prompt_optional_volume(required=True)
            if new_vol is None:
                safe_print("[-] No valid volume entered; cancelled.")
                if len(scan_results) == 1:
                    break
                continue
            apply_chunk_override_action(txt_file, master_output_dir, voice_indices, new_volume=new_vol)
        elif act == "5":
            apply_chunk_override_action(txt_file, master_output_dir, voice_indices, clear_override=True)
        else:
            safe_print("[-] Invalid action; cancelled.")
            if len(scan_results) == 1:
                break
            continue

        safe_print(f"[OK] Scheduled {len(voice_indices)} chunk(s) in '{base_name}' for re-recording.")
        again = input("\nSelect more chunks to re-record/edit? [y/N] (default n): ").strip().lower()
        if again != "y":
            break


def convert_audiobook(manifest_file=MANIFEST_FILE, enable_emotions=None, interactive=True, normalize_volume=True, force_rerecord_menu=False):
    """
    Renders an audiobook from a chapter manifest using simultaneous CPU Kokoro and GPU F5-TTS.
    Returns True on success, False on error.
    """
    if enable_emotions is None:
        if interactive:
            safe_print("\nDialogue Tone & Emotion Delivery:")
            safe_print("  [1] Enabled  - Apply tone modulation (whisper, shout, angry, etc.) [Default]")
            safe_print("  [2] Disabled - Pure baseline voices only (zero speed/pitch distortion)")
            emo_choice = input("Select option [1 or 2] (default 1): ").strip()
            enable_emotions = (emo_choice != "2")
        else:
            enable_emotions = True

    safe_print(f"[*] Tone & Emotion Mode: {'ENABLED (Acoustic Modulation)' if enable_emotions else 'DISABLED (Pure Baseline)'}")

    if not os.path.exists(manifest_file):
        safe_print(f"Error: Could not find manifest file '{manifest_file}'.")
        safe_print(f"Tip: Run 'new_text_parser.py' first to format chapters into '{manifest_file}'.")
        return False

    with open(manifest_file, "r", encoding="utf-8") as mf:
        chapter_files = [line.strip() for line in mf if line.strip() and not line.startswith("#")]

    if not chapter_files:
        safe_print(f"Error: Manifest file '{manifest_file}' is empty.")
        return False

    # Load character profiles and tag mappings
    characters = load_characters()
    tag_mappings = load_tag_mappings()
    safe_print(f"\nLoaded {len(characters)} character profiles from '{CHARACTERS_FILE}':")
    for name, cfg in characters.items():
        disp_name = cfg.get("display_name", name)
        engine = cfg.get("engine", "kokoro").upper()
        vol = float(cfg.get("volume", 1.0))
        vol_str = f" [vol: {vol:.2f}x]" if abs(vol - 1.0) > 1e-3 else ""
        variants = get_character_variants(cfg)
        var_str = f" (variants: {', '.join(v[0] for v in variants)})" if variants else ""
        if engine == "KOKORO":
            safe_print(f"  - [{disp_name}]: KOKORO (CPU) -> {cfg.get('voice', '')}{vol_str}{var_str}")
        else:
            safe_print(f"  - [{disp_name}]: F5-TTS (GPU) -> {cfg.get('ref_audio', '')}{vol_str}{var_str}")

    if interactive:
        safe_print("\nVoice Volume Options:")
        safe_print("  [1] Auto-balanced (recommended, uses per-voice 'volume' multipliers) [Default]")
        safe_print("  [2] Unbalanced / Raw volumes (only applies explicit 'volume' multipliers)")
        safe_print("  [3] Adjust voice volume(s) now in characters.json")
        vol_choice = input("Select option [1, 2, or 3] (default 1): ").strip()
        if vol_choice == "3":
            characters = interactive_adjust_volumes(characters)
            safe_print("  [1] Auto-balanced [Default]  |  [2] Unbalanced / Raw")
            bal_ans = input("Use auto-balancing on render? [1 or 2] (default 1): ").strip()
            normalize_volume = (bal_ans != "2")
        elif vol_choice == "2":
            normalize_volume = False
        safe_print(f"[*] Volume Mode: {'Auto-Balanced' if normalize_volume else 'Raw (Per-Voice Multipliers Only)'}")

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

    # Generate master production voice roster documenting all active characters
    write_production_voice_roster(master_output_dir, book_title_base, characters)

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

    if (has_existing or force_rerecord_menu) and interactive:
        safe_print("\n" + "=" * 70)
        safe_print("                 Existing Audio Sections Detected")
        safe_print("=" * 70)
        for item in scan_results:
            safe_print(f"  [{item['idx']:02d}] {item['base_name']:<40} -> [{item['status']}]")
        safe_print("=" * 70)
        safe_print("Options:")
        safe_print("  [1] Resume / Keep existing (skip completed, finish partials) [Default]")
        safe_print("  [2] Re-record EVERYTHING from scratch")
        safe_print("  [3] Choose specific sections to re-record completely")
        safe_print("  [4] Choose specific CHUNKS to re-record / change chunk voice")
        if force_rerecord_menu:
            choice = "4"
        else:
            choice = input("\nSelect how to proceed [1, 2, 3, or 4] (default 1): ").strip()

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
        elif choice == "4":
            interactive_rerecord_chunks(
                scan_results,
                master_output_dir,
                characters,
                tag_mappings=tag_mappings,
                enable_emotions=enable_emotions,
            )
    elif has_existing and not interactive:
        safe_print("\n[*] Existing audio sections detected. Resuming automatically (skipping completed).")

    # ==================== MAIN PROCESSING LOOP ====================
    for idx, txt_file in enumerate(chapter_files, 1):
        safe_print(f"\n[{idx}/{len(chapter_files)}] Processing chapter: {txt_file}")
        success = process_hybrid_chapter_simultaneous(txt_file, characters, master_output_dir, tag_mappings=tag_mappings, enable_emotions=enable_emotions, normalize_volume=normalize_volume)
        if not success:
            safe_print(f"[!] Pausing due to error in {txt_file}.")
            return False

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

    dest_folder = master_output_dir
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
    return True


def main():
    safe_print("==================================================================")
    safe_print("  Simultaneous Hybrid Audiobook Generator (CPU Kokoro + GPU F5)  ")
    safe_print("==================================================================")

    import argparse
    parser = argparse.ArgumentParser(description="Simultaneous Hybrid Audiobook Generator (CPU Kokoro + GPU F5)")
    parser.add_argument("--manifest", default=MANIFEST_FILE, help="Path to manifest file listing chapters (default: chapters.txt).")
    parser.add_argument("--emotions", dest="enable_emotions", action="store_true", default=None, help="Enable dialogue emotion/alt voice routing.")
    parser.add_argument("--no-emotions", dest="enable_emotions", action="store_false", help="Disable acoustic emotion routing (alt voices still apply).")
    parser.add_argument("--no-volume-balance", dest="normalize_volume", action="store_false", default=True, help="Disable automatic active-speech loudness balancing.")
    parser.add_argument("--rerecord", action="store_true", default=False, help="Jump directly to chunk-level re-record / voice switching when existing audio is found.")
    args, _ = parser.parse_known_args()

    convert_audiobook(
        manifest_file=args.manifest,
        enable_emotions=args.enable_emotions,
        interactive=True,
        normalize_volume=args.normalize_volume,
        force_rerecord_menu=args.rerecord,
    )


if __name__ == "__main__":
    main()
