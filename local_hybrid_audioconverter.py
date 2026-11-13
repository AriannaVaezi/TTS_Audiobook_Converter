import os
import re
import sys
import time
import json
import queue
import random
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
KOKORO_VOICE_CATALOG = {
    "american": {
        "female": [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        ],
        "male": [
            "am_adam", "am_echo", "am_eric", "am_fenrir",
            "am_liam", "am_michael", "am_onyx", "am_puck",
        ],
    },
    "british": {
        "female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
        "male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    },
}
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
        close_match = re.match(r"^\[\s*/\s*([a-zA-Z0-9_\'\-]+)(?::[a-zA-Z0-9_\'\-]+)?\s*\]$", raw_line)
        if close_match:
            in_explicit_block = False
            current_char = resolve_char(default_character)
            current_emotion = ""
            continue

        # 3. Check for opening block tag: [character] or [character:emotion] alone on a line
        open_block_match = re.match(r"^\[\s*([a-zA-Z0-9_\'\-]+)(?::([a-zA-Z0-9_\'\-]+))?\s*\]$", raw_line)
        if open_block_match:
            in_explicit_block = True
            current_char = resolve_char(open_block_match.group(1))
            current_emotion = resolve_variant(current_char, open_block_match.group(2))
            continue

        # 4. Check for embedded inline tags: e.g. Narration [Hero:whisper] "Dialogue" [/Hero] Narration
        inline_tag_pattern = re.compile(
            r"\[\s*([a-zA-Z0-9_\'\-]+)(?::([a-zA-Z0-9_\'\-]+))?\s*\](.*?)\[\s*/\s*\1(?::[a-zA-Z0-9_\'\-]+)?\s*\]",
            re.IGNORECASE,
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
        prefix_match = re.match(r"^\[\s*([a-zA-Z0-9_\'\-]+)(?::([a-zA-Z0-9_\'\-]+))?\s*\]\s*(.*)$", raw_line)
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


def _normalize_narrator_gender(gender):
    """Normalizes gender option aliases into canonical keys."""
    raw = str(gender or "any").strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "female": "female",
        "f": "female",
        "male": "male",
        "m": "male",
        "mostly_female": "mostly_female",
        "mf": "mostly_female",
        "mostly_male": "mostly_male",
        "mm": "mostly_male",
        "any": "any",
        "none": "any",
        "no_preference": "any",
        "mixed": "any",
        "all": "any",
    }
    if raw not in mapping:
        raise ValueError(
            f"Invalid narrator gender '{gender}'. Choose from: female, male, mostly-female, mostly-male, any."
        )
    return mapping[raw]


def _normalize_narrator_accent(accent):
    """Normalizes regional/accent option aliases into canonical keys."""
    raw = str(accent or "any").strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "american": "american",
        "us": "american",
        "a": "american",
        "british": "british",
        "uk": "british",
        "gb": "british",
        "b": "british",
        "any": "any",
        "none": "any",
        "no_preference": "any",
        "all": "any",
    }
    if raw not in mapping:
        raise ValueError(
            f"Invalid narrator accent '{accent}'. Choose from: american, british, any."
        )
    return mapping[raw]


def parse_voice_count_range(spec, max_available=None):
    """
    Parses a voice mix count or inclusive range such as 3, '3', '2-4', '2..5', or (2, 4).
    Returns clamped (low, high) integer bounds.
    """
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        lo, hi = int(spec[0]), int(spec[1])
    elif isinstance(spec, int):
        lo, hi = spec, spec
    else:
        text = str(spec or "2-4").strip()
        m_single = re.match(r"^(\d+)$", text)
        m_range = re.match(r"^(\d+)\s*(?:-|\.\.|to)\s*(\d+)$", text, re.IGNORECASE)
        if m_single:
            lo = hi = int(m_single.group(1))
        elif m_range:
            lo, hi = int(m_range.group(1)), int(m_range.group(2))
        else:
            raise ValueError(f"Invalid voice count or range '{spec}'. Expected e.g. '3' or '2-4'.")

    if lo < 1 or hi < 1:
        raise ValueError("Voice count must be at least 1.")
    if lo > hi:
        lo, hi = hi, lo
    if max_available is not None and max_available >= 1:
        hi = min(hi, max_available)
        lo = min(lo, hi)
    return lo, hi


def get_kokoro_voice_pool(accent="any"):
    """Returns (female_voices, male_voices) filtered by accent ('american', 'british', or 'any')."""
    norm_accent = _normalize_narrator_accent(accent)
    regions = ["american", "british"] if norm_accent == "any" else [norm_accent]
    females = []
    males = []
    for reg in regions:
        females.extend(KOKORO_VOICE_CATALOG[reg]["female"])
        males.extend(KOKORO_VOICE_CATALOG[reg]["male"])
    return females, males


def _distribute_rounded_weights(count, total_weight, rng):
    """Generates `count` positive 2-decimal weights summing cleanly to `total_weight`."""
    total_cents = max(count, int(round(total_weight * 100)))
    raw = [rng.uniform(0.35, 1.0) for _ in range(count)]
    raw_sum = sum(raw)
    cents = [max(1, int(round((val / raw_sum) * total_cents))) for val in raw]
    diff = total_cents - sum(cents)
    # Adjust the largest component so exact sum equals total_cents (keep >= 1)
    idx_order = sorted(range(count), key=lambda i: cents[i], reverse=True)
    step = 1 if diff > 0 else -1
    rem = abs(diff)
    cursor = 0
    while rem > 0 and idx_order:
        target_idx = idx_order[cursor % len(idx_order)]
        if step > 0 or cents[target_idx] > 1:
            cents[target_idx] += step
            rem -= 1
        cursor += 1
        if cursor > rem + count * 10:
            break
    return [c / 100.0 for c in cents]


def generate_random_kokoro_voice(gender="any", accent="any", voice_count="2-4", rng=None):
    """
    Generates a randomized Kokoro voice name or weighted blend string compatible with
    parse_and_load_kokoro_voice().

    Args:
        gender: 'female', 'male', 'mostly_female' (0.2-0.4 male), 'mostly_male' (0.2-0.4 female), or 'any'.
        accent: 'american', 'british', or 'any'.
        voice_count: exact integer or inclusive range ('3', '2-4', (2, 5)).
        rng: optional random.Random instance for deterministic testing.
    """
    r = rng if rng is not None else random
    norm_gender = _normalize_narrator_gender(gender)
    norm_accent = _normalize_narrator_accent(accent)
    female_pool, male_pool = get_kokoro_voice_pool(norm_accent)

    if norm_gender == "female":
        max_avail = len(female_pool)
        lo, hi = parse_voice_count_range(voice_count, max_available=max_avail)
        k = r.randint(lo, hi)
        chosen = r.sample(female_pool, k)
        if k == 1:
            return chosen[0]
        weights = _distribute_rounded_weights(k, 1.0, r)
        pairs = sorted(zip(chosen, weights), key=lambda x: x[1], reverse=True)
    elif norm_gender == "male":
        max_avail = len(male_pool)
        lo, hi = parse_voice_count_range(voice_count, max_available=max_avail)
        k = r.randint(lo, hi)
        chosen = r.sample(male_pool, k)
        if k == 1:
            return chosen[0]
        weights = _distribute_rounded_weights(k, 1.0, r)
        pairs = sorted(zip(chosen, weights), key=lambda x: x[1], reverse=True)
    elif norm_gender in ("mostly_female", "mostly_male"):
        max_avail = len(female_pool) + len(male_pool)
        lo, hi = parse_voice_count_range(voice_count, max_available=max_avail)
        eff_lo = min(max_avail, max(2, lo))
        eff_hi = max(eff_lo, hi)
        k = r.randint(eff_lo, eff_hi)

        maj_pool, min_pool = (female_pool, male_pool) if norm_gender == "mostly_female" else (male_pool, female_pool)
        max_min_count = min(len(min_pool), k - 1, max(1, k // 2))
        min_min_count = max(1, k - len(maj_pool))
        n_min = r.randint(min_min_count, max(min_min_count, max_min_count))
        n_maj = k - n_min

        minority_total = r.randint(20, 40) / 100.0  # Strictly 0.20 - 0.40 minority gender share
        majority_total = round(1.0 - minority_total, 2)

        maj_chosen = r.sample(maj_pool, n_maj)
        min_chosen = r.sample(min_pool, n_min)
        maj_weights = _distribute_rounded_weights(n_maj, majority_total, r)
        min_weights = _distribute_rounded_weights(n_min, minority_total, r)
        pairs = sorted(zip(maj_chosen, maj_weights), key=lambda x: x[1], reverse=True) + sorted(
            zip(min_chosen, min_weights), key=lambda x: x[1], reverse=True
        )
    else:
        combined = female_pool + male_pool
        lo, hi = parse_voice_count_range(voice_count, max_available=len(combined))
        k = r.randint(lo, hi)
        chosen = r.sample(combined, k)
        if k == 1:
            return chosen[0]
        weights = _distribute_rounded_weights(k, 1.0, r)
        pairs = sorted(zip(chosen, weights), key=lambda x: x[1], reverse=True)

    return " + ".join(f"{name}*{w:.2f}" for name, w in pairs)


def summarize_kokoro_blend(voice_spec):
    """Returns a concise summary of female/male and American/British share in a Kokoro voice string."""
    raw_parts = [p.strip() for p in re.split(r"[\+,]", str(voice_spec or "")) if p.strip()]
    items = []
    for part in raw_parts:
        m = re.match(r"^([a-zA-Z0-9_]+)\s*\*\s*([0-9.]+)$", part)
        if m:
            items.append((m.group(1), float(m.group(2))))
        else:
            items.append((part, 1.0))
    total = sum(w for _, w in items) or 1.0
    female_w = sum(w for name, w in items if len(name) >= 2 and name[1] == "f") / total
    male_w = sum(w for name, w in items if len(name) >= 2 and name[1] == "m") / total
    us_w = sum(w for name, w in items if name.startswith("a")) / total
    uk_w = sum(w for name, w in items if name.startswith("b")) / total
    return (
        f"{len(items)} voice(s) | "
        f"Female {female_w * 100:.0f}% / Male {male_w * 100:.0f}% | "
        f"American {us_w * 100:.0f}% / British {uk_w * 100:.0f}%"
    )


def save_narrator_voice_update(voice_spec, characters_file=CHARACTERS_FILE):
    """Persists a Kokoro narrator voice string into characters.json while keeping narrator settings."""
    try:
        with open(characters_file, "r", encoding="utf-8") as f:
            raw_chars = json.load(f)
    except Exception:
        raw_chars = {}

    matched_key = next((k for k in raw_chars.keys() if k.lower() == "narrator"), "narrator")
    existing = raw_chars.get(matched_key)
    if isinstance(existing, dict):
        updated_entry = dict(existing)
        updated_entry.pop("alias", None)
        updated_entry.pop("character", None)
    else:
        updated_entry = {}

    updated_entry["engine"] = "kokoro"
    updated_entry["voice"] = voice_spec
    updated_entry.setdefault("pitch", 1.0)
    updated_entry.setdefault("speed", 1.0)
    updated_entry.setdefault("volume", 1.0)
    raw_chars[matched_key] = updated_entry

    with open(characters_file, "w", encoding="utf-8") as f:
        json.dump(raw_chars, f, indent=4)
    return True


def apply_random_narrator_to_characters(
    characters_map,
    gender="any",
    accent="any",
    voice_count="2-4",
    persist=False,
    characters_file=CHARACTERS_FILE,
    rng=None,
):
    """Generates a random Kokoro voice blend and assigns it to `narrator` in characters_map."""
    voice_spec = generate_random_kokoro_voice(
        gender=gender, accent=accent, voice_count=voice_count, rng=rng
    )
    nar = dict(characters_map.get("narrator", {}))
    nar["display_name"] = nar.get("display_name", "narrator")
    nar["engine"] = "kokoro"
    nar["voice"] = voice_spec
    nar.setdefault("pitch", 1.0)
    nar.setdefault("speed", 1.0)
    nar.setdefault("volume", 1.0)
    characters_map["narrator"] = nar

    if persist:
        save_narrator_voice_update(voice_spec, characters_file=characters_file)
    return characters_map, voice_spec


def interactive_randomize_narrator(characters_map):
    """Interactive prompt letting the user randomize the narrator voice with Kokoro options."""
    cur_nar = characters_map.get("narrator", {})
    cur_voice = cur_nar.get("voice") or cur_nar.get("ref_audio") or "af_bella"
    safe_print(f"\nNarrator Voice Options (Currently: {cur_voice}):")
    safe_print("  [1] Keep current narrator from characters.json [Default]")
    safe_print("  [2] Randomize narrator with random Kokoro TTS voice blend")
    mode_choice = input("Select option [1 or 2] (default 1): ").strip()
    if mode_choice != "2":
        return characters_map

    gender_choices = {
        "1": "female",
        "2": "male",
        "3": "mostly_female",
        "4": "mostly_male",
        "5": "any",
        "": "any",
    }
    accent_choices = {
        "1": "american",
        "2": "british",
        "3": "any",
        "": "any",
    }

    while True:
        safe_print("\nRandom Narrator Gender Preference:")
        safe_print("  [1] Female only")
        safe_print("  [2] Male only")
        safe_print("  [3] Mostly female (0.2 - 0.4 male)")
        safe_print("  [4] Mostly male (0.2 - 0.4 female)")
        safe_print("  [5] No gender preference (mix any genders) [Default]")
        g_in = input("Select gender [1-5] (default 5): ").strip()
        gender = gender_choices.get(g_in, "any")

        safe_print("\nRandom Narrator Accent / Region:")
        safe_print("  [1] American")
        safe_print("  [2] British")
        safe_print("  [3] No regional preference [Default]")
        a_in = input("Select accent [1-3] (default 3): ").strip()
        accent = accent_choices.get(a_in, "any")

        while True:
            cnt_in = input("\nNumber of voices to mix (exact like '3' or range like '2-4', default 2-4): ").strip()
            count_spec = cnt_in or "2-4"
            try:
                parse_voice_count_range(count_spec)
                break
            except ValueError as e:
                safe_print(f"[-] {e}")

        if gender in ("mostly_female", "mostly_male") and parse_voice_count_range(count_spec)[1] < 2:
            safe_print(f"[*] '{gender.replace('_', ' ')}' requires blending at least 2 voices; using 2 voices.")

        while True:
            rolled = generate_random_kokoro_voice(gender=gender, accent=accent, voice_count=count_spec)
            safe_print(f"\n[*] Rolled Kokoro Narrator: {rolled}")
            safe_print(f"    Mix breakdown        : {summarize_kokoro_blend(rolled)}")
            confirm = input("Action: [Enter] Use this voice | [r] Re-roll | [c] Change options / cancel: ").strip().lower()
            if confirm == "r":
                continue
            if confirm == "c":
                break
            save_ans = input(f"Save this narrator voice to '{CHARACTERS_FILE}'? [Y/n] (default Y): ").strip().lower()
            persist = (save_ans != "n")
            nar = dict(characters_map.get("narrator", {}))
            nar["display_name"] = nar.get("display_name", "narrator")
            nar["engine"] = "kokoro"
            nar["voice"] = rolled
            nar.setdefault("pitch", 1.0)
            nar.setdefault("speed", 1.0)
            nar.setdefault("volume", 1.0)
            characters_map["narrator"] = nar
            if persist:
                save_narrator_voice_update(rolled)
                safe_print(f"[OK] Saved randomized narrator voice to '{CHARACTERS_FILE}'.")
            else:
                safe_print("[*] Using randomized narrator voice for this run only.")
            return characters_map

        again = input("Cancel randomizer and keep original narrator? [y/N] (default N): ").strip().lower()
        if again == "y":
            return characters_map


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


def apply_cli_volume_updates(characters_map, volume_specs, characters_file=CHARACTERS_FILE):
    """
    Applies non-interactive volume multiplier updates from specs like:
      'Alice=1.2', 'Bob:shout=0.85', 'narrator=1.0'
    Returns updated characters_map.
    """
    if not volume_specs:
        return characters_map

    for item in volume_specs:
        for piece in re.split(r"[;,]", str(item)):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            target, val_str = [p.strip() for p in piece.split("=", 1)]
            try:
                new_vol = float(val_str)
                if new_vol <= 0:
                    safe_print(f"[-] Volume must be positive (ignored '{piece}').")
                    continue
            except ValueError:
                safe_print(f"[-] Invalid volume number in '{piece}'.")
                continue

            if ":" in target:
                char_key, var_name = [p.strip() for p in target.split(":", 1)]
            else:
                char_key, var_name = target, None

            if save_characters_volume_update(char_key, new_vol, variant_name=var_name, characters_file=characters_file):
                lbl = f"{char_key}:{var_name}" if var_name else char_key
                safe_print(f"[OK] Updated volume={new_vol:.2f}x for [{lbl}] in '{characters_file}'.")
            else:
                safe_print(f"[!] Could not find character/variant '{target}' in '{characters_file}'.")

    return load_characters()


def _resolve_target_section_indices(section_spec, scan_results):
    """Resolves a section selector ('1', '1-3', or substring) into 0-based indices of `scan_results`."""
    if not section_spec:
        return list(range(len(scan_results)))
    raw = str(section_spec).strip()
    if raw.lower() == "all":
        return list(range(len(scan_results)))
    nums = parse_selection_indices(raw, len(scan_results))
    if nums:
        return [n - 1 for n in nums]
    raw_low = raw.lower()
    return [i for i, item in enumerate(scan_results) if raw_low in item["base_name"].lower()]


def list_chapter_chunks_report(
    scan_results,
    master_output_dir,
    characters_map,
    tag_mappings=None,
    enable_emotions=True,
    section_filter=None,
):
    """Prints numbered chunks and current voice/override assignments for selected section(s)."""
    target_indices = _resolve_target_section_indices(section_filter, scan_results)
    for si in target_indices:
        item = scan_results[si]
        txt_file = item["txt_file"]
        base_name = item["base_name"]
        if not os.path.exists(txt_file):
            continue
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
        safe_print(f" [{item['idx']:02d}] Chunks in '{base_name}' ({len(blocks)} chunks) [{item['status']}]:")
        safe_print("=" * 72)
        for idx_b, b in enumerate(blocks, 1):
            ov = overrides.get(str(idx_b), {})
            eff_char = ov.get("character", b["character"])
            eff_var = ov.get("variant", b.get("emotion", ""))
            eff_vol = ov.get("volume", None)
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


def apply_cli_chunk_operations(
    scan_results,
    master_output_dir,
    characters_map,
    tag_mappings=None,
    enable_emotions=True,
    section_filter=None,
    rerecord_chunks=None,
    set_chunk_voices=None,
    set_chunk_volumes=None,
    reset_chunks=None,
):
    """
    Applies non-interactive chunk re-recording, voice/variant switching, volume overrides,
    and override resets across selected sections.
    Spec formats:
      - rerecord_chunks : '[<sec>:]<chunk_indices>' (e.g. '2,5-7' or '1:all')
      - set_chunk_voices: '[<sec>:]<chunk_indices>=<char>[:<variant>][@<vol>]' or '2,3=:whisper'
      - set_chunk_volumes: '[<sec>:]<chunk_indices>=<vol>'
      - reset_chunks    : '[<sec>:]<chunk_indices>'
    Returns total number of chunks scheduled/updated.
    """
    def _split_semis(items):
        out = []
        for item in (items or []):
            for piece in str(item).split(";"):
                piece = piece.strip()
                if piece:
                    out.append(piece)
        return out

    def _resolve_chunks_for_section(si, idx_spec):
        item = scan_results[si]
        txt_file = item["txt_file"]
        if not os.path.exists(txt_file):
            return []
        with open(txt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        blocks = split_text_into_tagged_blocks(
            raw_text,
            default_character="narrator",
            characters_map=characters_map,
            tag_mappings=tag_mappings,
            enable_emotions=enable_emotions,
        )
        if idx_spec.strip().lower() == "all":
            sel = list(range(1, len(blocks) + 1))
        else:
            sel = parse_selection_indices(idx_spec, len(blocks))
        return [ci for ci in sel if blocks[ci - 1]["type"] != "DIVIDER"]

    def _split_section_and_chunks(lhs):
        lhs = lhs.strip()
        if ":" in lhs:
            sec_part, chunk_part = lhs.split(":", 1)
            return _resolve_target_section_indices(sec_part, scan_results), chunk_part.strip()
        return _resolve_target_section_indices(section_filter, scan_results), lhs

    total_updated = 0

    for spec in _split_semis(rerecord_chunks):
        sec_indices, chunk_spec = _split_section_and_chunks(spec)
        for si in sec_indices:
            v_indices = _resolve_chunks_for_section(si, chunk_spec)
            if v_indices:
                apply_chunk_override_action(scan_results[si]["txt_file"], master_output_dir, v_indices)
                total_updated += len(v_indices)
                safe_print(f"[OK] Scheduled {len(v_indices)} chunk(s) in '{scan_results[si]['base_name']}' for re-recording.")

    for spec in _split_semis(set_chunk_voices):
        if "=" not in spec:
            safe_print(f"[!] Ignoring invalid --set-chunk-voice spec '{spec}'.")
            continue
        lhs, rhs = spec.split("=", 1)
        sec_indices, chunk_spec = _split_section_and_chunks(lhs)
        rhs = rhs.strip()
        new_vol = None
        if "@" in rhs:
            rhs, vol_part = rhs.rsplit("@", 1)
            try:
                parsed_v = float(vol_part.strip())
                if parsed_v > 0:
                    new_vol = parsed_v
            except ValueError:
                pass
        if ":" in rhs:
            c_part, v_part = rhs.split(":", 1)
            new_char = c_part.strip() or None
            v_clean = v_part.strip().lower()
            new_var = "" if v_clean in ("", "none", "clear", "baseline", "-") else v_clean
            set_var = True
        else:
            new_char = rhs.strip() or None
            new_var = None
            set_var = False

        if new_char:
            canon = tag_mappings.get(new_char.lower(), new_char.lower()) if tag_mappings else new_char.lower()
            new_char = canon

        for si in sec_indices:
            v_indices = _resolve_chunks_for_section(si, chunk_spec)
            if v_indices:
                apply_chunk_override_action(
                    scan_results[si]["txt_file"],
                    master_output_dir,
                    v_indices,
                    new_char=new_char,
                    new_variant=new_var,
                    set_variant=set_var,
                    new_volume=new_vol,
                )
                total_updated += len(v_indices)
                safe_print(f"[OK] Updated voice override for {len(v_indices)} chunk(s) in '{scan_results[si]['base_name']}'.")

    for spec in _split_semis(set_chunk_volumes):
        if "=" not in spec:
            safe_print(f"[!] Ignoring invalid --set-chunk-volume spec '{spec}'.")
            continue
        lhs, rhs = spec.split("=", 1)
        sec_indices, chunk_spec = _split_section_and_chunks(lhs)
        try:
            new_vol = float(rhs.strip())
            if new_vol <= 0:
                continue
        except ValueError:
            safe_print(f"[!] Invalid volume in '{spec}'.")
            continue
        for si in sec_indices:
            v_indices = _resolve_chunks_for_section(si, chunk_spec)
            if v_indices:
                apply_chunk_override_action(
                    scan_results[si]["txt_file"],
                    master_output_dir,
                    v_indices,
                    new_volume=new_vol,
                )
                total_updated += len(v_indices)
                safe_print(f"[OK] Updated volume={new_vol:.2f}x for {len(v_indices)} chunk(s) in '{scan_results[si]['base_name']}'.")

    for spec in _split_semis(reset_chunks):
        sec_indices, chunk_spec = _split_section_and_chunks(spec)
        for si in sec_indices:
            v_indices = _resolve_chunks_for_section(si, chunk_spec)
            if v_indices:
                apply_chunk_override_action(
                    scan_results[si]["txt_file"],
                    master_output_dir,
                    v_indices,
                    clear_override=True,
                )
                total_updated += len(v_indices)
                safe_print(f"[OK] Reset overrides for {len(v_indices)} chunk(s) in '{scan_results[si]['base_name']}'.")

    return total_updated


def convert_audiobook(
    manifest_file=MANIFEST_FILE,
    enable_emotions=None,
    interactive=True,
    normalize_volume=None,
    force_rerecord_menu=False,
    random_narrator=None,
    narrator_gender="any",
    narrator_accent="any",
    narrator_voices="2-4",
    save_random_narrator=False,
    narrator_voice=None,
    set_volumes=None,
    adjust_volumes=False,
    review_tags=None,
    list_tags=False,
    set_tags=None,
    untag_specs=None,
    remap_tags=None,
    map_aliases=None,
    existing_action=None,
    rerecord_sections=None,
    list_chunks=False,
    rerecord_chunks=None,
    set_chunk_voices=None,
    set_chunk_volumes=None,
    reset_chunks=None,
    skip_render=False,
):
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

    # Optional CLI dialogue tag inspection / editing / review before audio rendering
    has_cli_tag_ops = bool(list_tags or set_tags or untag_specs or remap_tags or map_aliases)
    if has_cli_tag_ops or review_tags is True or (interactive and review_tags is None):
        try:
            import llm_dialogue_tagger as ldt
            if has_cli_tag_ops:
                ldt.apply_cli_tag_operations(
                    chapter_files,
                    list_tags=list_tags,
                    set_tags=set_tags,
                    untag_specs=untag_specs,
                    remap_tags=remap_tags,
                    map_aliases=map_aliases,
                )
            if review_tags is True:
                ldt.interactive_review_dialogue_tags(chapter_files, force_open=True)
            elif interactive and review_tags is None:
                ldt.interactive_review_dialogue_tags(chapter_files, force_open=False)
            characters = load_characters()
            tag_mappings = load_tag_mappings()
        except Exception as e:
            safe_print(f"[!] Warning during dialogue tag check: {e}")

    if set_volumes:
        characters = apply_cli_volume_updates(characters, set_volumes)

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

    if narrator_voice:
        nar = dict(characters.get("narrator", {}))
        nar["display_name"] = nar.get("display_name", "narrator")
        nar["engine"] = "kokoro"
        nar["voice"] = narrator_voice
        nar.setdefault("pitch", 1.0)
        nar.setdefault("speed", 1.0)
        nar.setdefault("volume", 1.0)
        characters["narrator"] = nar
        if save_random_narrator:
            save_narrator_voice_update(narrator_voice)
        safe_print(f"\n[*] Narrator voice overridden: {narrator_voice}")
    elif random_narrator is True:
        characters, rolled_voice = apply_random_narrator_to_characters(
            characters,
            gender=narrator_gender or "any",
            accent=narrator_accent or "any",
            voice_count=narrator_voices or "2-4",
            persist=save_random_narrator,
        )
        safe_print(f"\n[*] Randomized Kokoro Narrator: {rolled_voice}")
        safe_print(f"    ({summarize_kokoro_blend(rolled_voice)})")
        if save_random_narrator:
            safe_print(f"[OK] Saved randomized narrator voice to '{CHARACTERS_FILE}'.")
    elif random_narrator is None and interactive:
        characters = interactive_randomize_narrator(characters)

    if adjust_volumes and interactive:
        characters = interactive_adjust_volumes(characters)

    if normalize_volume is None:
        if interactive and not set_volumes and not adjust_volumes:
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
            else:
                normalize_volume = True
        else:
            normalize_volume = True

    safe_print(f"[*] Volume Mode: {'Auto-Balanced' if normalize_volume else 'Raw (Per-Voice Multipliers Only)'}")

    if tag_mappings:
        safe_print(f"\nLoaded {len(tag_mappings)} tag/alias mapping(s) from '{TAG_MAPPINGS_FILE}':")
        for tag, canon in sorted(tag_mappings.items()):
            if tag != canon:
                safe_print(f"  - Tag [{tag}] maps to character [{canon}]")

    parent_dir = os.path.basename(os.path.dirname(chapter_files[0]))
    if parent_dir and parent_dir.lower() not in ("parsed_chapters", "tagged_chapters", "raw_txt", ".", ""):
        book_title_base = parent_dir
    else:
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
    def _build_scan_results():
        results = []
        found_existing = False
        for idx, txt_file in enumerate(chapter_files, 1):
            bname = os.path.splitext(os.path.basename(txt_file))[0]
            final_mp3 = os.path.join(master_output_dir, f"{bname}_complete.mp3")
            chunk_dir = os.path.join(master_output_dir, "chunks_" + bname)
            chunks_count = 0
            if os.path.exists(chunk_dir):
                chunks_count = len([f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".wav") and os.path.getsize(os.path.join(chunk_dir, f)) > 0])

            if os.path.exists(final_mp3) and os.path.getsize(final_mp3) > 0:
                status = "COMPLETED (MP3 exists)"
                found_existing = True
            elif chunks_count > 0:
                status = f"PARTIAL ({chunks_count} chunks rendered)"
                found_existing = True
            else:
                status = "NEW"

            results.append({
                "idx": idx,
                "txt_file": txt_file,
                "base_name": bname,
                "status": status,
            })
        return results, found_existing

    scan_results, has_existing = _build_scan_results()

    # Non-interactive CLI chunk listing & overrides
    has_cli_chunk_ops = bool(rerecord_chunks or set_chunk_voices or set_chunk_volumes or reset_chunks)
    if has_cli_chunk_ops:
        apply_cli_chunk_operations(
            scan_results=scan_results,
            master_output_dir=master_output_dir,
            characters_map=characters,
            tag_mappings=tag_mappings,
            enable_emotions=enable_emotions,
            section_filter=rerecord_sections,
            rerecord_chunks=rerecord_chunks,
            set_chunk_voices=set_chunk_voices,
            set_chunk_volumes=set_chunk_volumes,
            reset_chunks=reset_chunks,
        )
        scan_results, has_existing = _build_scan_results()

    if list_chunks:
        list_chapter_chunks_report(
            scan_results=scan_results,
            master_output_dir=master_output_dir,
            characters_map=characters,
            tag_mappings=tag_mappings,
            enable_emotions=enable_emotions,
            section_filter=rerecord_sections,
        )

    # Non-interactive existing-audio actions
    if existing_action == "rerecord-all":
        safe_print("\n[*] Clearing ALL previous recordings (--rerecord-all). Re-recording entire book from scratch...")
        for item in scan_results:
            clear_chapter_recording(master_output_dir, item["base_name"])
        safe_print("[OK] All sections cleared.")
    elif existing_action == "rerecord-sections" or (rerecord_sections and not has_cli_chunk_ops and not list_chunks and existing_action != "rerecord-chunks"):
        sel_spec = rerecord_sections or "all"
        to_clear = [i + 1 for i in _resolve_target_section_indices(sel_spec, scan_results)]
        for num in to_clear:
            item = scan_results[num - 1]
            clear_chapter_recording(master_output_dir, item["base_name"])
            safe_print(f"[OK] Cleared section '{item['base_name']}' for re-recording.")
    elif existing_action == "rerecord-chunks" or (force_rerecord_menu and interactive):
        interactive_rerecord_chunks(
            scan_results,
            master_output_dir,
            characters,
            tag_mappings=tag_mappings,
            enable_emotions=enable_emotions,
        )
    elif (has_existing or force_rerecord_menu) and interactive and existing_action != "resume" and not has_cli_chunk_ops:
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
        safe_print("  [5] Review / edit dialogue tags in chapter text (if applicable)")
        choice = input("\nSelect how to proceed [1, 2, 3, 4, or 5] (default 1): ").strip()

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
        elif choice == "5":
            try:
                import llm_dialogue_tagger as ldt
                ldt.interactive_review_dialogue_tags(chapter_files, force_open=True)
                characters = load_characters()
                tag_mappings = load_tag_mappings()
            except Exception as e:
                safe_print(f"[!] Could not open dialogue tag reviewer: {e}")
    elif has_existing and not interactive:
        safe_print("\n[*] Existing audio sections detected. Resuming automatically (skipping completed).")

    if skip_render:
        safe_print("\n[*] Skipping audio synthesis (--skip-render).")
        return True

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
    parser.add_argument("--volume-balance", "--auto-balance", dest="normalize_volume", action="store_true", default=None, help="Enable automatic active-speech loudness balancing.")
    parser.add_argument("--no-volume-balance", "--raw-volume", dest="normalize_volume", action="store_false", help="Disable automatic active-speech loudness balancing.")
    parser.add_argument("--set-volume", "--volume", dest="set_volumes", action="append", default=[], help="Update character/variant volume in characters.json, e.g. 'Alice=1.2' or 'Bob:shout=0.9'.")
    parser.add_argument("--adjust-volumes", action="store_true", default=False, help="Open interactive voice volume editor.")
    parser.add_argument("--narrator-voice", type=str, default=None, help="Explicit Kokoro voice or blend string for narrator.")
    parser.add_argument("--random-narrator", dest="random_narrator", action="store_true", default=None, help="Randomize the narrator using random Kokoro TTS voices.")
    parser.add_argument("--keep-narrator", "--no-random-narrator", dest="random_narrator", action="store_false", help="Keep the existing narrator from characters.json without prompting.")
    parser.add_argument(
        "--narrator-gender",
        choices=["female", "male", "mostly-female", "mostly-male", "any"],
        default=None,
        help="Gender filter for randomized narrator: female, male, mostly-female (0.2-0.4 male), mostly-male (0.2-0.4 female), or any.",
    )
    parser.add_argument(
        "--narrator-accent",
        choices=["american", "british", "any"],
        default=None,
        help="Regional accent filter for randomized narrator: american, british, or any.",
    )
    parser.add_argument(
        "--narrator-voices",
        type=str,
        default=None,
        help="Number or range of Kokoro voices to mix for randomized narrator (e.g. '3' or '2-4').",
    )
    parser.add_argument("--save-narrator", dest="save_narrator", action="store_true", default=False, help="Save randomized/updated narrator voice into characters.json.")
    parser.add_argument("--no-save-narrator", dest="save_narrator", action="store_false", help="Use randomized narrator for this run only.")
    parser.add_argument("--review-tags", "--review", dest="review_tags", action="store_true", default=None, help="Interactively review and edit dialogue tags before rendering.")
    parser.add_argument("--no-review-tags", dest="review_tags", action="store_false", help="Skip interactive dialogue tag review.")
    parser.add_argument("--list-tags", "--show-tags", action="store_true", default=False, help="Print numbered dialogue tags for chapters in manifest.")
    parser.add_argument("--set-tag", "--change-tag", "--edit-tag", dest="set_tags", action="append", default=[], help="Change dialogue tag(s), e.g. '1=Alice' or '2,4-6=Bob:whisper'.")
    parser.add_argument("--untag", dest="untag_specs", action="append", default=[], help="Untag dialogue quote(s) so Narrator reads them.")
    parser.add_argument("--remap-tag", "--rename-tag", dest="remap_tags", action="append", default=[], help="Bulk remap speaker tags, e.g. 'OldName=NewName'.")
    parser.add_argument("--map-alias", dest="map_aliases", action="append", default=[], help="Map tag alias in tag_mappings.json, e.g. 'bobby=Bob'.")
    parser.add_argument("--existing-action", "--on-existing", dest="existing_action", choices=["resume", "rerecord-all", "rerecord-sections", "rerecord-chunks"], default=None, help="Action when existing audio is detected.")
    parser.add_argument("--resume", dest="existing_action", action="store_const", const="resume", help="Resume existing audio without prompting.")
    parser.add_argument("--rerecord-all", dest="existing_action", action="store_const", const="rerecord-all", help="Clear and re-record all chapters from scratch.")
    parser.add_argument("--rerecord-sections", "--sections", dest="rerecord_sections", type=str, default=None, help="Specific section indices or names to re-record or target (e.g. '1,3' or '2-4' or 'all').")
    parser.add_argument("--rerecord", action="store_true", default=False, help="Jump directly to chunk-level re-record / voice switching menu.")
    parser.add_argument("--list-chunks", "--show-chunks", action="store_true", default=False, help="Print numbered audio chunks for selected sections.")
    parser.add_argument("--rerecord-chunks", dest="rerecord_chunks", action="append", default=[], help="Re-record specific chunks, e.g. '2,5-7' or '1:2,5-7'.")
    parser.add_argument("--set-chunk-voice", "--chunk-voice", dest="set_chunk_voices", action="append", default=[], help="Override chunk voice, e.g. '2,3=Bob:whisper@1.1' or '4=:shout'.")
    parser.add_argument("--set-chunk-volume", "--chunk-volume", dest="set_chunk_volumes", action="append", default=[], help="Override chunk volume multiplier, e.g. '2,3=1.25'.")
    parser.add_argument("--reset-chunks", dest="reset_chunks", action="append", default=[], help="Clear chunk voice/volume overrides, e.g. '2,3' or 'all'.")
    parser.add_argument("--skip-render", "--no-render", action="store_true", default=False, help="Apply inspection/tag/chunk edits without running TTS synthesis.")
    parser.add_argument("--interactive", "-i", action="store_true", default=False, help="Prompt interactively for unspecified options.")
    parser.add_argument("--yes", "-y", "--non-interactive", dest="non_interactive", action="store_true", default=False, help="Run non-interactively with defaults for unspecified options.")
    args, _ = parser.parse_known_args()

    has_cli_flags = len(sys.argv) > 1
    is_interactive = args.interactive or (not has_cli_flags and not args.non_interactive)

    if args.random_narrator is True or any(
        val is not None for val in (args.narrator_gender, args.narrator_accent, args.narrator_voices)
    ):
        use_random_narrator = True
    elif args.random_narrator is False:
        use_random_narrator = False
    else:
        use_random_narrator = None

    convert_audiobook(
        manifest_file=args.manifest,
        enable_emotions=args.enable_emotions,
        interactive=is_interactive,
        normalize_volume=args.normalize_volume,
        force_rerecord_menu=args.rerecord,
        random_narrator=use_random_narrator,
        narrator_gender=args.narrator_gender or "any",
        narrator_accent=args.narrator_accent or "any",
        narrator_voices=args.narrator_voices or "2-4",
        save_random_narrator=args.save_narrator,
        narrator_voice=args.narrator_voice,
        set_volumes=args.set_volumes,
        adjust_volumes=args.adjust_volumes,
        review_tags=args.review_tags,
        list_tags=args.list_tags,
        set_tags=args.set_tags,
        untag_specs=args.untag_specs,
        remap_tags=args.remap_tags,
        map_aliases=args.map_aliases,
        existing_action=args.existing_action,
        rerecord_sections=args.rerecord_sections,
        list_chunks=args.list_chunks,
        rerecord_chunks=args.rerecord_chunks,
        set_chunk_voices=args.set_chunk_voices,
        set_chunk_volumes=args.set_chunk_volumes,
        reset_chunks=args.reset_chunks,
        skip_render=args.skip_render,
    )


if __name__ == "__main__":
    main()
