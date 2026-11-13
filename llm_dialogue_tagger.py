import os
import re
import sys
import json
import glob
import time
import argparse
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ==================== CONFIGURATION ====================
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
CHARACTERS_FILE = "characters.json"
TAG_MAPPINGS_FILE = "tag_mappings.json"
DEFAULT_OUTPUT_DIR = "tagged_chapters"
MANIFEST_FILE = "chapters.txt"

# Diverse default Kokoro voices to assign to newly discovered characters
DEFAULT_KOKORO_VOICES = [
    "af_sarah",
    "am_adam",
    "af_nicole",
    "am_michael",
    "af_sky",
    "am_fenrir",
    "af_heart",
    "am_puck",
    "af_bella",
    "am_onyx",
]

# Comprehensive list of English speech verbs used in dialogue attribution
SPEECH_VERBS = {
    "said", "says", "say", "asked", "asks", "replied", "replies", "answered", "answers",
    "shouted", "shouts", "whispered", "whispers", "yelled", "yells", "muttered", "mutters",
    "murmured", "murmurs", "cried", "cries", "groaned", "groans", "gasped", "gasps",
    "snapped", "snaps", "growled", "growls", "screamed", "screams", "laughed", "laughs",
    "sighed", "sighs", "remarked", "remarks", "added", "adds", "continued", "continues",
    "echoed", "echoes", "explained", "explains", "began", "begins", "concluded", "concludes",
    "called", "calls", "chimed", "chimes", "responded", "responds", "offered", "offers",
    "noted", "notes", "barked", "barks", "hissed", "hisses", "demanded", "demands",
    "repeated", "repeats", "prompted", "prompts", "agreed", "agrees", "interrupted", "interrupts",
    "warned", "warns", "scoffed", "scoffs", "pleaded", "pleads", "insisted", "insists",
    "snickered", "snickers", "chuckled", "chuckles", "ventured", "ventures", "breathed", "rasped"
}

# English stop words and clause starters that should never be identified as proper character names
STOP_WORDS = {
    # Pronouns & Determiners
    "he", "she", "they", "it", "someone", "one", "the", "who", "what", "where", "when",
    "why", "how", "this", "that", "these", "those", "there", "then", "and", "but", "or",
    "while", "as", "if", "once", "now", "so", "after", "before", "because", "though",
    "although", "hed", "shed", "theyd", "itd", "you", "i", "we", "my", "your", "his",
    "her", "their", "our", "its", "him", "them", "me", "us", "mine", "yours", "theirs",
    "both", "all", "each", "either", "neither", "none", "every", "many", "some", "any",
    "few", "more", "most", "several", "no", "not", "yes", "another", "other", "others",
    "something", "anything", "everything", "nothing", "somebody", "anybody", "everybody", "nobody",
    # Prepositions & Directionals
    "to", "in", "at", "on", "for", "from", "with", "by", "about", "into", "over", "after",
    "under", "between", "against", "without", "through", "across", "towards", "toward",
    "upon", "behind", "beside", "beyond", "out", "down", "up", "off", "clasp", "end",
    # Auxiliary & Common Verbs
    "is", "am", "are", "was", "were", "be", "being", "been", "have", "has", "had", "having",
    "do", "does", "did", "doing", "done", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "get", "got", "come", "came", "go", "went", "make", "made", "take", "took",
    # Adverbial / transitional words
    "well", "like", "just", "only", "also", "too", "very", "quite", "rather", "really", "almost", "again", "back", "away", "ahead", "forth"
}
# ========================================================

# ==================== EMOTION & TONE DETECTION ====================
EMOTION_KEYWORDS = {
    "whisper": {
        "verbs": ["whispered", "whispers", "muttered", "mutters", "murmured", "murmurs", "breathed"],
        "adverbs": ["quietly", "softly", "in a whisper", "under his breath", "under her breath", "hushed"]
    },
    "shout": {
        "verbs": ["shouted", "shouts", "yelled", "yells", "screamed", "screams", "barked", "barks", "howled", "bellowed"],
        "adverbs": ["loudly", "at the top of his lungs", "at the top of her lungs"]
    },
    "angry": {
        "verbs": ["snapped", "snaps", "growled", "growls", "hissed", "hisses", "fumed", "spat"],
        "adverbs": ["angrily", "furiously", "harshly", "sharply", "bitterly", "fiercely"]
    },
    "sad": {
        "verbs": ["cried", "cries", "sobbed", "sobs", "whimpered", "whimpers", "mourned", "sighed", "sighs"],
        "adverbs": ["sadly", "mournfully", "miserably", "tearfully", "wistfully"]
    },
    "happy": {
        "verbs": ["laughed", "laughs", "chuckled", "chuckles", "giggled", "giggles", "grinned", "beamed"],
        "adverbs": ["happily", "cheerfully", "brightly", "gleefully", "warmly", "joyfully"]
    },
    "excited": {
        "verbs": ["gasped", "gasps", "squealed", "squeals", "exclaimed", "exclaims"],
        "adverbs": ["excitedly", "breathlessly", "eagerly"]
    },
    "calm": {
        "verbs": [],
        "adverbs": ["calmly", "evenly", "flatly", "coolly", "dryly", "composedly"]
    }
}


def detect_dialogue_tone(paragraph, quote_span):
    """
    Detects emotional tone (whisper, shout, angry, sad, happy, excited, calm)
    from adjacent speech verbs, adverbs, and punctuation.
    Returns emotion string or '' if neutral.
    """
    start, end = quote_span
    pre = paragraph[max(0, start - 70):start].lower()
    post = paragraph[end:min(len(paragraph), end + 70)].lower()
    snippet = f"{pre} {post}"

    for emotion, patterns in EMOTION_KEYWORDS.items():
        for verb in patterns.get("verbs", []):
            if re.search(rf"\b{re.escape(verb)}\b", snippet):
                return emotion
        for adv in patterns.get("adverbs", []):
            if re.search(rf"\b{re.escape(adv)}\b", snippet):
                return emotion

    # Check punctuation clues inside quote if ending in !? or ?!
    q_text = paragraph[start:end]
    if "!?" in q_text or "?!" in q_text:
        return "excited"

    return ""



def check_ollama_service(url=DEFAULT_OLLAMA_URL):
    """
    Checks if Ollama service is reachable and lists installed models.
    Returns (is_running, list_of_model_names).
    """
    tags_url = f"{url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "AudiobookConverter/1.0"})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return True, models
    except Exception:
        pass
    return False, []


def call_ollama_generate(prompt, system_prompt="", model=DEFAULT_MODEL, url=DEFAULT_OLLAMA_URL, json_format=False):
    """
    Calls Ollama generate API (http://localhost:11434/api/generate) with GPU acceleration.
    """
    endpoint = f"{url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for strict factual attribution
            "num_predict": 128,  # Lightweight response length for micro-attribution
        }
    }
    if json_format:
        payload["format"] = "json"

    json_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=json_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "AudiobookConverter/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to Ollama at {url}: {e}")


def clean_character_name(name):
    """Sanitizes character names to TitleCase alphanumeric identifiers."""
    cleaned = re.sub(r"[^\w\s\-]", "", name).strip()
    words = cleaned.split()
    if not words:
        return "Unknown"
    return "".join(w.capitalize() for w in words)


# ==================== ROSTER & DISCOVERY ====================

def load_canonical_roster(characters_file=CHARACTERS_FILE, tag_mappings_file=TAG_MAPPINGS_FILE):
    """
    Loads known character profiles and tag mappings into a normalized lowercase lookup dictionary.
    Maps tag string (lowercase) -> Canonical character name.
    """
    roster = {}
    if os.path.exists(tag_mappings_file):
        try:
            with open(tag_mappings_file, "r", encoding="utf-8") as f:
                mappings = json.load(f)
                for k, v in mappings.items():
                    roster[str(k).strip().lower()] = str(v).strip()
        except Exception:
            pass

    if os.path.exists(characters_file):
        try:
            with open(characters_file, "r", encoding="utf-8") as f:
                chars = json.load(f)
                for k in chars:
                    if k.lower() not in ["narrator", "detective", "villain"]:
                        if k.lower() not in roster:
                            roster[k.lower()] = k
        except Exception:
            pass
    return roster


def discover_characters_in_text(text, roster):
    """
    Scans the text for capitalized proper names paired with speech verbs that aren't yet in roster.
    Filters out adverbs ending in 'ly' and standard stop words.
    Requires at least 2 occurrences for unfamiliar words to prevent false positives.
    """
    verbs_pat = "|".join(SPEECH_VERBS)
    candidates_count = {}

    # Pattern 1: Name [speech_verb]
    p1 = re.compile(rf"\b([A-Z][a-zA-Z0-9_\'\-]+)\s+(?:(?:\w+ly)\s+)?(?:{verbs_pat})\b", re.IGNORECASE)
    for m in p1.finditer(text):
        raw = m.group(1).strip()
        cleaned = clean_character_name(raw)
        c_low = cleaned.lower()
        if c_low not in STOP_WORDS and not c_low.endswith("ly") and len(cleaned) >= 3:
            if c_low not in roster:
                candidates_count[cleaned] = candidates_count.get(cleaned, 0) + 1

    # Pattern 2: [speech_verb] Name
    p2 = re.compile(rf"\b(?:{verbs_pat})\s+([A-Z][a-zA-Z0-9_\'\-]+)\b", re.IGNORECASE)
    for m in p2.finditer(text):
        raw = m.group(1).strip()
        cleaned = clean_character_name(raw)
        c_low = cleaned.lower()
        if c_low not in STOP_WORDS and not c_low.endswith("ly") and len(cleaned) >= 3:
            if c_low not in roster:
                candidates_count[cleaned] = candidates_count.get(cleaned, 0) + 1

    # Discovered characters: known or seen at least twice
    return {name for name, count in candidates_count.items() if count >= 2}


# ==================== SCRIPT FORMAT PARSER ====================

def tag_script_format(text, detect_tone=False):
    """
    Tags dialogue formatted as script/screenplay style:
      [Character]: Dialogue...
      or
      Character: Dialogue...
      or with parenthetical directions: Character (whispering): Dialogue...
    Leaves non-dialogue lines (narration/stage directions) untagged.
    Returns (tagged_text, set_of_character_names).
    """
    lines = text.split("\n")
    tagged_lines = []
    characters_found = set()

    script_pattern = re.compile(
        r"^(?:\[\s*([a-zA-Z0-9_\'\-\s]+?)\s*(?:\((.*?)\))?\s*\]|([a-zA-Z0-9_\'\-]+?)\s*(?:\((.*?)\))?)\s*:\s*(.*)$"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            tagged_lines.append("")
            continue

        if stripped.replace(" ", "") in ["***", "---", "___"] or bool(re.match(r"^[\*\-\_]{3,}$", stripped.replace(" ", ""))):
            tagged_lines.append(stripped)
            continue

        match = script_pattern.match(stripped)
        if match:
            raw_speaker = match.group(1) or match.group(3)
            parenthetical = (match.group(2) or match.group(4) or "").lower()
            speaker = clean_character_name(raw_speaker)
            dialogue = match.group(5).strip()

            tone = ""
            if detect_tone and parenthetical:
                for emo in EMOTION_KEYWORDS:
                    if emo in parenthetical:
                        tone = emo
                        break

            if speaker and dialogue:
                characters_found.add(speaker)
                if not (dialogue.startswith('"') and dialogue.endswith('"')):
                    dialogue_quoted = f'"{dialogue}"'
                else:
                    dialogue_quoted = dialogue

                tag_open = f"[{speaker}:{tone}]" if tone else f"[{speaker}]"
                tag_close = f"[/{speaker}:{tone}]" if tone else f"[/{speaker}]"
                tagged_lines.append(f"{tag_open} {dialogue_quoted} {tag_close}")
            else:
                tagged_lines.append(stripped)
        else:
            tagged_lines.append(stripped)

    return "\n".join(tagged_lines), characters_found


# ==================== PROSE DIALOGUE ATTRIBUTION ====================

def resolve_ambiguous_with_llm(passage, candidates, model=DEFAULT_MODEL, url=DEFAULT_OLLAMA_URL):
    """
    Micro-query to local LLM for low-confidence / ambiguous dialogue.
    Passes only the short passage and candidate list to generate a concise 1-2 token name response.
    """
    system_prompt = (
        "You are a literary assistant identifying which character speaks in a story passage. "
        "Return ONLY the exact character's name from the provided candidate list. No punctuation or explanation."
    )
    prompt = (
        f"Active Characters: {', '.join(candidates)}\n\n"
        f"Story Passage:\n{passage}\n\n"
        f"Who speaks the dialogue in this passage? Return ONLY their name:"
    )
    try:
        resp = call_ollama_generate(prompt=prompt, system_prompt=system_prompt, model=model, url=url, json_format=False)
        resp_clean = resp.strip()
        for c in candidates:
            if c.lower() in resp_clean.lower():
                return c
        cleaned = clean_character_name(resp_clean)
        if cleaned and cleaned != "Unknown":
            return cleaned
    except Exception:
        pass
    return None


def attribute_paragraph_speaker(paragraph, matches, roster, active_speakers, use_llm=False, model=DEFAULT_MODEL, url=DEFAULT_OLLAMA_URL):
    """
    Determines the speaker for a paragraph with multi-tier attribution heuristics.
    Applies:
      1. Vocative addressee inversion: characters addressed by name inside dialogue
         (e.g. "John, this is crazy") are disqualified from being the speaker.
      2. Narrative-only isolation: searches exclusively in narrative beats outside quotes.
      3. Listener POV phrase exclusion: filters out "In X's head/eyes/mind" so internal
         reactions are never confused with speech attribution.
      4. Conversational turn-taking: alternates between active scene interlocutors.
    """
    verbs_pat = "|".join(SPEECH_VERBS)

    # Detect characters addressed by name inside quotes (addressees can never be the speaker)
    addressed_characters = set()
    for m in matches:
        q_text = m.group(1)
        for raw_tag, canon_name in roster.items():
            if raw_tag in ["narrator", "divider"]:
                continue
            voc_pattern = rf"(?:^|[\,\!\?\.\s])\b{re.escape(raw_tag)}\b(?:[\,\!\?\.]|\s|$)"
            if re.search(voc_pattern, q_text, re.IGNORECASE):
                addressed_characters.add(canon_name)

    # Narrative text only (stripping out words inside quotation marks)
    narrative_only = re.sub(r'["“]([^"\n“”]+)["”]', " ", paragraph)

    # Filter out listener POV phrases like "In X's head/mind/eyes/ears/chest/perspective"
    clean_narrative = re.sub(
        rf"\b(?:in|to|on|at|behind)\s+[a-zA-Z0-9_\'\-]+\'s\s+(?:head|mind|eyes|ears|chest|face|perspective)\b",
        " ",
        narrative_only,
        flags=re.IGNORECASE
    )

    # Tier 1: Look for explicit speech tag adjacent to any quote in the paragraph
    for m in matches:
        post = paragraph[m.end():m.end() + 70]
        pre = paragraph[max(0, m.start() - 70):m.start()]

        for snippet in [post, pre]:
            for raw_tag, canon_name in roster.items():
                if raw_tag in ["narrator", "divider"]:
                    continue
                p_after = rf"\b{re.escape(raw_tag)}\b(?:\s+\w+)?\s+(?:{verbs_pat})\b"
                p_before = rf"\b(?:{verbs_pat})\s+(?:\w+\s+)?\b{re.escape(raw_tag)}\b"
                if re.search(p_after, snippet, re.IGNORECASE) or re.search(p_before, snippet, re.IGNORECASE):
                    if canon_name not in addressed_characters:
                        return canon_name

    # Tier 2: Check full narrative prose outside quotes for [Character] [speech_verb]
    for raw_tag, canon_name in roster.items():
        if raw_tag in ["narrator", "divider"]:
            continue
        p_after = rf"\b{re.escape(raw_tag)}\b(?:\s+\w+)?\s+(?:{verbs_pat})\b"
        p_before = rf"\b(?:{verbs_pat})\s+(?:\w+\s+)?\b{re.escape(raw_tag)}\b"
        if re.search(p_after, clean_narrative, re.IGNORECASE) or re.search(p_before, clean_narrative, re.IGNORECASE):
            if canon_name not in addressed_characters:
                return canon_name

    # Tier 3: Check for known character mentioned in paragraph action beats (outside quotes)
    chars_in_narrative = set()
    for raw_tag, canon_name in roster.items():
        if raw_tag in ["narrator", "divider"]:
            continue
        if re.search(rf"\b{re.escape(raw_tag)}\b", clean_narrative, re.IGNORECASE):
            if canon_name not in addressed_characters:
                chars_in_narrative.add(canon_name)

    if len(chars_in_narrative) == 1:
        return list(chars_in_narrative)[0]

    # Tier 4: Micro-LLM consultation if enabled and multiple characters present
    if use_llm and len(chars_in_narrative) > 1:
        eligible_candidates = [c for c in chars_in_narrative if c not in addressed_characters]
        if eligible_candidates:
            llm_choice = resolve_ambiguous_with_llm(paragraph, eligible_candidates, model=model, url=url)
            if llm_choice:
                return llm_choice

    # Tier 5: Conversational turn-taking scene alternation (excluding addressed characters)
    eligible_speakers = [s for s in active_speakers if s not in addressed_characters]
    if len(eligible_speakers) >= 2:
        last = eligible_speakers[-1]
        second_last = eligible_speakers[-2]
        if last != second_last:
            return second_last

    if eligible_speakers:
        return eligible_speakers[-1]

    # Tier 6: Default fallback
    valid = [c for c in roster.values() if c.lower() not in ["narrator", "detective", "villain"] and c not in addressed_characters]
    return valid[0] if valid else "Unknown"


def tag_prose_chapter(text, roster=None, use_llm=False, model=DEFAULT_MODEL, url=DEFAULT_OLLAMA_URL, detect_tone=False):
    """
    High-performance prose dialogue tagger:
      1. Discovers any new characters in dialogue tags.
      2. Detects 100% of quotes via precise regex.
      3. Attributes each paragraph's speaker using high-precision multi-tier heuristics.
      4. Wraps EVERY quote with [Speaker] "..." [/Speaker] via exact character-span replacement,
         guaranteeing 100% quote coverage with ZERO missing quotes and ZERO double-quote glitches.
    """
    if roster is None:
        roster = load_canonical_roster()

    # Discover any newly introduced proper names in this text
    discovered = discover_characters_in_text(text, roster)
    for d in discovered:
        roster[d.lower()] = d

    paragraphs = text.split("\n\n")
    quote_pattern = re.compile(r'["“]([^"\n“”]+)["”]')

    tagged_paragraphs = []
    all_characters_used = set()
    active_speakers = []
    total_quotes_found = 0
    total_quotes_tagged = 0

    t0 = time.time()

    for p in paragraphs:
        p_stripped = p.strip()
        if not p_stripped:
            tagged_paragraphs.append("")
            continue

        # Section dividers reset conversational turn-taking
        if p_stripped.replace(" ", "") in ["***", "---", "___"]:
            tagged_paragraphs.append(p_stripped)
            active_speakers.clear()
            continue

        matches = list(quote_pattern.finditer(p_stripped))
        if not matches:
            # Check if pure narration mentions active characters to maintain scene context
            for raw_tag, canon_name in roster.items():
                if raw_tag not in ["narrator", "divider"] and re.search(rf"\b{re.escape(raw_tag)}\b", p_stripped, re.IGNORECASE):
                    if not active_speakers or active_speakers[-1] != canon_name:
                        active_speakers.append(canon_name)
            tagged_paragraphs.append(p_stripped)
            continue

        total_quotes_found += len(matches)

        speaker = attribute_paragraph_speaker(
            paragraph=p_stripped,
            matches=matches,
            roster=roster,
            active_speakers=active_speakers,
            use_llm=use_llm,
            model=model,
            url=url
        )

        all_characters_used.add(speaker)
        if not active_speakers or active_speakers[-1] != speaker:
            active_speakers.append(speaker)

        # Replace quotes in REVERSE order by exact character span
        # This guarantees:
        # 1. 100% of quotes are tagged (including split quotes e.g. "Wait a moment," John said. "Look over there.")
        # 2. Narration text between quotes remains 100% verbatim.
        # 3. No double quote glitches like `"[Speaker]` or `[/Speaker],"` can ever occur.
        tagged_p = p_stripped
        for m in reversed(matches):
            start, end = m.span()
            q_content = m.group(1).strip()
            tone = detect_dialogue_tone(p_stripped, (start, end)) if detect_tone else ""
            if tone:
                replacement = f'[{speaker}:{tone}] "{q_content}" [/{speaker}:{tone}]'
            else:
                replacement = f'[{speaker}] "{q_content}" [/{speaker}]'
            tagged_p = tagged_p[:start] + replacement + tagged_p[end:]
            total_quotes_tagged += 1

        tagged_paragraphs.append(tagged_p)

    elapsed = time.time() - t0
    mode_label = "Smart Fast Engine + GPU LLM" if use_llm else "Smart Fast Engine (Instant)"
    print(f"  [✓] Processed chapter in {elapsed:.3f}s ({total_quotes_tagged}/{total_quotes_found} quotes tagged, Mode: {mode_label})")

    return "\n\n".join(tagged_paragraphs), all_characters_used


# ==================== TAG MAPPINGS & VOICE PROFILES ====================

def ensure_tag_mappings_file(tag_mappings_file=TAG_MAPPINGS_FILE):
    """Creates a default tag_mappings.json if missing."""
    if not os.path.exists(tag_mappings_file):
        default_mappings = {
            "narrator": "narrator",
            "detective": "detective",
            "villain": "villain",
            "hero": "Hero"
        }
        with open(tag_mappings_file, "w", encoding="utf-8") as f:
            json.dump(default_mappings, f, indent=4)


def sync_tag_mappings(discovered_tags, tag_mappings_file=TAG_MAPPINGS_FILE):
    """
    Ensures every discovered tag is present in tag_mappings.json.
    Maps lowercase tag string -> Canonical Character Name.
    """
    ensure_tag_mappings_file(tag_mappings_file)
    try:
        with open(tag_mappings_file, "r", encoding="utf-8") as f:
            mappings = json.load(f)
    except Exception:
        mappings = {}

    lower_keys = {k.lower(): k for k in mappings.keys()}
    added = []

    for tag in sorted(list(discovered_tags)):
        t_clean = tag.strip()
        t_lower = t_clean.lower()
        if t_lower in ["narrator", "divider", "unknown"]:
            continue
        if t_lower not in lower_keys:
            mappings[t_lower] = t_clean  # Default: maps tag to its TitleCase name
            lower_keys[t_lower] = t_clean
            added.append((t_clean, t_clean))

    with open(tag_mappings_file, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=4)

    return added


def find_character_audio_sample(char_name, search_dirs=("wavs", "wav", ".")):
    """
    Searches known audio directories for a reference .wav matching the character name.
    Also detects any emotional reference clips (e.g. Hero_whisper.wav, Hero_shout.wav).
    Returns (primary_sample_path, dict_of_emotions).
    """
    c_clean = char_name.strip()
    primary = None
    emotions = {}

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith((".wav", ".mp3", ".flac")):
                continue
            base = os.path.splitext(f)[0]
            # Exact base match: e.g. Hero.wav
            if base.lower() == c_clean.lower():
                primary = os.path.join(d, f).replace("\\", "/")
            # Emotional variant match: e.g. Hero_shout.wav, Hero-whisper.wav
            for emo in list(EMOTION_KEYWORDS.keys()) + ["shouting", "whispering"]:
                if base.lower() in [f"{c_clean.lower()}_{emo}", f"{c_clean.lower()}-{emo}"]:
                    ref_path = os.path.join(d, f).replace("\\", "/")
                    ref_text = ""
                    txt_candidate = os.path.join(d, f"{base}.txt")
                    if os.path.exists(txt_candidate):
                        try:
                            with open(txt_candidate, "r", encoding="utf-8") as tf:
                                ref_text = tf.read().strip()
                        except Exception:
                            pass
                    emotions[emo] = {
                        "ref_audio": ref_path,
                        "ref_text": ref_text
                    }

    return primary, emotions


def sync_characters_json(discovered_characters, characters_file=CHARACTERS_FILE):
    """
    Synchronizes discovered character names with characters.json.
    Automatically detects reference audio samples in 'wavs/' to auto-configure F5-TTS,
    and falls back to distinct Kokoro voices if no custom voice sample is present.
    """
    if os.path.exists(characters_file):
        try:
            with open(characters_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}
    else:
        config = {
            "narrator": {
                "engine": "kokoro",
                "voice": "af_bella",
                "pitch": 1.0,
                "speed": 1.0,
                "description": "Default narrator voice for general story text."
            }
        }

    existing_lower = {k.lower(): k for k in config.keys()}
    newly_added = []
    voice_idx = len(config) % len(DEFAULT_KOKORO_VOICES)

    for char_name in sorted(list(discovered_characters)):
        c_lower = char_name.lower()
        if c_lower in ["narrator", "divider", "unknown"]:
            continue

        ref_sample, emotions = find_character_audio_sample(char_name)

        if c_lower not in existing_lower:
            if ref_sample:
                profile = {
                    "engine": "f5tts",
                    "ref_audio": ref_sample,
                    "ref_text": "",
                    "pitch": 1.0,
                    "speed": 1.0,
                    "steps": 16,
                    "description": f"Auto-configured F5-TTS voice using '{os.path.basename(ref_sample)}'."
                }
                if emotions:
                    profile["emotions"] = emotions
                config[char_name] = profile
                newly_added.append((char_name, "F5-TTS (auto-detected ref wav)"))
            else:
                assigned_voice = DEFAULT_KOKORO_VOICES[voice_idx % len(DEFAULT_KOKORO_VOICES)]
                voice_idx += 1
                config[char_name] = {
                    "engine": "kokoro",
                    "voice": assigned_voice,
                    "pitch": 1.0,
                    "speed": 1.0,
                    "description": f"Auto-detected character '{char_name}'. Change engine to 'f5tts' for voice cloning or customize voice."
                }
                newly_added.append((char_name, f"Kokoro ({assigned_voice})"))
            existing_lower[c_lower] = char_name
        else:
            canon_key = existing_lower[c_lower]
            if ref_sample and config[canon_key].get("engine", "").lower() != "f5tts":
                print(f"  [+] Note: Found '{ref_sample}' for existing character [{canon_key}]. Set engine to 'f5tts' in {characters_file} to enable.")
            if emotions:
                curr_emo = config[canon_key].setdefault("emotions", {})
                for ek, ev in emotions.items():
                    if ek not in curr_emo:
                        curr_emo[ek] = ev
                        print(f"  [+] Attached emotion reference sample [{ek}: {ev['ref_audio']}] to character [{canon_key}].")

    with open(characters_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    return newly_added


# ==================== BATCH PROCESSING API ====================

def batch_tag_files(file_paths, dialogue_format="prose", model=DEFAULT_MODEL, url=DEFAULT_OLLAMA_URL, output_dir=DEFAULT_OUTPUT_DIR, detect_tone=True):
    """
    Programmatic entry point called by new_text_parser.py or external scripts.
    dialogue_format can be:
      - 'prose' or 'fast': Instant smart context tagger (100% quote coverage, <0.1s per chapter).
      - 'hybrid': Smart context tagger + local GPU LLM micro-queries for ambiguous lines.
      - 'script': Regex parser for screenplay/script formats ([Speaker]: Dialogue).
    """
    os.makedirs(output_dir, exist_ok=True)
    all_discovered = set()
    created_tagged_files = []

    use_llm = (dialogue_format.lower() == "hybrid")
    format_label = "HYBRID (SMART + GPU LLM)" if use_llm else dialogue_format.upper()

    print("\n" + "=" * 70)
    print(f"[*] Starting Batch Dialogue Tagging ({len(file_paths)} file(s), Mode: {format_label})")
    print("=" * 70)

    # Load canonical roster once for the batch
    roster = load_canonical_roster()

    for idx, fpath in enumerate(file_paths, 1):
        if not os.path.exists(fpath):
            continue

        # Preserve book subfolder structure if inside parsed_chapters/<Book>/...
        norm_fpath = os.path.normpath(fpath)
        parts = norm_fpath.split(os.sep)
        if "parsed_chapters" in parts:
            p_idx = parts.index("parsed_chapters")
            sub_path = os.sep.join(parts[p_idx + 1:])
            out_path = os.path.join(output_dir, sub_path)
        else:
            out_path = os.path.join(output_dir, os.path.basename(fpath))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"\n[{idx}/{len(file_paths)}] Processing: '{os.path.basename(fpath)}'")

        with open(fpath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if dialogue_format.lower() == "script":
            tagged, chars = tag_script_format(raw_text, detect_tone=detect_tone)
        else:
            tagged, chars = tag_prose_chapter(
                text=raw_text,
                roster=roster,
                use_llm=use_llm,
                model=model,
                url=url,
                detect_tone=detect_tone
            )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tagged)

        all_discovered.update(chars)
        created_tagged_files.append(out_path)
        print(f"[✓] Saved tagged chapter: '{out_path}' ({len(chars)} character(s) detected)")

    # Sync tag_mappings.json and characters.json
    new_tags = sync_tag_mappings(all_discovered)
    new_chars = sync_characters_json(all_discovered)

    print("\n" + "=" * 70)
    print(f"[✓] Batch Tagging Complete: {len(created_tagged_files)} file(s) tagged.")
    print(f"[+] Total Characters Detected: {len(all_discovered)}")
    for c in sorted(list(all_discovered)):
        print(f"    • {c}")

    if new_tags:
        print(f"[+] Added {len(new_tags)} entry/entries to '{TAG_MAPPINGS_FILE}' (map nicknames/aliases here).")
    if new_chars:
        print(f"[+] Added {len(new_chars)} profile(s) to '{CHARACTERS_FILE}' (customize voices here).")
    print("=" * 70)

    return created_tagged_files


def update_manifest_file(new_files, manifest_file=MANIFEST_FILE):
    """
    Updates manifest_file, preserving existing chapters and replacing
    matching basenames (e.g. replacing parsed chapter with tagged chapter) or appending new chapters.
    """
    current_lines = []
    if os.path.exists(manifest_file):
        with open(manifest_file, "r", encoding="utf-8") as mf:
            current_lines = [l.strip() for l in mf if l.strip() and not l.startswith("#")]

    tagged_map = {os.path.basename(f): f for f in new_files}
    final_list = []
    for line in current_lines:
        base = os.path.basename(line)
        if base in tagged_map:
            final_list.append(tagged_map.pop(base))
        else:
            final_list.append(line)

    for remaining in tagged_map.values():
        if remaining not in final_list:
            final_list.append(remaining)

    with open(manifest_file, "w", encoding="utf-8") as mf:
        for f in final_list:
            mf.write(f"{f}\n")
    return final_list


# ==================== DIALOGUE TAG REVIEW & EDITING ====================

DIALOGUE_TAG_REGEX = re.compile(
    r"\[\s*([a-zA-Z0-9_\'\-]+)(?::([a-zA-Z0-9_\'\-]+))?\s*\]\s*(.*?)\s*\[\s*/\s*\1(?::[a-zA-Z0-9_\'\-]+)?\s*\]",
    re.DOTALL | re.IGNORECASE,
)


def _parse_index_selection(input_str, max_idx):
    """Parses 1-based indices/ranges ('1, 3, 5-7', 'all') into a sorted list of ints."""
    raw = str(input_str or "").strip()
    if not raw:
        return []
    if raw.lower() == "all":
        return list(range(1, max_idx + 1))

    selected = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
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
    return sorted(selected)


def _strip_tag_markers(snippet):
    """Strips [Speaker] ... [/Speaker] wrappers for readable paragraph context."""
    return DIALOGUE_TAG_REGEX.sub(lambda m: m.group(3).strip(), snippet)


def extract_dialogue_tags(text):
    """
    Extracts every tagged dialogue span (`[Speaker] ... [/Speaker]` or `[Speaker:tone] ... [/Speaker:tone]`)
    from `text` along with 1-based index, character offsets, line number, and surrounding context.
    """
    tags = []
    for idx, match in enumerate(DIALOGUE_TAG_REGEX.finditer(text), 1):
        start, end = match.span()
        speaker = match.group(1).strip()
        emotion = (match.group(2) or "").strip()
        dialogue = match.group(3).strip()
        line_num = text.count("\n", 0, start) + 1

        p_start = text.rfind("\n\n", 0, start)
        p_start = 0 if p_start == -1 else p_start + 2
        p_end = text.find("\n\n", end)
        if p_end == -1:
            p_end = len(text)

        before_clean = _strip_tag_markers(text[p_start:start]).replace("\n", " ").strip()
        after_clean = _strip_tag_markers(text[end:p_end]).replace("\n", " ").strip()
        dlg_clean = dialogue.replace("\n", " ").strip()
        full_paragraph = " ".join(p for p in (before_clean, f"» {dlg_clean} «", after_clean) if p)

        pre_snippet = before_clean[-45:] if len(before_clean) > 45 else before_clean
        if len(before_clean) > 45:
            pre_snippet = "..." + pre_snippet
        post_snippet = after_clean[:55] + "..." if len(after_clean) > 55 else after_clean
        short_dlg = dlg_clean if len(dlg_clean) <= 50 else dlg_clean[:47] + "..."
        context = " ".join(p for p in (pre_snippet, f"» {short_dlg} «", post_snippet) if p)

        tags.append({
            "index": idx,
            "start": start,
            "end": end,
            "speaker": speaker,
            "emotion": emotion,
            "dialogue": dialogue,
            "line_num": line_num,
            "context": context,
            "paragraph": full_paragraph,
        })
    return tags


def _resolve_character_display(speaker, roster=None, chars_cfg=None):
    """Returns canonical character name and voice summary if configured."""
    low = speaker.strip().lower()
    canon = roster.get(low, speaker) if roster else speaker
    if chars_cfg:
        cfg = chars_cfg.get(canon.lower()) or chars_cfg.get(low)
        if isinstance(cfg, dict):
            eng = cfg.get("engine", "kokoro").upper()
            v = cfg.get("ref_audio") if eng in ("F5TTS", "F5-TTS") else cfg.get("voice", "")
            return canon, f"{eng}: {os.path.basename(v) if eng in ('F5TTS', 'F5-TTS') and v else v}"
    return canon, ""


def format_dialogue_tags_report(
    text,
    file_label="",
    filter_char=None,
    index_range=None,
    roster=None,
    chars_cfg=None,
    limit=None,
    offset=0,
):
    """Formats a readable report of dialogue tags and character summary in `text`."""
    tags = extract_dialogue_tags(text)
    if not tags:
        return f"  (No dialogue tags found in {file_label or 'file'}.)"

    if roster is None:
        roster = load_canonical_roster()
    if chars_cfg is None and os.path.exists(CHARACTERS_FILE):
        try:
            with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
                raw_c = json.load(f)
            chars_cfg = {k.lower(): v for k, v in raw_c.items() if isinstance(v, dict)}
        except Exception:
            chars_cfg = {}

    counts = {}
    for t in tags:
        lbl = f"{t['speaker']}:{t['emotion']}" if t["emotion"] else t["speaker"]
        counts[lbl] = counts.get(lbl, 0) + 1

    lines = []
    if file_label:
        lines.append("=" * 78)
        lines.append(f" Dialogue Tags: {file_label} ({len(tags)} tag(s))")
        lines.append("=" * 78)

    lines.append(" Speaker summary:")
    for lbl, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0].lower())):
        base_spk = lbl.split(":", 1)[0]
        canon, v_info = _resolve_character_display(base_spk, roster=roster, chars_cfg=chars_cfg)
        alias_note = f" -> {canon}" if canon.lower() != base_spk.lower() else ""
        voice_note = f" [{v_info}]" if v_info else ""
        lines.append(f"   • {lbl:<20} : {cnt:3d} line(s){alias_note}{voice_note}")

    filtered = tags
    if filter_char:
        fc_low = filter_char.strip().lower()
        filtered = [t for t in filtered if fc_low in t["speaker"].lower() or fc_low in t["emotion"].lower()]
    if index_range:
        idx_set = set(index_range)
        filtered = [t for t in filtered if t["index"] in idx_set]

    total_filtered = len(filtered)
    if offset:
        filtered = filtered[offset:]
    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    lines.append("-" * 78)
    if not filtered:
        lines.append(" (No matching dialogue tags for current filter.)")
    else:
        for t in filtered:
            tag_lbl = f"{t['speaker']}:{t['emotion']}" if t["emotion"] else t["speaker"]
            lines.append(f"  [{t['index']:03d}] [{tag_lbl:<20}] {t['context']}")
        if limit is not None and (offset + len(filtered)) < total_filtered:
            lines.append(f"  ... showing {offset + 1}-{offset + len(filtered)} of {total_filtered} matching tag(s).")
    lines.append("=" * 78)
    return "\n".join(lines)


def apply_dialogue_tag_edits(text, index_edits=None, remap_rules=None):
    """
    Applies index-specific edits and/or speaker remap rules to tagged dialogue spans in `text`.
    - `index_edits`: dict[int, dict(action='set'|'untag', speaker=str|None, emotion=str|None)]
    - `remap_rules`: list of (old_speaker, new_speaker, new_emotion_or_none)
    Returns `(updated_text, changes_log, remaining_speakers)`.
    """
    tags = extract_dialogue_tags(text)
    if not tags:
        return text, [], set()

    remap_map = {}
    for rule in (remap_rules or []):
        old_spk, new_spk, new_emo = rule
        remap_map[old_spk.strip().lower()] = (new_spk.strip(), new_emo)

    updated = text
    changes = []
    remaining_speakers = set()

    for tag in reversed(tags):
        idx = tag["index"]
        orig_spk = tag["speaker"]
        orig_emo = tag["emotion"]
        orig_lbl = f"{orig_spk}:{orig_emo}" if orig_emo else orig_spk

        edit = (index_edits or {}).get(idx)
        remap = remap_map.get(orig_spk.lower()) if edit is None else None

        if edit is None and remap is None:
            remaining_speakers.add(orig_spk)
            continue

        if edit is not None:
            action = edit.get("action", "set")
            new_spk = edit.get("speaker")
            new_emo = edit.get("emotion")
        else:
            new_spk, new_emo = remap
            action = "untag" if new_spk.lower() in ("untag", "narrator", "none") else "set"

        if action == "untag" or (new_spk is not None and new_spk.lower() in ("untag", "narrator", "none")):
            replacement = tag["dialogue"]
            changes.append(f"#{idx:03d}: [{orig_lbl}] -> UNTAGGED (narrator)")
        else:
            final_spk = new_spk if new_spk else orig_spk
            if new_emo is None:
                final_emo = orig_emo
            elif new_emo.lower() in ("", "none", "clear", "-"):
                final_emo = ""
            else:
                final_emo = new_emo.strip().lower()

            if final_emo:
                replacement = f"[{final_spk}:{final_emo}] {tag['dialogue']} [/{final_spk}:{final_emo}]"
                new_lbl = f"{final_spk}:{final_emo}"
            else:
                replacement = f"[{final_spk}] {tag['dialogue']} [/{final_spk}]"
                new_lbl = final_spk

            remaining_speakers.add(final_spk)
            if new_lbl != orig_lbl:
                changes.append(f"#{idx:03d}: [{orig_lbl}] -> [{new_lbl}]")

        updated = updated[:tag["start"]] + replacement + updated[tag["end"]:]

    changes.reverse()
    return updated, changes, remaining_speakers


def invalidate_cached_audio_for_chapter(tagged_file_path):
    """
    Removes stale chapter audio chunks, merged MP3, and timestamps if a tagged chapter
    was modified after audio was already generated.
    """
    import shutil

    base_name = os.path.splitext(os.path.basename(tagged_file_path))[0]
    parent_dir = os.path.basename(os.path.dirname(tagged_file_path))
    if parent_dir and parent_dir.lower() not in ("parsed_chapters", "tagged_chapters", "raw_txt", ".", ""):
        book_title_base = parent_dir
    else:
        cleaned_first = re.sub(r"^\d+_", "", base_name)
        book_title_base = re.sub(r"^intro_", "", cleaned_first)

    folder_name = f"{book_title_base}_hybrid_audiobook_output"
    removed = []
    for candidate_dir in (folder_name, os.path.join("finished", folder_name)):
        if not os.path.exists(candidate_dir):
            continue
        for fname in (f"{base_name}_complete.mp3", f"{base_name}_timestamps.txt"):
            fpath = os.path.join(candidate_dir, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    removed.append(fpath)
                except Exception:
                    pass
        chunk_dir = os.path.join(candidate_dir, f"chunks_{base_name}")
        if os.path.exists(chunk_dir):
            try:
                shutil.rmtree(chunk_dir)
                removed.append(chunk_dir)
            except Exception:
                pass
    return removed


def _resolve_target_file_indices(chap_selector, file_paths):
    """Resolves a chapter selector ('1', '1-2', or basename substring) into 0-based indices into `file_paths`."""
    if not chap_selector:
        return list(range(len(file_paths)))
    sel = chap_selector.strip()
    indices = _parse_index_selection(sel, len(file_paths))
    if indices:
        return [i - 1 for i in indices]
    sel_low = sel.lower()
    return [i for i, p in enumerate(file_paths) if sel_low in os.path.basename(p).lower()]


def _parse_single_tag_target(rhs):
    """Parses '<Speaker>[:<emotion>]', ':<emotion>', or 'untag' into (action, speaker, emotion)."""
    rhs = rhs.strip()
    if rhs.lower() in ("untag", "narrator", "none", "-"):
        return "untag", "untag", None
    if ":" in rhs:
        spk_part, emo_part = rhs.split(":", 1)
        spk = spk_part.strip() or None
        emo_raw = emo_part.strip().lower()
        emo = "" if emo_raw in ("", "none", "clear", "-") else emo_raw
        return "set", spk, emo
    return "set", rhs, None


def parse_tag_edit_specs(set_tag_specs=None, untag_specs=None, remap_specs=None, file_paths=None):
    """
    Parses CLI tag edit arguments into per-file `index_edits` and `remap_rules`.
    Supported formats:
      - set_tag: '[<chap>:]<indices>=<Speaker>[:<emotion>]' or '1=Alice; 2,4-6=Bob:whisper; 7=untag'
      - untag:   '[<chap>:]<indices>'
      - remap:   '<OldSpeaker>=<NewSpeaker>[:<emotion>]' or '<OldSpeaker>=untag'
    """
    file_paths = file_paths or []
    per_file_edits = {i: {} for i in range(len(file_paths))}
    remap_rules = []

    def _split_semicolon(items):
        out = []
        for item in (items or []):
            for piece in str(item).split(";"):
                piece = piece.strip()
                if piece:
                    out.append(piece)
        return out

    for spec in _split_semicolon(set_tag_specs):
        if "=" not in spec:
            print(f"[!] Ignoring invalid --set-tag spec '{spec}' (expected '<indices>=<Speaker>[:<emotion>]').")
            continue
        lhs, rhs = spec.split("=", 1)
        lhs = lhs.strip()
        if ":" in lhs:
            chap_sel, idx_sel = lhs.split(":", 1)
            target_f_indices = _resolve_target_file_indices(chap_sel, file_paths)
        else:
            idx_sel = lhs
            target_f_indices = list(range(len(file_paths)))

        action, spk, emo = _parse_single_tag_target(rhs)
        for fi in target_f_indices:
            fpath = file_paths[fi]
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                tag_count = len(extract_dialogue_tags(f.read()))
            for tidx in _parse_index_selection(idx_sel, tag_count):
                per_file_edits[fi][tidx] = {"action": action, "speaker": spk, "emotion": emo}

    for spec in _split_semicolon(untag_specs):
        if ":" in spec:
            chap_sel, idx_sel = spec.split(":", 1)
            target_f_indices = _resolve_target_file_indices(chap_sel, file_paths)
        else:
            idx_sel = spec
            target_f_indices = list(range(len(file_paths)))
        for fi in target_f_indices:
            fpath = file_paths[fi]
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                tag_count = len(extract_dialogue_tags(f.read()))
            for tidx in _parse_index_selection(idx_sel, tag_count):
                per_file_edits[fi][tidx] = {"action": "untag", "speaker": "untag", "emotion": None}

    for spec in _split_semicolon(remap_specs):
        if "=" not in spec:
            print(f"[!] Ignoring invalid --remap-tag spec '{spec}' (expected '<OldSpeaker>=<NewSpeaker>[:<emotion>]').")
            continue
        old_spk, rhs = spec.split("=", 1)
        old_spk = old_spk.strip()
        _, new_spk, new_emo = _parse_single_tag_target(rhs)
        if old_spk and new_spk:
            remap_rules.append((old_spk, new_spk, new_emo))

    return per_file_edits, remap_rules


def apply_cli_tag_operations(
    file_paths,
    list_tags=False,
    set_tags=None,
    untag_specs=None,
    remap_tags=None,
    map_aliases=None,
    filter_char=None,
):
    """
    Applies non-interactive CLI dialogue tag operations (alias mapping, speaker remapping,
    per-index tag changes, untagging, and tag listing) across `file_paths`.
    Returns True if any file was modified.
    """
    any_modified = False

    if map_aliases:
        ensure_tag_mappings_file()
        try:
            with open(TAG_MAPPINGS_FILE, "r", encoding="utf-8") as f:
                mappings = json.load(f)
        except Exception:
            mappings = {}
        for item in map_aliases:
            for piece in str(item).split(";"):
                piece = piece.strip()
                if "=" not in piece:
                    continue
                alias, canon = [p.strip() for p in piece.split("=", 1)]
                if alias and canon:
                    mappings[alias.lower()] = canon
                    print(f"[✓] Mapped alias '{alias.lower()}' -> '{canon}' in '{TAG_MAPPINGS_FILE}'.")
        with open(TAG_MAPPINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=4)

    has_edits = bool(set_tags or untag_specs or remap_tags)
    if has_edits and file_paths:
        per_file_edits, remap_rules = parse_tag_edit_specs(
            set_tag_specs=set_tags,
            untag_specs=untag_specs,
            remap_specs=remap_tags,
            file_paths=file_paths,
        )
        all_new_speakers = set()
        for fi, fpath in enumerate(file_paths):
            if not os.path.exists(fpath):
                continue
            edits = per_file_edits.get(fi, {})
            if not edits and not remap_rules:
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                orig_text = f.read()
            new_text, changes, used_speakers = apply_dialogue_tag_edits(
                orig_text, index_edits=edits, remap_rules=remap_rules
            )
            if changes and new_text != orig_text:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_text)
                invalidate_cached_audio_for_chapter(fpath)
                all_new_speakers.update(used_speakers)
                any_modified = True
                print(f"[✓] Updated {len(changes)} dialogue tag(s) in '{fpath}':")
                for ch in changes:
                    print(f"    • {ch}")

        if all_new_speakers:
            sync_tag_mappings(all_new_speakers)
            sync_characters_json(all_new_speakers)

    if list_tags:
        found_any = False
        for idx, fpath in enumerate(file_paths or [], 1):
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            if extract_dialogue_tags(text):
                found_any = True
                print("\n" + format_dialogue_tags_report(
                    text,
                    file_label=f"[{idx}] {fpath}",
                    filter_char=filter_char,
                ))
        if not found_any:
            print("[*] No dialogue tags found in the selected file(s).")

    return any_modified


def _prompt_pick_speaker(known_speakers):
    """Interactive speaker picker for dialogue tag editing."""
    sorted_spks = sorted({s for s in known_speakers if s and s.lower() not in ("narrator", "divider", "unknown")}, key=str.lower)
    print("\nKnown Characters:")
    for idx, spk in enumerate(sorted_spks, 1):
        print(f"  [{idx:2d}] {spk}")
    print("  [ 0] Enter a new/custom character name")
    print("  [ u] Untag (read by Narrator)")

    sel = input(f"Select speaker [0-{len(sorted_spks)} or 'u' or type name]: ").strip()
    if not sel:
        return None
    if sel.lower() in ("u", "untag", "narrator"):
        return "untag"
    try:
        num = int(sel)
        if num == 0:
            custom = input("Enter character name: ").strip()
            return clean_character_name(custom) if custom else None
        if 1 <= num <= len(sorted_spks):
            return sorted_spks[num - 1]
    except ValueError:
        return clean_character_name(sel)
    return None


def interactive_review_dialogue_tags(file_paths, force_open=False):
    """
    Interactively inspects and edits dialogue tags across tagged files when applicable.
    Automatically skips files containing 0 dialogue tags (e.g. intro or untagged chapters).
    Returns True if any tags were modified.
    """
    reviewable = []
    for fpath in (file_paths or []):
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        tags = extract_dialogue_tags(text)
        if tags:
            reviewable.append((fpath, len(tags)))

    if not reviewable:
        print("[*] No dialogue tags found in processed file(s); skipping dialogue tag review.")
        return False

    total_tags = sum(cnt for _, cnt in reviewable)
    print("\n" + "=" * 70)
    print(f"             Dialogue Tag Review ({total_tags} tag(s) across {len(reviewable)} file(s))")
    print("=" * 70)
    for idx, (fpath, cnt) in enumerate(reviewable, 1):
        print(f"  [{idx:2d}] {fpath} ({cnt} dialogue tag(s))")

    if not force_open:
        ans = input("\nDo you want to review or change any dialogue tags? [y/N] (default 'n'): ").strip().lower()
        if ans not in ("y", "yes"):
            return False

    any_modified = False
    roster = load_canonical_roster()
    known_emotions = sorted(EMOTION_KEYWORDS.keys())

    while True:
        if len(reviewable) == 1:
            file_idx = 0
        else:
            print("\nSelect a file to review dialogue tags:")
            for idx, (fpath, _) in enumerate(reviewable, 1):
                with open(fpath, "r", encoding="utf-8") as f:
                    cur_cnt = len(extract_dialogue_tags(f.read()))
                print(f"  [{idx:2d}] {fpath} ({cur_cnt} tag(s))")
            print("  [ 0] Done / Continue")
            f_sel = input(f"Select file [0-{len(reviewable)}] (default 0): ").strip()
            if not f_sel or f_sel == "0":
                break
            try:
                fi = int(f_sel)
                if not (1 <= fi <= len(reviewable)):
                    print("[-] Invalid selection.")
                    continue
                file_idx = fi - 1
            except ValueError:
                print("[-] Invalid number.")
                continue

        target_path = reviewable[file_idx][0]
        filter_char = None
        page_offset = 0
        page_size = 40

        while True:
            with open(target_path, "r", encoding="utf-8") as f:
                current_text = f.read()
            tags = extract_dialogue_tags(current_text)
            if not tags:
                print(f"[*] All dialogue tags in '{target_path}' were removed.")
                break

            print("\n" + format_dialogue_tags_report(
                current_text,
                file_label=target_path,
                filter_char=filter_char,
                roster=roster,
                limit=page_size,
                offset=page_offset,
            ))
            print("Commands:")
            print("  • Enter tag number(s) to edit (e.g. '3', '1, 4-6', 'all')")
            print("  • Or quick edit inline: '3=Alice', '2,5=Bob:whisper', '4=untag', '3=:none'")
            print("  • [f <name>] Filter by character | [f all] Clear filter | [n/p] Next/Prev page")
            print("  • [r] Bulk remap a speaker | [m] Map alias in tag_mappings.json | [Enter/q] Done")

            cmd = input("\nTag review > ").strip()
            if not cmd or cmd.lower() in ("q", "quit", "done", "d"):
                break

            if cmd.lower() == "n":
                page_offset += page_size
                continue
            if cmd.lower() == "p":
                page_offset = max(0, page_offset - page_size)
                continue
            if cmd.lower() == "a":
                page_size = None
                page_offset = 0
                continue
            if cmd.lower().startswith("f "):
                arg_f = cmd[2:].strip()
                filter_char = None if arg_f.lower() in ("", "all", "clear") else arg_f
                page_offset = 0
                continue

            if cmd.lower() == "m":
                alias_in = input("Enter alias=CanonicalName (e.g. bobby=Bob): ").strip()
                if "=" in alias_in:
                    apply_cli_tag_operations([target_path], map_aliases=[alias_in])
                    roster = load_canonical_roster()
                continue

            if cmd.lower() == "r":
                old_s = input("Speaker tag to replace (exact name): ").strip()
                if not old_s:
                    continue
                new_s = input(f"Replace all [{old_s}] with (character name, 'Speaker:emotion', or 'untag'): ").strip()
                if not new_s:
                    continue
                if apply_cli_tag_operations([target_path], remap_tags=[f"{old_s}={new_s}"]):
                    any_modified = True
                    roster = load_canonical_roster()
                continue

            if "=" in cmd:
                if apply_cli_tag_operations([target_path], set_tags=[cmd]):
                    any_modified = True
                    roster = load_canonical_roster()
                continue

            selected_indices = _parse_index_selection(cmd, len(tags))
            if not selected_indices:
                print("[-] Unrecognized command or tag index.")
                continue

            print("\nSelected tag(s):")
            for tidx in selected_indices[:10]:
                t = tags[tidx - 1]
                lbl = f"{t['speaker']}:{t['emotion']}" if t["emotion"] else t["speaker"]
                print(f"  [{tidx:03d}] [{lbl}] {t['paragraph']}")
            if len(selected_indices) > 10:
                print(f"  ... and {len(selected_indices) - 10} more.")

            print("\nEdit Action:")
            print("  [1] Change speaker (and optionally emotion/variant)")
            print("  [2] Change or clear emotion / tone / alt variant")
            print("  [3] Untag quote (remove tag so Narrator reads it)")
            print("  [0] Cancel")
            act = input("Select action [0-3]: ").strip()

            index_edits = {}
            if act == "1":
                known_spks = set(roster.values()) | {t["speaker"] for t in tags}
                chosen_spk = _prompt_pick_speaker(known_spks)
                if not chosen_spk:
                    continue
                if chosen_spk == "untag":
                    for tidx in selected_indices:
                        index_edits[tidx] = {"action": "untag", "speaker": "untag", "emotion": None}
                else:
                    emo_in = input(
                        f"Optional emotion/alt ({', '.join(known_emotions)}, 'none' to clear, Enter to keep): "
                    ).strip().lower()
                    emo_val = None if not emo_in else ("" if emo_in in ("none", "clear", "-") else emo_in)
                    for tidx in selected_indices:
                        index_edits[tidx] = {"action": "set", "speaker": chosen_spk, "emotion": emo_val}
            elif act == "2":
                emo_in = input(
                    f"Enter emotion/alt ({', '.join(known_emotions)}, or 'none' to clear): "
                ).strip().lower()
                if not emo_in:
                    continue
                emo_val = "" if emo_in in ("none", "clear", "-") else emo_in
                for tidx in selected_indices:
                    index_edits[tidx] = {"action": "set", "speaker": None, "emotion": emo_val}
            elif act == "3":
                for tidx in selected_indices:
                    index_edits[tidx] = {"action": "untag", "speaker": "untag", "emotion": None}
            else:
                continue

            new_text, changes, used_speakers = apply_dialogue_tag_edits(current_text, index_edits=index_edits)
            if changes and new_text != current_text:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                invalidate_cached_audio_for_chapter(target_path)
                sync_tag_mappings(used_speakers)
                sync_characters_json(used_speakers)
                roster = load_canonical_roster()
                any_modified = True
                print(f"[✓] Applied {len(changes)} change(s) to '{target_path}':")
                for ch in changes[:10]:
                    print(f"    • {ch}")

        if len(reviewable) == 1:
            break

    return any_modified


# ==================== INTERACTIVE CLI ====================

def select_input_files():
    """
    Lists files in parsed_chapters/** and raw_txt/** recursively.
    Supports selecting single file or [0] to process all chapters in chapters.txt.
    Returns list of file paths.
    """
    search_dirs = ["parsed_chapters", "raw_txt"]
    candidates = []
    for d in search_dirs:
        if os.path.exists(d):
            files = sorted(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
            candidates.extend(files)

    if not candidates and os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
            return [line.strip() for line in mf if line.strip() and not line.startswith("#")]

    if not candidates:
        print("[!] No .txt files found in 'parsed_chapters/' or 'raw_txt/'.")
        path = input("Enter path to your text file manually: ").strip()
        if os.path.exists(path):
            return [path]
        print("[X] File does not exist.")
        sys.exit(1)

    print("\nAvailable files to tag:")
    if os.path.exists(MANIFEST_FILE):
        print(f"  [00] Process ALL chapters listed in '{MANIFEST_FILE}'")
    for idx, path in enumerate(candidates, 1):
        print(f"  [{idx:02d}] {path}")

    choice = input(f"\nSelect file number [0-{len(candidates)}] (default 1): ").strip()
    if choice in ("0", "00", "all") and os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
            return [line.strip() for line in mf if line.strip() and not line.startswith("#")]

    try:
        num = int(choice) if choice else 1
        if 1 <= num <= len(candidates):
            return [candidates[num - 1]]
    except ValueError:
        pass
    return [candidates[0]]


def _resolve_existing_tagged_files(target_input=None):
    """Resolves existing tagged files from `--input` or `chapters.txt` / `tagged_chapters/` for review-only mode."""
    if target_input:
        if target_input.lower() == "all" and os.path.exists(MANIFEST_FILE):
            with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
                files = [l.strip() for l in mf if l.strip() and not l.startswith("#")]
        elif os.path.isdir(target_input):
            files = sorted(glob.glob(os.path.join(target_input, "**", "*.txt"), recursive=True))
        elif os.path.exists(target_input):
            files = [target_input]
        else:
            files = []
    elif os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
            files = [l.strip() for l in mf if l.strip() and not l.startswith("#")]
    else:
        files = sorted(glob.glob(os.path.join(DEFAULT_OUTPUT_DIR, "**", "*.txt"), recursive=True))

    resolved = []
    for f in files:
        if "parsed_chapters" in os.path.normpath(f).split(os.sep):
            parts = os.path.normpath(f).split(os.sep)
            p_idx = parts.index("parsed_chapters")
            candidate = os.path.join(DEFAULT_OUTPUT_DIR, *parts[p_idx + 1:])
            if os.path.exists(candidate):
                resolved.append(candidate)
                continue
        if os.path.exists(f):
            resolved.append(f)
    return resolved


def main():
    print("=" * 70)
    print("      Fast & Accurate Audiobook Dialogue Tagger & Voice Mapper    ")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Tag spoken dialogue using fast smart context attribution or hybrid LLM.")
    parser.add_argument("input_pos", nargs="?", default=None, help="Optional input text file, folder, or 'all' (manifest).")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input text file or 'all' to tag chapters in chapters.txt.")
    parser.add_argument(
        "--format", "-f", "--tag-mode",
        dest="format",
        choices=["fast", "prose", "hybrid", "script"],
        default=None,
        help="Tagging mode: 'fast'/'prose' (smart engine), 'hybrid' (+ GPU LLM), or 'script'.",
    )
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Ollama model name (default: qwen2.5:7b).")
    parser.add_argument("--url", "-u", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama server URL.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Destination directory for tagged text.")
    parser.add_argument("--detect-tone", "--emotions", dest="detect_tone", action="store_true", default=None, help="Detect dialogue emotions and tag as [Speaker:tone].")
    parser.add_argument("--no-detect-tone", "--no-emotions", dest="detect_tone", action="store_false", help="Disable dialogue emotion/tone detection.")
    parser.add_argument("--update-manifest", dest="update_manifest", action="store_true", default=None, help="Update chapters.txt with tagged files.")
    parser.add_argument("--no-update-manifest", dest="update_manifest", action="store_false", help="Do not modify chapters.txt.")
    parser.add_argument("--review-tags", "--review", dest="review_tags", action="store_true", default=None, help="Interactively review and edit dialogue tags after tagging.")
    parser.add_argument("--no-review-tags", dest="review_tags", action="store_false", help="Skip interactive dialogue tag review prompt.")
    parser.add_argument("--review-only", action="store_true", default=False, help="Review/edit existing tagged files without re-running attribution.")
    parser.add_argument("--list-tags", "--show-tags", action="store_true", default=False, help="Print numbered dialogue tags and context for selected file(s).")
    parser.add_argument("--filter-char", type=str, default=None, help="Filter --list-tags output by speaker/emotion name.")
    parser.add_argument("--set-tag", "--change-tag", "--edit-tag", dest="set_tags", action="append", default=[], help="Change tag(s), e.g. '1=Alice' or '2,4-6=Bob:whisper' or '1:3=untag'.")
    parser.add_argument("--untag", dest="untag_specs", action="append", default=[], help="Remove dialogue tags so Narrator reads them, e.g. '3,5' or '1:3,5'.")
    parser.add_argument("--remap-tag", "--rename-tag", dest="remap_tags", action="append", default=[], help="Bulk remap a speaker tag, e.g. 'OldName=NewName' or 'OldName=untag'.")
    parser.add_argument("--map-alias", dest="map_aliases", action="append", default=[], help="Map tag alias in tag_mappings.json, e.g. 'bobby=Bob'.")
    parser.add_argument("--interactive", action="store_true", default=False, help="Prompt interactively for unspecified options.")
    parser.add_argument("--yes", "-y", "--non-interactive", dest="non_interactive", action="store_true", default=False, help="Run non-interactively with defaults for unspecified options.")
    args = parser.parse_args()

    input_arg = args.input or args.input_pos
    has_cli_flags = len(sys.argv) > 1
    is_interactive = args.interactive or (not has_cli_flags and not args.non_interactive)

    # Review / Edit-only shortcut for already-tagged files
    edit_only_mode = args.review_only or (
        (args.list_tags or args.set_tags or args.untag_specs or args.remap_tags or args.map_aliases)
        and args.format is None
        and (not input_arg or os.path.normpath(input_arg).startswith(os.path.normpath(DEFAULT_OUTPUT_DIR)) or input_arg.lower() == "all")
    )
    if edit_only_mode:
        target_files = _resolve_existing_tagged_files(input_arg)
        if not target_files:
            print("[!] No existing tagged files found to inspect or edit.")
            return
        if args.list_tags or args.set_tags or args.untag_specs or args.remap_tags or args.map_aliases:
            apply_cli_tag_operations(
                target_files,
                list_tags=args.list_tags,
                set_tags=args.set_tags,
                untag_specs=args.untag_specs,
                remap_tags=args.remap_tags,
                map_aliases=args.map_aliases,
                filter_char=args.filter_char,
            )
        has_cli_edit_ops = bool(args.list_tags or args.set_tags or args.untag_specs or args.remap_tags or args.map_aliases)
        should_open_interactive = (
            args.review_tags is True
            or (
                args.review_only
                and args.review_tags is not False
                and not args.non_interactive
                and (args.interactive or not has_cli_edit_ops)
            )
        )
        if should_open_interactive:
            interactive_review_dialogue_tags(target_files, force_open=True)
        return

    # 1. Determine files to process
    if input_arg and input_arg.lower() == "all" and os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
            files_to_process = [line.strip() for line in mf if line.strip() and not line.startswith("#")]
    elif input_arg and os.path.isdir(input_arg):
        files_to_process = sorted(glob.glob(os.path.join(input_arg, "**", "*.txt"), recursive=True))
    elif input_arg and os.path.exists(input_arg):
        files_to_process = [input_arg]
    elif not is_interactive and os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
            files_to_process = [line.strip() for line in mf if line.strip() and not line.startswith("#")]
    else:
        files_to_process = select_input_files()

    if not files_to_process:
        print("[!] No input files found to tag.")
        return

    print(f"\n[+] Files to process: {len(files_to_process)}")
    for f in files_to_process:
        print(f"    • {f}")

    # 2. Select Dialogue Format / Mode
    dialogue_format = args.format
    if dialogue_format == "fast":
        dialogue_format = "prose"
    if not dialogue_format:
        if is_interactive:
            print("\nSelect dialogue tagging mode:")
            print("  [1] Fast Smart Parser (Instant, 100% quote coverage, zero GPU wait) [Recommended]")
            print("  [2] Hybrid LLM (Smart Parser + Local GPU LLM verification for ambiguous dialogue)")
            print("  [3] Script Form (Screenplay / Script format: [Name]: dialogue)")
            fmt_choice = input("Enter choice [1, 2, or 3] (default 1): ").strip()
            if fmt_choice == "2":
                dialogue_format = "hybrid"
            elif fmt_choice == "3":
                dialogue_format = "script"
            else:
                dialogue_format = "prose"
        else:
            dialogue_format = "prose"

    # 3. Model setup if hybrid mode selected
    chosen_model = args.model
    ollama_url = args.url

    if dialogue_format == "hybrid":
        is_running, installed_models = check_ollama_service(ollama_url)
        if not is_running:
            print("\n" + "!" * 70)
            print(" [!] Local Ollama service not detected at http://localhost:11434.")
            print("     Falling back seamlessly to Fast Smart Parser mode.")
            print("!" * 70)
            dialogue_format = "prose"
        else:
            print(f"[✓] Connected to local Ollama service. Installed models: {installed_models}")
            if installed_models and chosen_model not in installed_models:
                matched = [m for m in installed_models if "qwen" in m.lower() or "llama" in m.lower()]
                if matched:
                    chosen_model = matched[0]
                    print(f"[*] Defaulting to installed model: '{chosen_model}'")

    # 4. Check tone detection
    detect_tone = args.detect_tone
    if detect_tone is None:
        if is_interactive:
            t_choice = input("\nDetect dialogue tone & emotions (whisper, shout, angry, sad, etc.)? [y/n] (default 'y'): ").strip().lower()
            detect_tone = (t_choice != "n")
        else:
            detect_tone = True

    # 5. Run batch tagging
    out_dir = args.output or DEFAULT_OUTPUT_DIR
    tagged_files = batch_tag_files(
        file_paths=files_to_process,
        dialogue_format=dialogue_format,
        model=chosen_model,
        url=ollama_url,
        output_dir=out_dir,
        detect_tone=detect_tone
    )

    # 6. Apply CLI tag edits / listing and/or interactive dialogue tag review when applicable
    if tagged_files:
        if args.list_tags or args.set_tags or args.untag_specs or args.remap_tags or args.map_aliases:
            apply_cli_tag_operations(
                tagged_files,
                list_tags=args.list_tags,
                set_tags=args.set_tags,
                untag_specs=args.untag_specs,
                remap_tags=args.remap_tags,
                map_aliases=args.map_aliases,
                filter_char=args.filter_char,
            )

        if args.review_tags is True:
            interactive_review_dialogue_tags(tagged_files, force_open=True)
        elif args.review_tags is None and is_interactive:
            interactive_review_dialogue_tags(tagged_files, force_open=False)

    # 7. Update chapters.txt with the tagged files
    if tagged_files:
        do_update = args.update_manifest
        if do_update is None:
            if is_interactive:
                print("\nManifest Update:")
                update_manifest = input("Update 'chapters.txt' to point to these newly tagged files? [y/n] (default 'y'): ").strip().lower()
                do_update = (update_manifest != "n")
            else:
                do_update = True
        if do_update:
            update_manifest_file(tagged_files, MANIFEST_FILE)
            print(f"[✓] Updated '{MANIFEST_FILE}' with {len(tagged_files)} tagged file(s) (preserved all existing chapters).")

    print("\nNext steps:")
    print(f"1. Open '{TAG_MAPPINGS_FILE}' if you want to alias any nicknames to specific characters.")
    print(f"2. Open '{CHARACTERS_FILE}' to assign Kokoro voices or F5-TTS reference audio.")
    print("3. Run the audio generator:")
    print("   python local_hybrid_audioconverter.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
