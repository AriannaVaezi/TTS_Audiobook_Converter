import os
import sys
import json
import time

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import local_hybrid_audioconverter as lha


def main():
    print("=" * 75)
    print("      A/B Comparison Test: Dialogue Tone & Emotion ON vs OFF        ")
    print("=" * 75)

    # 1. Load local characters
    characters_map = lha.load_characters()
    tag_mappings = lha.load_tag_mappings()

    # Find one active F5-TTS character and one active Kokoro character
    f5_char_name = None
    kokoro_char_name = None

    # Check for Arianna first if present
    for target_name in ["Arianna", "arianna"]:
        if target_name in characters_map and os.path.exists(characters_map[target_name].get("ref_audio", "")):
            f5_char_name = target_name
            break

    for name, cfg in characters_map.items():
        engine = cfg.get("engine", "").lower()
        if engine in ["f5tts", "f5-tts"] and os.path.exists(cfg.get("ref_audio", "")):
            if not f5_char_name:
                f5_char_name = name
        elif engine == "kokoro" and name.lower() not in ["narrator"]:
            if not kokoro_char_name:
                kokoro_char_name = name

    if not f5_char_name:
        print("[!] No valid F5-TTS character with existing reference audio found in characters.json.")
        return

    if not kokoro_char_name:
        kokoro_char_name = "narrator"

    disp_f5 = characters_map[f5_char_name].get("display_name", f5_char_name.title())
    disp_kokoro = characters_map[kokoro_char_name].get("display_name", kokoro_char_name.title())

    print(f"[*] Test Voice Setup:")
    print(f"    • F5-TTS (GPU) : [{disp_f5}] (ref: {characters_map[f5_char_name].get('ref_audio')})")
    print(f"    • Kokoro (CPU) : [{disp_kokoro}] (voice: {characters_map[kokoro_char_name].get('voice')})")
    print(f"    • Narrator     : [Narrator] (voice: {characters_map.get('narrator', {}).get('voice', 'af_bella')})")
    print("-" * 75)

    output_dir = "comparison_test_output"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Prepare test scripts
    test_text = (
        f"The evening mist drifted silently over the cobblestones.\n\n"
        f"[{f5_char_name}:whisper] \"Did you hear that? Keep your voice down, someone is out there.\" [/{f5_char_name}]\n\n"
        f"The shadows deepened along the alleyway.\n\n"
        f"[{kokoro_char_name}:shout] \"Identify yourself! We will not ask twice!\" [/{kokoro_char_name}]\n\n"
        f"[{f5_char_name}:calm] \"Calm down, there is no need to shout. We are all on the same side.\" [/{f5_char_name}]"
    )

    path_with_emo = os.path.join(output_dir, "sample_WITH_emotion.txt")
    path_without_emo = os.path.join(output_dir, "sample_WITHOUT_emotion.txt")

    with open(path_with_emo, "w", encoding="utf-8") as f:
        f.write(test_text)
    with open(path_without_emo, "w", encoding="utf-8") as f:
        f.write(test_text)

    # 3. Clean any existing chunks from previous test runs to ensure fresh generation
    for base in ["sample_WITH_emotion", "sample_WITHOUT_emotion"]:
        lha.clear_chapter_recording(output_dir, base)

    # 4. Run PASS 1: WITH Emotion
    print("\n" + "=" * 75)
    print(">>> PASS 1: Rendering WITH Tone & Emotion (Speed & Pitch Modulation ON)")
    print("=" * 75)
    t0_with = time.time()
    success_with = lha.process_hybrid_chapter_simultaneous(
        txt_filename=path_with_emo,
        characters_map=characters_map,
        master_output_dir=output_dir,
        tag_mappings=tag_mappings,
        enable_emotions=True
    )
    t_with = time.time() - t0_with

    # 5. Run PASS 2: WITHOUT Emotion
    print("\n" + "=" * 75)
    print(">>> PASS 2: Rendering WITHOUT Tone & Emotion (Pure Baseline Speeds & Pitches)")
    print("=" * 75)
    t0_no = time.time()
    success_no = lha.process_hybrid_chapter_simultaneous(
        txt_filename=path_without_emo,
        characters_map=characters_map,
        master_output_dir=output_dir,
        tag_mappings=tag_mappings,
        enable_emotions=False
    )
    t_no = time.time() - t0_no

    if not success_with or not success_no:
        print("[!] One or both rendering passes encountered an error.")
        return

    # 6. Gather and display detailed comparison
    mp3_with = os.path.join(output_dir, "sample_WITH_emotion_complete.mp3")
    mp3_no = os.path.join(output_dir, "sample_WITHOUT_emotion_complete.mp3")

    dur_with = lha.get_wav_duration(mp3_with) if os.path.exists(mp3_with) else 0.0
    dur_no = lha.get_wav_duration(mp3_no) if os.path.exists(mp3_no) else 0.0

    print("\n" + "=" * 75)
    print("                        COMPARISON RESULTS")
    print("=" * 75)
    print(f"  • WITH Emotion MP3    : {mp3_with}")
    print(f"      Duration          : {dur_with:.2f}s  (Render time: {t_with:.2f}s)")
    print(f"  • WITHOUT Emotion MP3 : {mp3_no}")
    print(f"      Duration          : {dur_no:.2f}s  (Render time: {t_no:.2f}s)")
    print("-" * 75)

    # Compare chunk by chunk
    dir_with = os.path.join(output_dir, "chunks_sample_WITH_emotion")
    dir_no = os.path.join(output_dir, "chunks_sample_WITHOUT_emotion")

    chunks_with = sorted([f for f in os.listdir(dir_with) if f.startswith("chunk_") and f.endswith(".wav")])

    print(f"{'Chunk':<8} | {'Speaker':<14} | {'Tone':<10} | {'WITH Emotion':<14} | {'WITHOUT Emotion':<16} | {'Delta':<8}")
    print("-" * 75)

    tone_list = ["neutral", "whisper", "neutral", "shout", "calm"]

    for idx, c_file in enumerate(chunks_with):
        p_w = os.path.join(dir_with, c_file)
        p_n = os.path.join(dir_no, c_file)
        d_w = lha.get_wav_duration(p_w) if os.path.exists(p_w) else 0.0
        d_n = lha.get_wav_duration(p_n) if os.path.exists(p_n) else 0.0
        diff = d_w - d_n
        tone_label = tone_list[idx] if idx < len(tone_list) else ""
        diff_str = f"{diff:+.2f}s" if abs(diff) > 0.01 else "Identical"
        print(f"Chunk {idx+1:02d} | {'Voice Chunk':<14} | {tone_label:<10} | {d_w:6.2f}s        | {d_n:6.2f}s          | {diff_str}")

    print("=" * 75)
    print("[OK] Test comparison complete! Both audio files are ready for listening.")


if __name__ == "__main__":
    main()
