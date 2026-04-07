import re
import random
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None

def get_or_load_model():
    """Loads the ChatterboxTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using ChatterboxTTS model with optional reference audio styling.
    This tool synthesizes natural-sounding speech from input text. When a reference audio file
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.
    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5.
    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")

    # Handle optional audio prompt
    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }

    if audio_prompt_path_input:
        generate_kwargs["audio_prompt_path"] = audio_prompt_path_input

    wav = current_model.generate(
        text_input[:300],  # Truncate text to max chars
        **generate_kwargs
    )
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())


def parse_transcript(filepath: str) -> list[dict]:
    """Parse a transcript file into a list of {speaker, text} dicts.

    Lines matching 'Host: ...' or 'Guest: ...' (metadata) are skipped.
    Multi-line blocks belonging to the same speaker turn are joined.
    """
    metadata_keys = {"Host", "Guest"}
    segments = []
    with open(filepath, "r") as f:
        content = f.read()

    # Match "SpeakerName: text" blocks (text can span multiple lines)
    pattern = re.compile(
        r'^([A-Z][A-Za-z]*):\s+(.+?)(?=^[A-Z][A-Za-z]*:\s|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(content):
        speaker = match.group(1).strip()
        text = " ".join(match.group(2).strip().split())  # normalise whitespace
        if speaker not in metadata_keys and text:
            segments.append({"speaker": speaker, "text": text})
    return segments


def chunk_text(text: str, max_chars: int = 280) -> list[str]:
    """Split text into chunks of at most max_chars, breaking at sentence endings."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            if current:
                chunks.append(current)
            # Hard-split sentences that are still too long
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def generate_conversation(
    transcript_path: str = "transcript.txt",
    output_path: str = "conversation_output.wav",
    speaker_audio_prompts: dict = None,
    speaker_settings: dict = None,
    silence_between_turns_ms: int = 600,
) -> str:
    """Generate a full conversation WAV from a two-person transcript file.

    Each speaker turn is synthesised in order with (optionally) different
    voice settings per speaker, then all segments are concatenated with
    short silences between turns and written to output_path.

    Args:
        transcript_path: Path to the transcript text file.
        output_path: Destination WAV file path.
        speaker_audio_prompts: Optional dict mapping speaker name to a
            reference audio file path for voice cloning, e.g.
            {"Crystal": "crystal.wav", "Christopher": "chris.wav"}.
        speaker_settings: Optional dict mapping speaker name to generation
            kwargs (exaggeration, temperature, cfgw).  Defaults are applied
            when omitted so the two speakers sound distinguishable.
        silence_between_turns_ms: Milliseconds of silence inserted between
            consecutive speaker turns.

    Returns:
        The path to the saved WAV file.
    """
    segments = parse_transcript(transcript_path)
    if not segments:
        raise ValueError(f"No dialogue segments found in '{transcript_path}'.")

    # Preserve insertion order so speaker index is deterministic
    speakers = list(dict.fromkeys(s["speaker"] for s in segments))
    print(f"Speakers detected: {', '.join(speakers)}")

    # Default per-speaker settings – slightly differentiate the two voices
    _defaults = [
        {"exaggeration": 0.5,  "temperature": 0.70, "cfgw": 0.5},
        {"exaggeration": 0.65, "temperature": 0.85, "cfgw": 0.6},
    ]
    if speaker_settings is None:
        speaker_settings = {
            spk: _defaults[i % len(_defaults)]
            for i, spk in enumerate(speakers)
        }

    if speaker_audio_prompts is None:
        speaker_audio_prompts = {}

    sample_rate: int | None = None
    all_audio: list[np.ndarray] = []

    for idx, segment in enumerate(segments):
        speaker = segment["speaker"]
        text = segment["text"]
        settings = speaker_settings.get(speaker, {})
        audio_prompt = speaker_audio_prompts.get(speaker)

        preview = text[:70] + ("..." if len(text) > 70 else "")
        print(f"\n[{idx + 1}/{len(segments)}] {speaker}: {preview}")

        for chunk in chunk_text(text):
            sr, audio = generate_tts_audio(
                text_input=chunk,
                audio_prompt_path_input=audio_prompt,
                exaggeration_input=settings.get("exaggeration", 0.5),
                temperature_input=settings.get("temperature", 0.8),
                cfgw_input=settings.get("cfgw", 0.5),
            )
            if sample_rate is None:
                sample_rate = sr
            all_audio.append(audio.astype(np.float32))

        # Silence gap after each turn
        silence = np.zeros(int(sample_rate * silence_between_turns_ms / 1000), dtype=np.float32)
        all_audio.append(silence)

    conversation_audio = np.concatenate(all_audio)

    # Normalise to avoid clipping
    peak = np.abs(conversation_audio).max()
    if peak > 0:
        conversation_audio = conversation_audio / peak * 0.9

    audio_int16 = (conversation_audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)

    duration = len(conversation_audio) / sample_rate
    print(f"\n✅ Conversation saved to: {output_path}")
    print(f"   Speakers : {', '.join(speakers)}")
    print(f"   Turns    : {len(segments)}")
    print(f"   Duration : {duration:.1f}s")
    return output_path


if __name__ == "__main__":
    # Optionally supply per-speaker reference audio for voice cloning:
    # speaker_audio_prompts = {
    #     "Crystal":     "reference_crystal.wav",
    #     "Christopher": "reference_christopher.wav",
    # }
    generate_conversation(
        transcript_path="transcript.txt",
        output_path="conversation_output.wav",
        # speaker_audio_prompts=speaker_audio_prompts,
    )
