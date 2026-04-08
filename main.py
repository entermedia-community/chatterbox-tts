import io
import json
import os
import re
import random
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL: Optional[ChatterboxTTS] = None


# ---------------------------------------------------------------------------
# Lifespan – model is loaded once on startup and released on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print(f"🚀 Running on device: {DEVICE}")
    print("Loading ChatterboxTTS model…")
    MODEL = ChatterboxTTS.from_pretrained(DEVICE)
    if hasattr(MODEL, "to") and str(MODEL.device) != DEVICE:
        MODEL.to(DEVICE)
    print(f"Model ready. Internal device: {getattr(MODEL, 'device', 'N/A')}")
    yield
    MODEL = None


app = FastAPI(title="ChatterboxTTS API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_model() -> ChatterboxTTS:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return MODEL


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _generate(
    text: str,
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    seed: int = 0,
    cfg_weight: float = 0.5,
) -> tuple[int, np.ndarray]:
    model = _require_model()
    if seed != 0:
        set_seed(seed)
    kwargs = {
        "exaggeration": exaggeration,
        "temperature": temperature,
        "cfg_weight": cfg_weight,
    }
    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path
    wav = model.generate(text[:300], **kwargs)
    return model.sr, wav.squeeze(0).numpy()


def _to_wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    """Encode a float32 numpy array as 16-bit PCM WAV bytes."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, pcm)
    buf.seek(0)
    return buf.read()


def _wav_response(sample_rate: int, audio: np.ndarray) -> StreamingResponse:
    data = _to_wav_bytes(sample_rate, audio)
    return StreamingResponse(io.BytesIO(data), media_type="audio/wav")


def _chunk_text(text: str, max_chars: int = 280) -> list[str]:
    """Split text at sentence boundaries so each chunk <= max_chars."""
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _build_conversation_audio(
    segments: list[dict],
    speaker_audio_prompts: dict,
    speaker_settings: dict,
    silence_between_turns_ms: int,
) -> tuple[int, np.ndarray]:
    sample_rate: Optional[int] = None
    all_audio: list[np.ndarray] = []

    for idx, segment in enumerate(segments):
        speaker = segment["speaker"]
        text = segment["content"]
        settings = speaker_settings.get(speaker, {})
        audio_prompt = speaker_audio_prompts.get(speaker)

        preview = text[:70] + ("..." if len(text) > 70 else "")
        print(f"[{idx + 1}/{len(segments)}] {speaker}: {preview}")

        for chunk in _chunk_text(text):
            sr, audio = _generate(
                text=chunk,
                audio_prompt_path=audio_prompt,
                exaggeration=settings.get("exaggeration", 0.5),
                temperature=settings.get("temperature", 0.8),
                seed=settings.get("seed", 0),
                cfg_weight=settings.get("cfg_weight", 0.5),
            )
            if sample_rate is None:
                sample_rate = sr
            all_audio.append(audio.astype(np.float32))

        silence = np.zeros(
            int(sample_rate * silence_between_turns_ms / 1000), dtype=np.float32
        )
        all_audio.append(silence)

    return sample_rate, np.concatenate(all_audio)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["utility"])
def health():
    """Return model readiness status."""
    return {"status": "ok", "model_loaded": MODEL is not None, "device": DEVICE}


@app.post("/tts", tags=["tts"], response_class=StreamingResponse)
async def tts(
    text: str = Form(..., description="Text to synthesise (max 300 chars)."),
    exaggeration: float = Form(0.5, description="Expressiveness 0.25-2.0."),
    temperature: float = Form(0.8, description="Sampling temperature 0.05-5.0."),
    seed: int = Form(0, description="Random seed; 0 = random."),
    cfg_weight: float = Form(0.5, description="CFG/pace weight 0.2-1.0."),
    audio_prompt: Optional[UploadFile] = File(None, description="Optional reference voice WAV."),
):
    """Generate speech for a single piece of text and return a WAV file."""
    _require_model()

    prompt_path: Optional[str] = None
    tmp_file = None

    try:
        if audio_prompt is not None:
            data = await audio_prompt.read()
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_file.write(data)
            tmp_file.flush()
            tmp_file.close()
            prompt_path = tmp_file.name

        sr, audio = await run_in_threadpool(
            _generate, text, prompt_path, exaggeration, temperature, seed, cfg_weight
        )
    finally:
        if tmp_file is not None:
            os.unlink(tmp_file.name)

    return _wav_response(sr, audio)

@app.post("/conversation", tags=["tts"], response_class=StreamingResponse)
async def conversation(
    transcript_file: UploadFile = File(..., description="Transcript .json file."),
    speaker_audio_1: UploadFile = File(..., description="Voice WAV for the first speaker."),
    speaker_audio_2: UploadFile = File(..., description="Voice WAV for the second speaker."),
    silence_ms: int = Form(600, description="Silence between turns in milliseconds."),
):
    """Generate a full conversation from a JSON transcript and two speaker voice files.

    The JSON must contain a 'segments' list (each with 'speaker' and 'content') and
    an 'audio_prompt' list that defines speaker order (e.g. [{"speaker": "Alice", ...},
    {"speaker": "Bob", ...}]). speaker_audio_1 maps to the first speaker and
    speaker_audio_2 to the second.
    """
    _require_model()

    raw = await transcript_file.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}")

    segments = data.get("segments")
    if not segments:
        raise HTTPException(status_code=422, detail="No 'segments' found in transcript JSON.")

    # Determine speaker order from the JSON audio_prompt list, falling back to segment order
    audio_prompt_list = data.get("audio_prompt", [])
    speakers_in_order = [entry["speaker"] for entry in audio_prompt_list]
    if not speakers_in_order:
        speakers_in_order = list(dict.fromkeys(s["speaker"] for s in segments))

    if len(speakers_in_order) < 2:
        raise HTTPException(status_code=422, detail="At least two speakers are required.")

    tmp_files: list[str] = []
    try:
        speaker_audio_prompts: dict[str, str] = {}
        for speaker, upload in zip(speakers_in_order[:2], [speaker_audio_1, speaker_audio_2]):
            audio_data = await upload.read()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(audio_data)
            tmp.flush()
            tmp.close()
            tmp_files.append(tmp.name)
            speaker_audio_prompts[speaker] = tmp.name

        _voice_defaults = [
            {"exaggeration": 0.5,  "temperature": 0.70, "cfg_weight": 0.5},
            {"exaggeration": 0.65, "temperature": 0.85, "cfg_weight": 0.6},
        ]
        speaker_settings = {
            spk: _voice_defaults[i % len(_voice_defaults)]
            for i, spk in enumerate(speakers_in_order)
        }

        print(f"Speakers: {', '.join(speakers_in_order)} | Turns: {len(segments)}")

        sr, audio = await run_in_threadpool(
            _build_conversation_audio,
            segments,
            speaker_audio_prompts,
            speaker_settings,
            silence_ms,
        )
    finally:
        for path in tmp_files:
            os.unlink(path)

    return _wav_response(sr, audio)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
