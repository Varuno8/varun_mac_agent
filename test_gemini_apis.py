"""
Test all 3 Gemini APIs using the official google-genai SDK + google-auth.
Auth: service account credentials from .env (email + private key + project_id).
No manual JWT — google.oauth2.service_account.Credentials handles it.
"""

import json
import os
import re
import wave
from pathlib import Path

from google import genai
from google.genai import types
from google.oauth2 import service_account


def load_env(path: str = ".env") -> dict:
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        # Strip surrounding quotes, expand literal \n (same as Go's loadEnvFile)
        val = val.strip().strip('"').strip("'")
        val = val.replace("\\n", "\n").replace("\\r", "\r")
        env[key.strip()] = val
    return env


def clean_pem_key(raw: str) -> str:
    """Strip non-base64 chars from PEM body lines.
    The .env key has stray \\y \\K sequences (corrupted line breaks) that
    Go's PEM parser silently ignores but Python's cryptography library rejects.
    This reproduces Go's permissive behaviour."""
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        if line.startswith("-----"):
            cleaned.append(line)
        else:
            cleaned.append(re.sub(r"[^A-Za-z0-9+/=]", "", line))
    return "\n".join(cleaned)


def build_credentials(env: dict) -> service_account.Credentials:
    sa_info = {
        "type": "service_account",
        "client_email": env["GOOGLE_CLIENT_EMAIL"],
        "private_key": clean_pem_key(env["GOOGLE_PRIVATE_KEY"]),
        "private_key_id": env.get("GOOGLE_PRIVATE_KEY_ID", ""),
        "project_id": env["GOOGLE_PROJECT_ID"],
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    return service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def save_wav(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


# ── Test 1: Text → Text ───────────────────────────────────────────────────────

def test_text_to_text(client: genai.Client, model: str) -> bool:
    print(f"\n[TEST 1] Text → Text  model={model}")
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: GEMINI_TEXT_OK",
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )
    text = response.text
    print(f"  Response: {text!r}")
    ok = "GEMINI_TEXT_OK" in text
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ── Test 2: Text → Speech ─────────────────────────────────────────────────────

def test_text_to_speech(client: genai.Client, model: str) -> bool:
    print(f"\n[TEST 2] Text → Speech  model={model}")
    response = client.models.generate_content(
        model=model,
        contents="Say cheerfully: Have a wonderful day!",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
        ),
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data
    out_path = "/tmp/clicky_tts_test.wav"
    save_wav(out_path, pcm)
    size = Path(out_path).stat().st_size
    print(f"  Saved {size} bytes → {out_path}")
    ok = size > 1000
    print(f"  {'PASS' if ok else 'FAIL (file too small)'}")
    return ok


# ── Test 3: Audio → Text ──────────────────────────────────────────────────────

def test_audio_to_text(client: genai.Client, model: str) -> bool:
    print(f"\n[TEST 3] Audio → Text  model={model}")
    import io, math, struct

    # Generate a tiny 1-second 440 Hz sine WAV in memory
    rate, duration = 16000, 1
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(rate * duration)]
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate); wf.writeframes(pcm)
    audio_bytes = buf.getvalue()

    response = client.models.generate_content(
        model=model,
        contents=[
            "Describe this audio clip briefly.",
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )
    text = response.text
    print(f"  Response: {text!r}")
    ok = len(text.strip()) > 0
    print(f"  {'PASS' if ok else 'FAIL (empty response)'}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = load_env(str(Path(__file__).parent / ".env"))

    project_id = env["GOOGLE_PROJECT_ID"]
    model_text = env.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    model_tts  = env.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
    model_stt  = env.get("GEMINI_STT_MODEL", "gemini-3.1-flash-lite-preview")

    print("Building credentials from service account...")
    credentials = build_credentials(env)

    print(f"Creating Vertex AI client  project={project_id}")
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location="global",
        credentials=credentials,
    )

    results = {}
    for name, fn, model in [
        ("text_to_text",   test_text_to_text,   model_text),
        ("text_to_speech", test_text_to_speech, model_tts),
        ("audio_to_text",  test_audio_to_text,  model_stt),
    ]:
        try:
            results[name] = fn(client, model)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 50)
    print("ALL PASS" if all_pass else "SOME TESTS FAILED")
