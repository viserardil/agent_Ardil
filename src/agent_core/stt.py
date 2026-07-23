"""Ses tanıma (STT) — kullanıcının sesini metne çevirir.

Groq'un OpenAI-uyumlu transkripsiyon endpoint'i kullanılır
(``/openai/v1/audio/transcriptions``): çok hızlı ve Whisper modellerini barındırıyor.
LLM config'inden (config.py) AYRI tutulur — farklı sağlayıcı, farklı anahtar
(GROQ_API_KEY) ve tamamen farklı bir modalite (ses) olduğu için birbirine karışmasın.

Ayarlar .env'den okunur:
  - GROQ_API_KEY  : zorunlu.
  - STT_MODEL     : varsayılan 'whisper-large-v3-turbo'. Groq'ta model id'si prefix'siz;
                    sağlayıcı farklıysa (ör. OpenAI) 'whisper-1' gibi bir değere çekilir.
  - STT_BASE_URL  : transkripsiyon endpoint'i; varsayılan Groq.
  - STT_LANGUAGE  : ör. 'tr'. Boş bırakılırsa Whisper dili kendi tespit eder.
"""

from __future__ import annotations

import os

import requests

_DEFAULT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_DEFAULT_MODEL = "whisper-large-v3-turbo"


def _settings() -> dict:
    return {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": (os.getenv("STT_MODEL") or _DEFAULT_MODEL).strip(),
        "url": (os.getenv("STT_BASE_URL") or _DEFAULT_URL).strip(),
        "language": (os.getenv("STT_LANGUAGE") or "").strip() or None,
    }


def transcribe(
    audio_bytes: bytes,
    *,
    filename: str = "audio.webm",
    content_type: str | None = None,
    language: str | None = None,
    model: str | None = None,
) -> str:
    """Ses baytlarını metne çevirir; boş/başarısızsa RuntimeError yükseltir.

    filename ve content_type Whisper'ın formatı doğru çözmesi için önemlidir
    (tarayıcı MediaRecorder genelde webm/ogg/mp4 üretir).
    """
    settings = _settings()
    if not settings["api_key"]:
        raise RuntimeError("GROQ_API_KEY tanımlı değil (.env).")

    file_tuple = (
        filename,
        audio_bytes,
        content_type or "application/octet-stream",
    )
    data = {
        "model": model or settings["model"],
        "response_format": "json",
        "temperature": "0",
    }
    lang = language or settings["language"]
    if lang:
        data["language"] = lang

    response = requests.post(
        settings["url"],
        headers={"Authorization": f"Bearer {settings['api_key']}"},
        files={"file": file_tuple},
        data=data,
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Groq STT {response.status_code}: {response.text[:300]}")

    return (response.json().get("text") or "").strip()
