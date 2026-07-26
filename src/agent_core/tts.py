"""Sesli çıktı: nihai cevabı Chatterbox TTS ile seslendirir.

Mimari (neden HTTP?): Chatterbox `numpy<2` dayatıyor; ana venv'de ise scipy
numpy 2.x'e göre derli — ikisi tek venv'de barışmıyor. Bu yüzden Chatterbox ayrı
bir venv'de (.venv-voice) ayrı bir SÜREÇ olarak koşar (bkz. voice_service.py) ve bu
modül ona HTTP ile konuşur. Böylece ağır/çakışan TTS bağımlılıkları ana agent
sürecinden tamamen izole kalır. Sesli mod kapalıyken buraya hiç girilmez.

Uzun metinlerde TTS bozulabildiği (ve Chatterbox tek seferde ~bir paragraf
kaldırdığı) için çıktı cümle sınırlarında parçalanır: parça başına en çok
``CHUNK_MAX_TOKENS`` token, en çok ``MAX_CHUNKS`` parça. Tavanı aşan artık metin
SESLENDİRİLMEZ (metin arayüzde tam görünür). Parçalar sırayla numaralanır ki arayüz
birini bitirince diğerini çalabilsin (aşamalı oynatma).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

# Chatterbox mikroservisinin adresi (voice_service.py). run_api.py bunu otomatik
# başlatabilir; ayrı da başlatılabilir.
VOICE_SERVICE_URL = os.getenv("VOICE_SERVICE_URL", "http://127.0.0.1:8756")
# Sentez uzun sürebilir (Chatterbox + birden çok parça); "ses ile birlikte" modda
# cevap zaten bunu bekliyor.
_HTTP_TIMEOUT = int(os.getenv("VOICE_HTTP_TIMEOUT", "600"))

# --- Parçalama ayarları (env ile override edilebilir) ------------------------
# Chatterbox tek generate çağrısında ~bir paragraf/birkaç cümle kaldırır; çok
# büyük değerler burada iş görmez. Parça başına makul bir token bütçesi.
CHUNK_MAX_TOKENS = int(os.getenv("TTS_CHUNK_TOKENS", "120"))
# Toplam parça tavanı (aşamalı oynatma + patolojik uzunluklara karşı güvenlik).
MAX_CHUNKS = int(os.getenv("TTS_MAX_CHUNKS", "24"))


def _count_tokens(text: str) -> int:
    """Parça sınırı için token sayısı. tiktoken yoksa kaba tahmine düşer."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)  # ~4 karakter/token


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    max_chunks: int = MAX_CHUNKS,
) -> List[str]:
    """Metni cümle sınırlarında ≤``max_tokens``'lık parçalara böler.

    En çok ``max_chunks`` parça döner; tavanı aşan artık metin dışarıda bırakılır.
    Tek bir cümle tek başına sınırı aşıyorsa yine tek parça olur.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]

    chunks: List[str] = []
    cur = ""
    for s in sentences:
        candidate = (cur + " " + s).strip() if cur else s
        if cur and _count_tokens(candidate) > max_tokens:
            chunks.append(cur)
            if len(chunks) >= max_chunks:
                return chunks
            cur = s
        else:
            cur = candidate
    if cur and len(chunks) < max_chunks:
        chunks.append(cur)
    return chunks


def service_ready() -> bool:
    """Chatterbox servisi ayakta mı (hızlı sağlık kontrolü)."""
    try:
        with urllib.request.urlopen(VOICE_SERVICE_URL + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def synthesize(text: str, out_dir: Path, run_id: str) -> List[str]:
    """Metni parçalayıp Chatterbox servisine gönderir; üretilen wav yollarını döner.

    Parçalar ``<run_id>_00.wav`` ... diye SIRAYLA yazılır (aşamalı oynatma). Servis
    kapalı/erişilemezse istisna fırlatır — çağıran taraf (api.py) bunu yakalayıp
    metin cevabını sesli olmadan döndürür.
    """
    chunks = chunk_text(text)
    if not chunks:
        return []

    payload = json.dumps(
        {"chunks": chunks, "out_dir": str(out_dir), "run_id": run_id}
    ).encode("utf-8")
    req = urllib.request.Request(
        VOICE_SERVICE_URL + "/tts",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
        data = json.loads(r.read())
    return data.get("paths", [])
