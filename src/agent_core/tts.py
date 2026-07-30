"""Sesli çıktı: nihai cevabı Coqui XTTS-v2 ile seslendirir (SÜREÇ İÇİ).

XTTS ana venv'le uyumlu (numpy 2.x) olduğundan ayrı venv/süreç GEREKMEZ; model bu
süreçte TEMBEL yüklenir (ilk sesli istekte, sonra bellekte sıcak kalır). Sesli mod
kapalıyken torch/TTS hiç import edilmez — hiçbir maliyeti olmaz.

Uzun metin cümle sınırında parçalanır (parça başına ≤``CHUNK_MAX_CHARS`` karakter,
en çok ``MAX_CHUNKS`` parça); parçalar SIRAYLA numaralanır ki arayüz aşamalı
oynatabilsin. Parça-sınırı klik/sessizliği için hafif kırpma + fade uygulanır.

NEDEN KARAKTER (token değil)? XTTS'in Türkçe için KENDİ iç sınırı 226 karakterdir
(TTS/tts/layers/xtts/tokenizer.py: char_limits["tr"]). Bunu aşan girdilerde XTTS
kendi içinde (spaCy ile, daha az kontrollü) yeniden bölüyor ve uyarı basıyor
("this might cause truncated audio") — cümleler arası tempo/prozodi SAPMASININ
asıl sebebi buydu. tiktoken tabanlı token sayımı İngilizce odaklı olduğu için
Türkçe karaktere güvenilir eşlenmiyor; bu yüzden ham karakter uzunluğuna geçildi.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from agent_core.tr_normalize import normalize as _normalize_tr

# XTTS-v2 model lisans onayını otomatikle (aksi halde interaktif prompt sunucuyu bloklar).
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# XTTS'i tamamen kapatma anahtarı: STT (Groq, cloud/VRAM'siz) aktif kalırken TTS'in
# (yerel, ~2GB VRAM) hiç yüklenmemesini garantiler — ör. 6GB kartta başka bir GPU
# modeliyle (Qwen ASR gibi) çakışmasın diye.
_TTS_DISABLED = os.getenv("TTS_DISABLED", "0").strip().lower() in ("1", "true", "yes")

# --- Parçalama ayarları (env ile override edilebilir) ------------------------
# XTTS'in Türkçe iç sınırı 226 karakter (tokenizer.py: char_limits["tr"]); 200
# güvenlik payı bırakır. Dil değişirse (en=250, de=253, ...) elle ayarlanmalı.
CHUNK_MAX_CHARS = int(os.getenv("TTS_CHUNK_CHARS", "200"))
MAX_CHUNKS = int(os.getenv("TTS_MAX_CHUNKS", "24"))

# --- Model / ses ayarları ----------------------------------------------------
_MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
_LANGUAGE = os.getenv("TTS_LANGUAGE", "tr")
# Yerleşik konuşmacı (XTTS-v2'de 58 seçenek). Kendi sesini klonlamak istersen
# TTS_SPEAKER_WAV'a kısa bir referans wav yolu ver (o zaman bu yok sayılır).
_SPEAKER = os.getenv("TTS_SPEAKER", "Claribel Dervla")
_SPEAKER_WAV = os.getenv("TTS_SPEAKER_WAV", "").strip() or None

# --- XTTS motor (inference) parametreleri (env ile ayarlanır) ----------------
# speed: konuşma temposu; >1 daha hızlı (çok az hızlandırmak için 1.1). Çok yüksek
#   doğallığı bozar.
_SPEED = float(os.getenv("TTS_SPEED", "1.1"))
# temperature: GPT örnekleme sıcaklığı. Düşük = daha monoton/kararlı, yüksek = ifadeli.
_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.75"))
# repetition_penalty: tekrar cezası (yüksek = kekeleme/tekrar azalır).
_REPETITION_PENALTY = float(os.getenv("TTS_REPETITION_PENALTY", "10.0"))
_TOP_K = int(os.getenv("TTS_TOP_K", "50"))
_TOP_P = float(os.getenv("TTS_TOP_P", "0.85"))
_LENGTH_PENALTY = float(os.getenv("TTS_LENGTH_PENALTY", "1.0"))

# --- Parça-sınırı temizliği: baş/son sessizlik kırpma + kısa fade in/out ------
_FADE_MS = float(os.getenv("TTS_FADE_MS", "12"))
_TRIM_SILENCE = os.getenv("TTS_TRIM_SILENCE", "1").lower() in ("1", "true", "yes")
_SILENCE_THRESH = float(os.getenv("TTS_SILENCE_THRESH", "0.015"))
_KEEP_PAD_MS = float(os.getenv("TTS_KEEP_PAD_MS", "30"))


def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    max_chunks: int = MAX_CHUNKS,
) -> List[str]:
    """Metni cümle sınırlarında ≤``max_chars`` karakterlik parçalara böler.

    En çok ``max_chunks`` parça döner; tavanı aşan artık metin dışarıda bırakılır.
    Tek bir cümle TEK BAŞINA sınırı aşıyorsa (noktalaması az uzun bir cümle),
    XTTS'in kendi iç bölücüsüne hiç bırakmamak için kelime sınırında ZORLA bölünür.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]

    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if len(s) > max_chars:
            # Tek cümle sınırı aşıyor: önce birikeni kapat, sonra kelime kelime böl.
            if cur:
                chunks.append(cur)
                if len(chunks) >= max_chunks:
                    return chunks
                cur = ""
            piece = ""
            for w in s.split():
                candidate = (piece + " " + w).strip() if piece else w
                if piece and len(candidate) > max_chars:
                    chunks.append(piece)
                    if len(chunks) >= max_chunks:
                        return chunks
                    piece = w
                else:
                    piece = candidate
            cur = piece
            continue

        candidate = (cur + " " + s).strip() if cur else s
        if cur and len(candidate) > max_chars:
            chunks.append(cur)
            if len(chunks) >= max_chunks:
                return chunks
            cur = s
        else:
            cur = candidate
    if cur and len(chunks) < max_chunks:
        chunks.append(cur)
    return chunks


@lru_cache(maxsize=1)
def _get_model():
    """XTTS-v2 modelini bir kez yükler (tembel), sonra bellekte sıcak tutar.

    torch/TTS import'ları burada yapılır ki sesli mod kapalıyken bu ağır
    bağımlılıklar hiç yüklenmesin. CUDA varsa GPU, yoksa CPU.
    """
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TTS(_MODEL_NAME).to(device)


def _postprocess(wav, sr: int):
    """Parça-sınırı artefaktları: baş/son sessizliği kırp + kısa fade in/out.

    Sessizlik kırpma dikişteki ölü havayı, fade ise iki uçtaki klik/onset'i
    yumuşatır. numpy dizisi döner.
    """
    import numpy as np

    x = np.asarray(wav, dtype=np.float32).flatten()
    n = x.size
    if n == 0:
        return x

    if _TRIM_SILENCE:
        idx = np.where(np.abs(x) > _SILENCE_THRESH)[0]
        if idx.size:
            pad = int(sr * _KEEP_PAD_MS / 1000)
            start = max(0, int(idx[0]) - pad)
            end = min(n, int(idx[-1]) + 1 + pad)
            x = x[start:end].copy()

    f = min(int(sr * _FADE_MS / 1000), x.size // 2)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        x[:f] *= ramp
        x[-f:] *= ramp[::-1]
    return x


def synthesize(text: str, out_dir: Path, run_id: str) -> List[str]:
    """Metni parçalayıp XTTS ile seslendirir; üretilen wav yollarını döner.

    Dosyalar ``<run_id>_00.wav`` ... diye SIRAYLA numaralanır (aşamalı oynatma).
    Boş metinde boş liste döner. Hata olursa çağıran (api.py) yakalar ve metin
    cevabını sesli olmadan döndürür.
    """
    if _TTS_DISABLED:
        # XTTS'i (VRAM) hiç yükleme — bu backend'de yalnızca Groq STT (cloud,
        # VRAM'siz) aktif kalsın. torch/TTS import'una bile girmeden dönülür.
        return []

    # Parçalamadan ÖNCE tam metin üzerinde: sıra sayısı/kısaltma gibi kurallar
    # cümle bağlamına (sonraki kelime, noktalama) bakıyor; parça sınırı bunu bölerse
    # yanlış eşleşebilir.
    text = _normalize_tr(text)
    chunks = chunk_text(text)
    if not chunks:
        return []

    import soundfile as sf

    model = _get_model()
    sr = model.synthesizer.output_sample_rate
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    for i, chunk in enumerate(chunks):
        kwargs = {
            "text": chunk,
            "language": _LANGUAGE,
            "speed": _SPEED,
            "temperature": _TEMPERATURE,
            "repetition_penalty": _REPETITION_PENALTY,
            "top_k": _TOP_K,
            "top_p": _TOP_P,
            "length_penalty": _LENGTH_PENALTY,
        }
        if _SPEAKER_WAV:
            kwargs["speaker_wav"] = _SPEAKER_WAV
        else:
            kwargs["speaker"] = _SPEAKER
        wav = _postprocess(model.tts(**kwargs), sr)
        path = out_dir / f"{run_id}_{i:02d}.wav"
        sf.write(str(path), wav, sr)
        paths.append(str(path))
    return paths
