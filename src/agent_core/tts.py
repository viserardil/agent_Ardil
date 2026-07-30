"""Sesli çıktı: nihai cevabı Fish Audio S2-Pro (cloud API) ile seslendirir.

XTTS-v2 (yerel) yerine Fish Audio'nun cloud API'si kullanılıyor: aynı 5B parametreli
model (huggingface.co/fishaudio/s2-pro) Fish Audio'nun kendi sunucularında (yüksek
VRAM'li donanım — S2-Pro yerelde ~24 GB VRAM ister, bizim GPU'ya sığmaz) çalışıyor;
biz yalnızca HTTP isteği atıyoruz. Yerel GPU/VRAM/torch GEREKMEZ — bu modül hiçbir
ağır ML kütüphanesi import etmez.

Gerekli: .env'de ``FISH_AUDIO_API_KEY`` (bkz. fish.audio/app/developers — API
kredisi PLATFORM kredisinden AYRIDIR, orada ayrıca yüklenmesi gerekir).

Uzun metin cümle sınırında parçalanır (parça başına ≤``CHUNK_MAX_TOKENS`` token, en
çok ``MAX_CHUNKS`` parça); parçalar SIRAYLA numaralanır ki arayüz aşamalı oynatabilsin.
Her parça ayrı bir Fish Audio isteğidir; parça-sınırı klik/sessizliği için hafif
kırpma + fade uygulanır (XTTS'teki mantığın aynısı).

SES TUTARLILIĞI: Referans verilmezse Fish Audio her istekte FARKLI bir varsayılan
ses örnekliyor — parçalar arasında (hatta aynı cevap içinde) konuşmacı değişirdi.
Bunu önlemek için ``_get_reference`` BİR KEZ (ilk sentezde) referanssız bir klip
üretip diske cache'ler; sonraki TÜM parçalar/istekler bu klibi in-context referans
olarak kullanır — süreç yeniden başlasa bile ses hep aynı kalır.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from agent_core.tr_normalize import normalize as _normalize_tr

# --- Parçalama ayarları (env ile override edilebilir) ------------------------
CHUNK_MAX_TOKENS = int(os.getenv("TTS_CHUNK_TOKENS", "120"))
MAX_CHUNKS = int(os.getenv("TTS_MAX_CHUNKS", "24"))

# --- Fish Audio ayarları ------------------------------------------------------
# backend: s2-pro (en doğal/ifadeli, 5B) | s1 | s1-mini | speech-1.6 | speech-1.5
# | agent-x0. Dil AYRI bir parametre DEĞİL — Fish metnin dilini kendisi tespit
# eder (Türkçe metin Türkçe seslendirilir).
_BACKEND = os.getenv("FISH_BACKEND", "s2-pro")
# Fish Audio platformunda önceden oluşturulmuş bir ses modeli id'si (klonlama).
# Verilirse aşağıdaki otomatik-referans mekanizması ATLANIR.
_REFERENCE_ID = os.getenv("FISH_REFERENCE_ID", "").strip() or None
# Kendi ses klibini referans vermek istersen (ör. kayıtlı bir wav) + o klibin
# BİREBİR transkripti. İkisi de verilmezse otomatik referans (aşağıda) kullanılır.
_REFERENCE_AUDIO_PATH = os.getenv("FISH_REFERENCE_AUDIO_PATH", "").strip() or None
_REFERENCE_TEXT = os.getenv("FISH_REFERENCE_TEXT", "").strip() or None
_TOP_P = float(os.getenv("FISH_TOP_P", "0.7"))
_TEMPERATURE = float(os.getenv("FISH_TEMPERATURE", "0.7"))

# Otomatik referans cache'i: ilk sentezde bu sabit cümleyle referanssız bir klip
# üretilip diske yazılır, sonrasında hep o klip kullanılır (bkz. _get_reference).
_AUTO_REF_DIR = Path(__file__).resolve().parents[2] / "src" / "scratch" / "tts"
_AUTO_REF_AUDIO = _AUTO_REF_DIR / "_fish_voice_reference.wav"
_AUTO_REF_TEXT = _AUTO_REF_DIR / "_fish_voice_reference.txt"
_AUTO_REF_SEED_TEXT = (
    "Merhaba, ben yapay zeka asistanınızım. Size yardımcı olmak için buradayım."
)

# --- Parça-sınırı temizliği: baş/son sessizlik kırpma + kısa fade in/out ------
_FADE_MS = float(os.getenv("TTS_FADE_MS", "12"))
_TRIM_SILENCE = os.getenv("TTS_TRIM_SILENCE", "1").lower() in ("1", "true", "yes")
_SILENCE_THRESH = float(os.getenv("TTS_SILENCE_THRESH", "0.015"))
_KEEP_PAD_MS = float(os.getenv("TTS_KEEP_PAD_MS", "30"))

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


@lru_cache(maxsize=1)
def _get_session():
    """Fish Audio API oturumunu bir kez kurar (tembel)."""
    from fish_audio_sdk import Session

    api_key = os.getenv("FISH_AUDIO_API_KEY")
    if not api_key:
        raise RuntimeError("FISH_AUDIO_API_KEY tanımlı değil (.env).")
    return Session(api_key)


@lru_cache(maxsize=1)
def _get_reference():
    """Tüm parçalarda/isteklerde AYNI sesin konuşması için sabit bir referans.

    Öncelik: FISH_REFERENCE_ID (Fish Audio'da oluşturulmuş kalıcı ses) > kullanıcının
    verdiği FISH_REFERENCE_AUDIO_PATH/TEXT > otomatik referans (diskte cache'li,
    yoksa bir kez üretilir). Dönen ``None`` demek "reference_id kullanılacak, bu
    mekanizmaya gerek yok".
    """
    if _REFERENCE_ID:
        return None

    if _REFERENCE_AUDIO_PATH:
        if not _REFERENCE_TEXT:
            raise RuntimeError(
                "FISH_REFERENCE_AUDIO_PATH verildiyse FISH_REFERENCE_TEXT "
                "(klibin birebir transkripti) de gerekir."
            )
        return Path(_REFERENCE_AUDIO_PATH).read_bytes(), _REFERENCE_TEXT

    if _AUTO_REF_AUDIO.exists() and _AUTO_REF_TEXT.exists():
        return _AUTO_REF_AUDIO.read_bytes(), _AUTO_REF_TEXT.read_text(encoding="utf-8")

    # İlk sentez: referanssız bir klip üret (Fish'in o an seçtiği varsayılan ses),
    # diske yaz — bundan sonraki HER sentez bu klibi referans alacak.
    from fish_audio_sdk import TTSRequest

    session = _get_session()
    audio = b"".join(
        session.tts(
            TTSRequest(
                text=_AUTO_REF_SEED_TEXT, format="wav", top_p=_TOP_P, temperature=_TEMPERATURE
            ),
            backend=_BACKEND,
        )
    )
    _AUTO_REF_DIR.mkdir(parents=True, exist_ok=True)
    _AUTO_REF_AUDIO.write_bytes(audio)
    _AUTO_REF_TEXT.write_text(_AUTO_REF_SEED_TEXT, encoding="utf-8")
    return audio, _AUTO_REF_SEED_TEXT


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
    """Metni parçalayıp Fish Audio (S2-Pro) ile seslendirir; üretilen wav yollarını döner.

    Dosyalar ``<run_id>_00.wav`` ... diye SIRAYLA numaralanır (aşamalı oynatma).
    Boş metinde boş liste döner. Hata olursa çağıran (api.py) yakalar ve metin
    cevabını sesli olmadan döndürür.
    """
    # Parçalamadan ÖNCE tam metin üzerinde: sıra sayısı/kısaltma gibi kurallar
    # cümle bağlamına (sonraki kelime, noktalama) bakıyor; parça sınırı bunu bölerse
    # yanlış eşleşebilir. Ayrıca API'ye ham rakam değil Türkçe kelime gitsin diye
    # (Fish Audio rakamları bazen yanlış dilde okuyordu).
    text = _normalize_tr(text)
    chunks = chunk_text(text)
    if not chunks:
        return []

    import io

    import soundfile as sf
    from fish_audio_sdk import ReferenceAudio, TTSRequest

    session = _get_session()
    reference = _get_reference()  # tüm parçalarda AYNI sesi garantiler
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    for i, chunk in enumerate(chunks):
        kwargs = {"text": chunk, "format": "wav", "top_p": _TOP_P, "temperature": _TEMPERATURE}
        if _REFERENCE_ID:
            kwargs["reference_id"] = _REFERENCE_ID
        elif reference is not None:
            ref_audio, ref_text = reference
            kwargs["references"] = [ReferenceAudio(audio=ref_audio, text=ref_text)]
        audio_bytes = b"".join(session.tts(TTSRequest(**kwargs), backend=_BACKEND))
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        wav = _postprocess(wav, sr)
        path = out_dir / f"{run_id}_{i:02d}.wav"
        sf.write(str(path), wav, sr)
        paths.append(str(path))
    return paths
