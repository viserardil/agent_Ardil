"""STT: Qwen3-ASR ile SAF (batch) test modu.

DURUM: faster-whisper + silero-VAD tabanlı VAD-segmentli interim/commit akışı
GEÇİCİ OLARAK DEVRE DIŞI. Amaç, Qwen3-ASR'ın ham doğruluğunu/davranışını, VAD
segmentasyonunun veya interim mantığının etkisi karışmadan test etmek.

Bu modda:
- ``add_audio`` gelen PCM'i SADECE biriktirir; interim/ara güncelleme YOK.
- Mikrofon durdurulup WebSocket kapanınca (``finalize``) tüm ses TEK SEFERDE
  Qwen3-ASR-1.7B ile deşifre edilir ve tek bir final mesaj döner.

Eski VAD/interim tasarımı (faster-whisper + silero-VAD, ~0.5 sn'de bir interim,
sessizlikte commit) git geçmişinde durur; Qwen streaming için yeterince hızlı
çıkarsa ya da hibrit (interim hızlı model + final Qwen) istenirse geri getirilir.

NOT — torchcodec: Qwen'in processor'ı ses dosyasını doğrudan yolla okurken
Windows'ta ``libtorchcodec`` DLL'lerini bulamayıp çöküyor (FFmpeg entegrasyonu
eksik/uyumsuz). Bunu atlamak için ses HER ZAMAN numpy dizisi + sampling_rate
olarak verilir (dosya yolu değil).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import numpy as np

SR = 16000  # Qwen3-ASR 16 kHz mono bekler

_MODEL_ID = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf")
_LANG = os.getenv("STT_STREAM_LANG_NAME", "Turkish")  # Qwen dil adı ister ("tr" değil)
_MAX_NEW_TOKENS = int(os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "256"))

# processor.apply_transcription_request(language=...) sistem mesajına SADECE dil
# adını tek kelime olarak koyuyor (ör. "Turkish") — bu zayıf bir İPUCU, katı bir
# kısıtlama DEĞİL; model yine de başka dile kayabiliyor. Bunu güçlü bir açık
# talimatla değiştiriyoruz (aynı chat-template mekanizması, elle kurulmuş mesaj).
_STRONG_LANG_PROMPT = os.getenv(
    "QWEN_ASR_SYSTEM_PROMPT",
    f"{_LANG}. Transcribe the audio strictly in {_LANG} only. Never switch to "
    "another language, even if a word or phrase sounds foreign; render "
    f"foreign-sounding words phonetically in {_LANG}.",
)


@lru_cache(maxsize=1)
def _get_asr():
    """Qwen3-ASR model + processor'ı bir kez yükler (tembel)."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(_MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        _MODEL_ID, device_map="auto", dtype="auto"
    )
    return processor, model


def warmup() -> None:
    """Modeli önceden yükle (ilk sesli istekte beklememek için)."""
    _get_asr()


class LiveTranscriber:
    """Tek bir konuşma oturumu — ŞU AN saf batch modda (bkz. modül dokstring'i).

    ``add_audio`` sesi biriktirir, hiçbir zaman ara güncelleme döndürmez.
    ``finalize`` biriken tüm sesi Qwen3-ASR ile TEK SEFERDE deşifre eder.
    """

    def __init__(self) -> None:
        self.committed = ""
        self._buf = np.zeros(0, dtype=np.float32)

    def add_audio(self, pcm: np.ndarray) -> List[Dict]:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size:
            self._buf = np.concatenate([self._buf, pcm])
        return []  # interim yok (devre dışı)

    def finalize(self) -> Dict:
        """Akış kapanınca biriken sesi tek seferde deşifre eder."""
        text = self._transcribe()
        if text:
            self.committed = (self.committed + " " + text).strip()
        return {"committed": self.committed, "interim": "", "final": True}

    # --- iç ------------------------------------------------------------

    def _transcribe(self) -> str:
        if len(self._buf) < int(0.2 * SR):  # <0.2 sn'yi çevirme
            return ""
        from transformers.models.qwen3_asr.processing_qwen3_asr import (
            _audio_content_item,
            make_list_of_audio_chat_template,
        )

        processor, model = _get_asr()
        # NOT: audio=dosya_yolu DEĞİL, numpy dizisi — torchcodec/FFmpeg DLL
        # sorununu atlamak için (bkz. modül dokstring'i).
        #
        # apply_transcription_request(language=...) yerine mesajı ELLE kuruyoruz:
        # o yardımcı, sistem mesajına SADECE dil adını (ör. "Turkish") koyuyor —
        # zayıf bir ipucu. Burada _STRONG_LANG_PROMPT ile dili SIKI şekilde
        # zorluyoruz (aynı chat-template mekanizması, daha güçlü metin).
        audio_item = make_list_of_audio_chat_template([self._buf])[0]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _STRONG_LANG_PROMPT}]},
            {"role": "user", "content": [_audio_content_item(audio_item)]},
        ]
        inputs = processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, model.dtype)
        output_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = processor.decode(generated_ids, return_format="transcription_only")[0]
        return text.strip()
