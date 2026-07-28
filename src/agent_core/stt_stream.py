"""Canlı (streaming) STT: konuşurken deşifre — faster-whisper + silero-VAD.

Groq batch yolundan (stt.py) TAMAMEN AYRI. Tarayıcı 16 kHz mono float32 PCM akıtır;
``LiveTranscriber`` konuşmayı VAD ile cümlelere böler:
- Konuşurken kısa aralıklarla (~0.5 sn) mevcut cümleyi deşifre edip **interim**
  (geçici) metni yollar → ekranda soluk, canlı büyür.
- Kısa bir sessizlik (pause) gelince cümleyi deşifre edip **committed** (kesin)
  yapar, tamponu sıfırlar → cümle sabitlenir, sonraki cümleye geçilir.

Neden VAD-segmentli? Whisper batch bir model; büyüyen tamponu tekrar tekrar çevirmek
pahalanır. Cümleleri sessizlikte kesip kısa tutmak, deşifreyi hızlı ve canlı tutar
(RTX 4050'de `small` ~6-7x gerçek-zaman → interim akıcı).

NOT (v1): silero VAD durum-bilgili (stateful) modeldir; tek kullanıcı/tek oturum
varsayımıyla paylaşılır. Eşzamanlı çok oturum gerekirse oturum başına ayrı VAD
örneği gerekir.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import numpy as np

SR = 16000  # Whisper + silero 16 kHz bekler

_MODEL_NAME = os.getenv("STT_STREAM_MODEL", "medium")
_LANG = (os.getenv("STT_STREAM_LANG", "tr") or "tr").strip()
# İnterim'i bu kadar YENİ sesle bir yenile (throttle).
_INTERIM_EVERY_S = float(os.getenv("STT_INTERIM_EVERY", "0.5"))
# Bu kadar ardışık sessizlik = cümle sonu (commit).
_SILENCE_END_S = float(os.getenv("STT_SILENCE_END", "0.6"))
# silero konuşma olasılığı eşiği (0-1).
_VAD_THRESHOLD = float(os.getenv("STT_VAD_THRESHOLD", "0.5"))
# Güvenlik: cümle bu kadar uzarsa zorla kes (sessizlik gelmese bile).
_MAX_PHRASE_S = float(os.getenv("STT_MAX_PHRASE", "20"))

_VAD_FRAME = 512  # silero 16 kHz'de 512 örneklik kareler ister


@lru_cache(maxsize=1)
def _get_asr():
    """faster-whisper modelini bir kez yükler (tembel)."""
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8_float16" if device == "cuda" else "int8"
    return WhisperModel(_MODEL_NAME, device=device, compute_type=compute)


@lru_cache(maxsize=1)
def _get_vad():
    """silero-VAD modelini bir kez yükler (tembel)."""
    from silero_vad import load_silero_vad

    return load_silero_vad()


def warmup() -> None:
    """Modelleri önceden yükle (ilk sesli istekte beklememek için)."""
    _get_asr()
    _get_vad()


class LiveTranscriber:
    """Tek bir konuşma oturumunun canlı deşifresi.

    ``add_audio`` her PCM parçasında çağrılır; yollanacak güncelleme(ler)i
    ``[{committed, interim, final}]`` olarak döner. ``committed`` sabitlenmiş metin,
    ``interim`` o an konuşulan cümlenin geçici hâli, ``final`` bir cümlenin
    kesinleştiğini bildirir.
    """

    def __init__(self) -> None:
        self.committed = ""                                # sabitlenmiş metin (tüm cümleler)
        self._phrase = np.zeros(0, dtype=np.float32)       # o anki cümlenin sesi
        self._vad_tail = np.zeros(0, dtype=np.float32)     # VAD çerçevesine sığmayan artık
        self._has_speech = False                           # cümlede konuşma oldu mu
        self._silence = 0                                  # ardışık sessizlik örneği
        self._len_at_last_tx = 0                           # son interim deşifredeki uzunluk
        self._interim = ""

    def add_audio(self, pcm: np.ndarray) -> List[Dict]:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return []
        self._phrase = np.concatenate([self._phrase, pcm])
        self._run_vad(pcm)
        return self._maybe_emit()

    def finalize(self) -> Dict:
        """Akış kapanınca kalan cümleyi sabitle."""
        if self._has_speech:
            t = self._transcribe()
            if t:
                self.committed = (self.committed + " " + t).strip()
        self._reset_phrase()
        return {"committed": self.committed, "interim": "", "final": True}

    # --- iç ----------------------------------------------------------------

    def _run_vad(self, pcm: np.ndarray) -> None:
        import torch

        vad = _get_vad()
        buf = np.concatenate([self._vad_tail, pcm])
        n = (len(buf) // _VAD_FRAME) * _VAD_FRAME
        for i in range(0, n, _VAD_FRAME):
            frame = buf[i : i + _VAD_FRAME]
            prob = vad(torch.from_numpy(frame), SR).item()
            if prob >= _VAD_THRESHOLD:
                self._has_speech = True
                self._silence = 0
            else:
                self._silence += _VAD_FRAME
        self._vad_tail = buf[n:]

    def _maybe_emit(self) -> List[Dict]:
        # Cümle sonu: konuşma oldu + yeterli sessizlik → commit
        if self._has_speech and self._silence >= int(_SILENCE_END_S * SR):
            return self._commit_phrase()
        # Güvenlik: cümle çok uzadıysa zorla kes
        if len(self._phrase) >= int(_MAX_PHRASE_S * SR):
            return self._commit_phrase()
        # İnterim: konuşma var + yeterince yeni ses geldi
        if self._has_speech and (len(self._phrase) - self._len_at_last_tx) >= int(_INTERIM_EVERY_S * SR):
            self._len_at_last_tx = len(self._phrase)
            interim = self._transcribe()
            if interim and interim != self._interim:
                self._interim = interim
                return [{"committed": self.committed, "interim": interim, "final": False}]
        # Konuşma yokken tamponu şişirme: baştaki uzun sessizliği at
        if not self._has_speech and len(self._phrase) > SR:
            self._reset_phrase()
        return []

    def _commit_phrase(self) -> List[Dict]:
        text = self._transcribe()
        if text:
            self.committed = (self.committed + " " + text).strip()
        self._reset_phrase()
        return [{"committed": self.committed, "interim": "", "final": True}]

    def _transcribe(self) -> str:
        if len(self._phrase) < int(0.2 * SR):  # <0.2 sn'yi çevirme
            return ""
        segments, _ = _get_asr().transcribe(
            self._phrase,
            language=_LANG,
            beam_size=1,
            condition_on_previous_text=False,
        )
        return "".join(s.text for s in segments).strip()

    def _reset_phrase(self) -> None:
        self._phrase = np.zeros(0, dtype=np.float32)
        self._vad_tail = np.zeros(0, dtype=np.float32)
        self._has_speech = False
        self._silence = 0
        self._len_at_last_tx = 0
        self._interim = ""
        try:
            _get_vad().reset_states()
        except Exception:
            pass
