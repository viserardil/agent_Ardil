"""Chatterbox TTS mikroservisi — İZOLE .venv-voice içinde çalışır.

Neden ayrı süreç? Chatterbox `numpy<2` dayatıyor; ana AgentArdil venv'inde ise
`scipy` numpy 2.x'e göre derli. İkisi tek venv'de barışmıyor (binary uyumsuzluk).
Bu yüzden Chatterbox kendi venv'inde (.venv-voice) ayrı bir HTTP servisi olarak
koşar; ana API (agent_core/tts.py) buna HTTP ile konuşur.

Model BİR KEZ yüklenir (süreç boyu sıcak kalır). POST /tts, parça listesi alır,
her parçayı ayrı wav'a sentezler ve dosya yollarını döner. run_api.py bu servisi
başlangıçta otomatik başlatabilir.

Çalıştırma (izole venv ile):
    .venv-voice/Scripts/python voice_service.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio as ta
from fastapi import FastAPI
from pydantic import BaseModel

# Chatterbox stabilite/ses knob'ları (env ile ayarlanır).
_LANG = os.getenv("TTS_LANGUAGE", "tr")
# temperature: DÜŞÜK = daha deterministik/stabil (dalgalanma azalır). Varsayılan 0.8.
_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.6"))
_EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", "0.5"))  # düşük = daha nötr/stabil
_CFG_WEIGHT = float(os.getenv("TTS_CFG_WEIGHT", "0.5"))      # ~0.3 daha hızlı/akıcı
_REPETITION_PENALTY = float(os.getenv("TTS_REPETITION_PENALTY", "2.0"))
# İsteğe bağlı ses klonlama referansı (kısa bir wav yolu). Boşsa varsayılan ses.
_AUDIO_PROMPT = os.getenv("TTS_AUDIO_PROMPT", "").strip() or None

app = FastAPI(title="AgentArdil Voice Service (Chatterbox)")
_model = None


def _get_model():
    """Chatterbox Multilingual modelini bir kez yükler (tembel), sonra sıcak tutar."""
    global _model
    if _model is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    return _model


class TTSRequest(BaseModel):
    chunks: List[str]
    out_dir: str
    run_id: str
    language_id: Optional[str] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": _model is not None}


@app.post("/tts")
def tts(req: TTSRequest) -> dict:
    """Parçaları sırayla sentezler; <run_id>_NN.wav diye yazıp yolları döner."""
    model = _get_model()
    out = Path(req.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lang = req.language_id or _LANG
    exa = req.exaggeration if req.exaggeration is not None else _EXAGGERATION
    cfg = req.cfg_weight if req.cfg_weight is not None else _CFG_WEIGHT

    paths: List[str] = []
    for i, chunk in enumerate(req.chunks):
        wav = model.generate(
            chunk,
            language_id=lang,
            audio_prompt_path=_AUDIO_PROMPT,
            exaggeration=exa,
            cfg_weight=cfg,
            temperature=_TEMPERATURE,
            repetition_penalty=_REPETITION_PENALTY,
        )
        path = out / f"{req.run_id}_{i:02d}.wav"
        ta.save(str(path), wav, model.sr)
        paths.append(str(path))
    return {"paths": paths}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("VOICE_SERVICE_PORT", "8756"))
    # Modeli açılışta yükle ki ilk istek beklemesin (sıcak başlat).
    print(f"[voice] Chatterbox yükleniyor (lang={_LANG}, temp={_TEMPERATURE})...", flush=True)
    _get_model()
    print(f"[voice] hazır -> http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
