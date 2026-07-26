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

# .env'i BU süreç de yükler: voice_service ana API'den ayrı bir süreç olduğu için
# ana venv'in config'i onun ortamına ulaşmaz. TTS_* ayarları buradan okunur.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except Exception:  # python-dotenv yoksa env'ler yine os ortamından okunur
    pass

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

# --- Parça-sınırı artefaktı temizliği ----------------------------------------
# Chatterbox parça sonunda kısa bir "kuyruk" (nefes/cızırtı) ve parça başında onset
# transient üretebiliyor; parçalar arka arkaya çalınınca bunlar DİKİŞTE duyuluyor.
# Baş/son sessizliği kırpıp kısa bir fade in/out uygulayarak yumuşatıyoruz.
_FADE_MS = float(os.getenv("TTS_FADE_MS", "12"))              # 0 = fade kapalı
_TRIM_SILENCE = os.getenv("TTS_TRIM_SILENCE", "1").lower() in ("1", "true", "yes")
_SILENCE_THRESH = float(os.getenv("TTS_SILENCE_THRESH", "0.015"))  # mutlak genlik eşiği
_KEEP_PAD_MS = float(os.getenv("TTS_KEEP_PAD_MS", "30"))      # kırpmada bırakılan pay
# Kuyruk hallüsinasyonu: Chatterbox konuşma bittikten sonra bir BOŞLUK bırakıp kısa
# bir gürültü/nefes patlaması üretebiliyor. Bu, sessizlik-kırpmayla YAKALANMAZ
# (gürültü eşiğin üstünde). Sondaki boşluktan (>= gap) sonra gelen KISA (<= max_tail)
# segment(ler)i tamamen keseriz. Gerçek son cümle boşluktan önce bittiği için korunur.
_TRIM_TAIL = os.getenv("TTS_TRIM_TAIL", "1").lower() in ("1", "true", "yes")
_TAIL_GAP_MS = float(os.getenv("TTS_TAIL_GAP_MS", "180"))    # segmentleri ayıran boşluk
_MAX_TAIL_MS = float(os.getenv("TTS_MAX_TAIL_MS", "450"))    # bu süreden kısa son segment = junk

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


def _voiced_segments(voiced: List[bool], gap_frames: int) -> List[tuple]:
    """Sesli (voiced) kare dizisinden segmentleri çıkarır.

    Bir segment, arası ``gap_frames``'ten kısa boşluklarla bağlı sesli karelerdir;
    boşluk >= gap_frames olunca segment biter. (start, end) kare indeksleri döner.
    """
    segs: List[tuple] = []
    i, nf = 0, len(voiced)
    while i < nf:
        if not voiced[i]:
            i += 1
            continue
        start = i
        last_voiced = i
        silence = 0
        j = i
        while j < nf:
            if voiced[j]:
                last_voiced = j
                silence = 0
            else:
                silence += 1
                if silence >= gap_frames:
                    break
            j += 1
        segs.append((start, last_voiced + 1))
        i = j + 1
    return segs


def _postprocess(wav: "torch.Tensor", sr: int) -> "torch.Tensor":
    """Parça-sınırı artefaktlarını azalt: kuyruk hallüsinasyonunu kes + kırp + fade.

    - Kuyruk kesme: konuşma bittikten sonra bir BOŞLUKLA ayrılan kısa gürültü/
      nefes patlamasını (Chatterbox hallüsinasyonu) tamamen atar.
    - Sessizlik kırpma: baştaki/sondaki ölü havayı ve boşluğu atar.
    - Fade in/out: iki uçtaki klik/onset ve kalan kuyruğu yumuşatır (~12 ms).
    Çıktı her zaman [1, N] şeklindedir.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mono = wav.detach().to("cpu").float()[0].clone()
    n = mono.numel()
    if n == 0:
        return mono.unsqueeze(0)

    frame = max(1, int(sr * 0.010))  # 10 ms kare
    nf = n // frame
    if nf >= 2 and (_TRIM_SILENCE or _TRIM_TAIL):
        energy = mono[: nf * frame].abs().reshape(nf, frame).amax(dim=1)
        voiced = (energy > _SILENCE_THRESH).tolist()
        segs = _voiced_segments(voiced, gap_frames=max(1, int(_TAIL_GAP_MS / 10)))
        if segs:
            # Sondaki KISA segment(ler)i (boşlukla ayrılmış junk) at.
            if _TRIM_TAIL:
                max_tail = max(1, int(_MAX_TAIL_MS / 10))
                while len(segs) >= 2 and (segs[-1][1] - segs[-1][0]) <= max_tail:
                    segs.pop()
            pad = int(sr * _KEEP_PAD_MS / 1000)
            start = max(0, segs[0][0] * frame - pad) if _TRIM_SILENCE else 0
            end = min(n, segs[-1][1] * frame + pad)
            mono = mono[start:end].clone()

    f = min(int(sr * _FADE_MS / 1000), mono.numel() // 2)
    if f > 0:
        ramp = torch.linspace(0.0, 1.0, f, dtype=mono.dtype)
        mono[:f] *= ramp
        mono[-f:] *= ramp.flip(0)

    return mono.unsqueeze(0)


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
        wav = _postprocess(wav, model.sr)  # dikiş artefaktlarını yumuşat
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
