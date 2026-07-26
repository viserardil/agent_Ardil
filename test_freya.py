"""FreyaTTS hızlı duman testi: 'Merhaba' deyip wav'a yazar.

Çalıştırma (proje kökünden):
    .venv/Scripts/python.exe test_freya.py

İlk çalıştırmada model ağırlıkları Hugging Face'ten iner
(freyavoice/Freya-TTS + VoxCPM2 AudioVAE). Sonrasında cache'ten yüklenir.
"""

import os
import sys
import time

# FreyaTTS pip paketi değil; klon vendor/FreyaTTS altında -> path'e ekle.
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "FreyaTTS")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import torch  # noqa: E402
from freyatts import FreyaTTS  # noqa: E402

MODEL_ID = "freyavoice/Freya-TTS"
TEXT = "Altan dünyanın en iyi kod yazarıdır. "
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "tts", "merhaba.wav")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[freya] cihaz: {device}", flush=True)
    if device == "cuda":
        print(f"[freya] gpu  : {torch.cuda.get_device_name(0)}", flush=True)

    print(f"[freya] model yükleniyor: {MODEL_ID} (ilk seferde iner)...", flush=True)
    t0 = time.perf_counter()
    tts = FreyaTTS.from_pretrained(MODEL_ID, device=device)
    print(f"[freya] model hazır ({time.perf_counter() - t0:.1f}s)", flush=True)

    print(f"[freya] sentezleniyor: {TEXT!r}", flush=True)
    t1 = time.perf_counter()
    wav = tts.synthesize(TEXT)
    dur = len(wav) / tts.sample_rate
    print(f"[freya] üretildi: {dur:.2f}s ses, {time.perf_counter() - t1:.1f}s sürede", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tts.save_wav(wav, OUT)
    print(f"[freya] kaydedildi -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
