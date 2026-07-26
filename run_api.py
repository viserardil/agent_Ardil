"""API sunucusunu başlatır.

    python run_api.py            # http://127.0.0.1:8000
    python run_api.py --reload   # geliştirme: dosya değişince yeniden başlat

Ara süreç logları: logs/sessions/<session_id>.log
Canlı izlemek için:  Get-Content logs\sessions\<id>.log -Wait -Tail 40
"""

from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn  # noqa: E402

_ROOT = Path(__file__).parent
_voice_proc: subprocess.Popen | None = None


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _stop_voice_service() -> None:
    if _voice_proc and _voice_proc.poll() is None:
        _voice_proc.terminate()


def _maybe_start_voice_service() -> None:
    """İzole Chatterbox TTS servisini (voice_service.py) otomatik başlatır.

    Servis ayrı bir venv'de (.venv-voice) koşar çünkü Chatterbox numpy<2 isterken
    ana venv numpy 2.x kullanıyor. .venv-voice yoksa ya da servis zaten ayaktaysa
    sessizce atlanır; sesli mod o durumda çalışmaz ama ana API etkilenmez.
    """
    global _voice_proc
    if os.getenv("VOICE_AUTOSTART", "1").lower() not in ("1", "true", "yes"):
        return

    port = int(os.getenv("VOICE_SERVICE_PORT", "8756"))
    if _port_open("127.0.0.1", port):
        print(f"[voice] Chatterbox servisi zaten ayakta (:{port})")
        return

    vpy = _ROOT / ".venv-voice" / "Scripts" / "python.exe"
    if not vpy.exists():
        print("[voice] .venv-voice bulunamadı — sesli mod devre dışı "
              "(kurulum için README > Sesli çıktı bölümüne bak)")
        return

    log_path = _ROOT / "logs" / "voice_service.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8")
    _voice_proc = subprocess.Popen(
        [str(vpy), str(_ROOT / "voice_service.py")],
        stdout=logf, stderr=logf, cwd=str(_ROOT),
    )
    atexit.register(_stop_voice_service)
    print(f"[voice] Chatterbox servisi başlatıldı (pid={_voice_proc.pid}); "
          f"model yüklenirken ilk sesli istek biraz bekleyebilir. Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentArdil API sunucusu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Geliştirme modu")
    parser.add_argument("--no-voice", action="store_true",
                        help="Sesli çıktı servisini otomatik başlatma")
    args = parser.parse_args()

    if not args.no_voice:
        _maybe_start_voice_service()

    uvicorn.run(
        "agent_core.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
