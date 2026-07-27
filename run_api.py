"""API sunucusunu başlatır.

    python run_api.py            # http://127.0.0.1:8000
    python run_api.py --reload   # geliştirme: dosya değişince yeniden başlat

Sesli çıktı (XTTS) ayrı bir servis GEREKTİRMEZ: aynı süreçte, ilk sesli istekte
tembel yüklenir (bkz. agent_core/tts.py).

Ara süreç logları: logs/sessions/<session_id>.log
Canlı izlemek için:  Get-Content logs\sessions\<id>.log -Wait -Tail 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentArdil API sunucusu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Geliştirme modu")
    args = parser.parse_args()

    uvicorn.run(
        "agent_core.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
