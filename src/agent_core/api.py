"""Frontend'in konuştuğu HTTP API.

TASARIM KARARLARI (bilinçli):
- Session ve mesajlar BELLEKTE tutulur; sunucu yeniden başlayınca gider. Kalıcılık
  şimdilik istenmedi.
- Frontend'e yalnızca NİHAİ CEVAP döner. Ara süreç (triyaj kararı, düğümler, LLM
  girdi/çıktıları, token, süre) oraya akmaz; koşu sürerken logs/sessions/<id>.log
  dosyasına anlık yazılır. UI o sırada "düşünüyor" gösterir.
- Ajan turlar arası hafızaya SAHİP DEĞİL: her mesaj bağımsız çalışır. Sohbet
  hafızası ayrı bir adımda ele alınacak.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent_core.router import build_agent_graph
from agent_core.tracing import RunTracer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs" / "sessions"
# plot_chart / visualize_data üretilen görselleri buraya yazıyor.
CHART_DIR = PROJECT_ROOT / "src" / "scratch" / "charts"
# Sesli çıktı modunda Chatterbox'ın ürettiği wav parçaları buraya yazılır.
AUDIO_DIR = PROJECT_ROOT / "src" / "scratch" / "tts"


# --- Bellek içi depo ---------------------------------------------------------


class Message(BaseModel):
    id: str
    role: str  # "user" | "ai"
    text: str
    created_at: str
    # Yalnızca ai mesajlarında dolu: koşunun özeti (şerit, süre, token, dosyalar).
    run: Optional[dict] = None


class Session(BaseModel):
    id: str
    title: str
    created_at: str


class _Store:
    """Bellek içi session deposu. Süreç yeniden başlayınca sıfırlanır."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Session] = {}
        self.messages: Dict[str, List[Message]] = {}
        self.runs: Dict[str, dict] = {}  # run_id -> {"summary":..., "events":[...]}

    def create_session(self, title: str = "Yeni sohbet") -> Session:
        session_id = uuid.uuid4().hex[:8]
        session = Session(
            id=session_id,
            title=title,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.sessions[session_id] = session
        self.messages[session_id] = []
        return session

    def require(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Oturum bulunamadı: {session_id}")
        return session

    def add_message(self, session_id: str, role: str, text: str, run: Optional[dict] = None) -> Message:
        message = Message(
            id=uuid.uuid4().hex[:12],
            role=role,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
            run=run,
        )
        self.messages[session_id].append(message)
        return message


store = _Store()


# --- İstek/cevap şemaları ----------------------------------------------------


class SendMessageRequest(BaseModel):
    text: str = Field(min_length=1)
    # Sesli çıktı modu: açıksa nihai cevap Chatterbox ile seslendirilip parça parça
    # döner (opsiyonel). Kapalıyken sesli çıktı servisine hiç istek gitmez.
    voice: bool = False


class SendMessageResponse(BaseModel):
    user_message: Message
    ai_message: Message
    run: dict


# --- Uygulama ----------------------------------------------------------------

app = FastAPI(title="AgentArdil API", version="0.1.0")

# Vite dev sunucusu başka bir portta çalışıyor; tarayıcı isteği engellemesin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None


def _get_graph():
    """Grafiği ilk istekte kurar (import anında LLM anahtarı aranmasın diye)."""
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "sessions": len(store.sessions)}


@app.get("/api/sessions", response_model=List[Session])
def list_sessions() -> List[Session]:
    return list(store.sessions.values())


@app.post("/api/sessions", response_model=Session)
def create_session() -> Session:
    return store.create_session()


@app.get("/api/sessions/{session_id}/messages", response_model=List[Message])
def get_messages(session_id: str) -> List[Message]:
    store.require(session_id)
    return store.messages[session_id]


@app.post("/api/sessions/{session_id}/messages", response_model=SendMessageResponse)
def send_message(session_id: str, request: SendMessageRequest) -> SendMessageResponse:
    """Kullanıcı mesajını ajana verir ve NİHAİ cevabı döndürür.

    Ara süreç bu cevaba girmez; koşarken log dosyasına yazılır. Dönen ``run``
    özeti sadece rakamlardır (şerit, süre, token, üretilen dosyalar) — UI bunu
    footer'da gösterebilir.
    """
    session = store.require(session_id)

    # Session hafızası: bu turdan ÖNCEKİ tüm konuşma (kullanıcı mesajları + ajan
    # cevapları). Güncel mesajı eklemeden önce topluyoruz ki geçmişe kendisi girmesin.
    history = [{"role": m.role, "text": m.text} for m in store.messages[session_id]]

    user_message = store.add_message(session_id, "user", request.text)

    # İlk kullanıcı mesajı sohbetin başlığı olsun (sidebar'da anlamlı görünsün).
    if session.title == "Yeni sohbet":
        session.title = request.text[:48] + ("…" if len(request.text) > 48 else "")

    tracer = RunTracer(session_id=session_id, user_input=request.text, log_dir=LOG_DIR)
    config = {
        "recursion_limit": 50,
        "callbacks": [tracer],          # LLM ve araç olaylarını yakalar
        "configurable": {"tracer": tracer},  # düğümlerin açık çağrıları için
    }

    try:
        result = _get_graph().invoke(
            {"input": request.text, "history": history}, config=config
        )
        answer = result.get("response") or "(cevap üretilemedi)"
        # Şeridin normalize ettiği dosya listesi yetkili kaynaktır; izleyicininkiyle
        # birleştir ki hiçbir üretilmiş görsel özetin dışında kalmasın.
        tracer.add_artifacts(result.get("artifacts") or [])
        summary = tracer.finish(answer)
    except Exception as exc:  # koşu çökerse UI boş kalmasın, hata da loglansın
        summary = tracer.finish("", error=f"{type(exc).__name__}: {exc}")
        answer = f"Koşu sırasında hata oluştu: {type(exc).__name__}: {exc}"
        result = {}

    # Üretilen dosyaları frontend'in çekebileceği URL'lere çevir.
    summary["artifact_urls"] = [
        f"/api/artifacts/{Path(path).name}" for path in summary.get("artifacts", [])
    ]

    # Sesli çıktı: istenmişse cevabı XTTS-v2 ile seslendir (parça parça, sırayla).
    # Metin ile BİRLİKTE döner. Sentez süreç içinde yapılır (agent_core/tts.py);
    # model ilk çağrıda tembel yüklenir. Üretim çökerse metin cevabı bozulmasın diye
    # hata yalnızca loglanır; audio_urls boş kalır.
    summary["audio_urls"] = []
    if request.voice and answer and not summary.get("error"):
        try:
            from agent_core import tts

            paths = tts.synthesize(answer, AUDIO_DIR, summary["run_id"])
            summary["audio_urls"] = [f"/api/audio/{Path(p).name}" for p in paths]
        except Exception as exc:  # noqa: BLE001 — ses opsiyonel, metin akışı korunur
            print(f"[TTS] seslendirme başarısız: {type(exc).__name__}: {exc}")

    store.runs[summary["run_id"]] = {"summary": summary, "events": tracer.events}

    ai_message = store.add_message(session_id, "ai", answer, run=summary)
    return SendMessageResponse(user_message=user_message, ai_message=ai_message, run=summary)


@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket):
    """Canlı (streaming) STT: tarayıcı 16 kHz mono float32 PCM akıtır; konuşurken
    deşifre edilen metin geri akar.

    Her ikili (binary) mesaj bir PCM parçasıdır. Sunucu ``{committed, interim, final}``
    JSON'ları yollar: ``interim`` o an konuşulan cümlenin geçici hâli, ``final=true``
    bir cümlenin sabitlendiğini bildirir.
    """
    await ws.accept()
    import numpy as np

    from agent_core.stt_stream import LiveTranscriber

    tr = LiveTranscriber()
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await ws.receive_bytes()
            pcm = np.frombuffer(data, dtype=np.float32)
            # Deşifre bloklayıcı (GPU/CPU); event loop'u kilitlememek için thread'e ver.
            updates = await loop.run_in_executor(None, tr.add_audio, pcm)
            for upd in updates:
                await ws.send_json(upd)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        try:
            await ws.send_json(tr.finalize())
        except Exception:
            pass


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Bir koşunun tam olay akışı (log dosyasının yapılandırılmış hâli)."""
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Koşu bulunamadı: {run_id}")
    return run


@app.get("/api/artifacts/{filename}")
def get_artifact(filename: str):
    """Ajanın ürettiği görseli servis eder.

    Yalnızca dosya ADI kabul edilir ve sabit dizinle birleştirilip çözümlenir;
    yol dizinin dışına çıkıyorsa reddedilir (path traversal'a kapalı).
    """
    target = (CHART_DIR / filename).resolve()
    if not str(target).startswith(str(CHART_DIR.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")
    return FileResponse(target)


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    """Sesli çıktı modunda üretilen wav parçasını servis eder.

    Görsellerle aynı güvenlik: yalnızca dosya ADI kabul edilir, sabit dizinle
    birleştirilip çözümlenir; dizin dışına çıkan yol reddedilir (path traversal'a
    kapalı).
    """
    target = (AUDIO_DIR / filename).resolve()
    if not str(target).startswith(str(AUDIO_DIR.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="Ses bulunamadı")
    return FileResponse(target, media_type="audio/wav")
