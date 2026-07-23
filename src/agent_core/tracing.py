"""Koşu izleme (tracing) — her kullanıcı mesajının arka planda ne yaptığının kaydı.

TASARIM: Frontend'e yalnızca NİHAİ CEVAP döner; ara süreç oraya akmaz. Ama sürecin
tamamı bir log dosyasına OLAY OLDUĞU AN yazılır ve her yazımdan sonra flush edilir,
böylece koşu sürerken `tail -f` ile canlı izlenebilir.

İki kaynaktan besleniyor:
  1. LangChain callback'leri (BaseCallbackHandler) — grafiğin İÇİNDEKİ her LLM ve
     araç çağrısını otomatik yakalar: model, girdi mesajları, çıktı, token sayıları,
     süre. İç içe alt-grafiklere de iner.
  2. Açık çağrılar — triyaj kararı, şerit başlangıcı, düğüm geçişleri gibi bizim
     bildiğimiz, LangChain'in bilmediği olaylar.

Yapılandırılmış olaylar ayrıca bellekte tutulur; API bunları JSON olarak sunabilir.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

# Log satırlarında uzun metinler bu uzunlukta kırpılır. Amaç dosyayı okunabilir
# tutmak; ham hâli yapılandırılmış olayda (events) tam olarak saklanır.
_LOG_TEXT_LIMIT = 1500

_SEP = "═" * 78
_SUB = "─" * 78


def get_tracer(config) -> Optional["RunTracer"]:
    """Düğümlere geçen config'ten izleyiciyi çıkarır.

    İzleme ZORUNLU değildir: CLI'dan ya da testten çalıştırıldığında tracer
    olmayabilir, o zaman None döner ve düğümler sessizce çalışır.
    """
    return ((config or {}).get("configurable") or {}).get("tracer")


def _clip(text: Any, limit: int = _LOG_TEXT_LIMIT) -> str:
    text = str(text if text is not None else "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [kesildi, ham {len(text)} karakter]"


def _indent(text: str, prefix: str) -> str:
    """Çok satırlı metni log hizasına sokar.

    Sondaki boş satırlar atılır: model çıktıları sık sık satır sonuyla bitiyor ve
    bu, logda içi boş bir kenarlık satırı bırakıyordu.
    """
    return "\n".join(prefix + line for line in str(text).rstrip().splitlines())


def _fmt_args(args: Dict[str, Any]) -> str:
    """Araç argümanlarını okunur tek satıra çevirir (reason ayıklandıktan sonra).

    Tek anahtar kalırsa (tipik: tool_input) doğrudan değeri yaz; gürültü olmasın.
    """
    if not args:
        return ""
    if len(args) == 1:
        return str(next(iter(args.values())))
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)


class RunTracer(BaseCallbackHandler):
    """Tek bir koşunun (bir kullanıcı mesajı) izleyicisi.

    Aynı anda birden çok koşu olabileceği için dosyaya yazım kilitle korunur;
    her koşu kendi dosyasına yazar ama session log'u paylaşılabilir.
    """

    def __init__(self, session_id: str, user_input: str, log_dir: Path) -> None:
        self.session_id = session_id
        self.run_id = uuid.uuid4().hex[:8]
        self.user_input = user_input
        self.started_at = time.perf_counter()

        self.events: List[Dict[str, Any]] = []
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.lane: str = ""
        self.artifacts: List[str] = []

        # Süre ölçümü için açık kalan işlemler: run_id -> (etiket, başlangıç)
        self._open: Dict[str, tuple] = {}
        self._lane_started: Optional[float] = None
        self._lane_tokens_at_start = 0
        self._iteration = 0  # aşama içindeki düşün→eylem→gözlem tur sayacı
        self._pending_tools = 0  # turda gözlemi henüz gelmemiş araç sayısı
        self._phase_totals: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{session_id}.log"
        self._write_header()

    # --- yazım ------------------------------------------------------------

    def _ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    def _write(self, text: str) -> None:
        """Log dosyasına yazar ve HEMEN flush eder (canlı izlenebilsin diye)."""
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
                handle.flush()

    def _line(self, text: str) -> None:
        self._write(f"[{self._ms():>7,} ms] {text}")

    def _event(self, kind: str, **payload: Any) -> None:
        """Yapılandırılmış olayı belleğe ekler (API buradan okur)."""
        self.events.append(
            {"t_ms": self._ms(), "kind": kind, **payload}
        )

    def _write_header(self) -> None:
        self._write("")
        self._write(_SEP)
        self._write(
            f"KOŞU {self.run_id} · oturum {self.session_id} · "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._write(f"KULLANICI: {_clip(self.user_input, 500)}")
        self._write(_SUB)

    # --- açık (bizim tetiklediğimiz) olaylar --------------------------------

    def triage_decision(self, decision: Dict[str, Any]) -> None:
        """Triyajın kararı — sistemin ilk ve en belirleyici adımı."""
        self.lane = decision.get("lane", "")
        tools = decision.get("tools") or []
        self._line(f"◆ KARAR → şerit={self.lane} · alan={decision.get('domain', '')}")
        self._write(f"{' ' * 14}araçlar : {', '.join(tools) if tools else '(yok)'}")
        self._write(f"{' ' * 14}gerekçe : {decision.get('reason', '')}")
        self._event("triage", **decision)

    def phase_start(self, name: str) -> None:
        """Bir aşama (triyaj / şerit) başladı."""
        self._lane_started = time.perf_counter()
        self._lane_tokens_at_start = self.tokens_in + self.tokens_out
        # Döngü sayacı her aşamada sıfırlanır: "DÖNGÜ 1" o şeridin ilk turu demek.
        self._iteration = 0
        self._line(f"▶ {name.upper()} başladı")
        self._event("phase_start", name=name)

    def phase_end(self, name: str) -> None:
        """Aşama bitti; süresini ve harcadığı token'ı özete işler."""
        elapsed = (time.perf_counter() - (self._lane_started or time.perf_counter()))
        tokens = (self.tokens_in + self.tokens_out) - self._lane_tokens_at_start
        self._phase_totals[name] = {"sn": elapsed, "token": tokens}
        self._line(f"◀ {name.upper()} bitti ({elapsed:.2f} sn, {tokens:,} token)")
        self._event("phase_end", name=name, seconds=elapsed, tokens=tokens)

    def node(self, name: str, update: Dict[str, Any]) -> None:
        """Şerit içindeki bir düğüm çalıştı ve state'i güncelledi."""
        self._line(f"  ● düğüm: {name}")

        if update.get("plan"):
            for index, step in enumerate(update["plan"], 1):
                self._write(f"{' ' * 16}plan {index}. {step}")

        for record in update.get("memory") or []:
            self._write(f"{' ' * 16}✓ adım: {record['step']}")
            self._write(f"{' ' * 16}  sonuç: {_clip(record['result'], 600)}")
            if record.get("artifacts"):
                self._write(f"{' ' * 16}  dosya: {', '.join(record['artifacts'])}")
            # Plan-Execute'ta üretilen dosyalar adım KAYITLARININ içinde gelir;
            # ReAct'te ise üst düzey 'artifacts' alanında. İkisini de topla.
            self.add_artifacts(record.get("artifacts") or [])

        self.add_artifacts(update.get("artifacts") or [])
        self._event("node", name=name, keys=sorted(update))

    def add_artifacts(self, paths) -> None:
        """Üretilen dosya yollarını tekilleştirerek biriktirir."""
        for path in paths or []:
            if path not in self.artifacts:
                self.artifacts.append(path)

    # --- LangChain callback'leri -------------------------------------------

    @staticmethod
    def _model_name(serialized: Optional[dict], kwargs: dict) -> str:
        meta = kwargs.get("metadata") or {}
        for candidate in (
            meta.get("ls_model_name"),
            (serialized or {}).get("kwargs", {}).get("model_name"),
            (serialized or {}).get("kwargs", {}).get("model"),
        ):
            if candidate:
                return str(candidate)
        return "?"

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        """Sohbet modeli çağrısı başladı — GİRDİ mesajlarını olduğu gibi kaydeder."""
        model = self._model_name(serialized, kwargs)
        self._open[str(run_id)] = (model, time.perf_counter())

        flat = messages[0] if messages else []
        self._line(f"  ↗ LLM çağrısı başladı · {model}")
        for message in flat:
            role = getattr(message, "type", "?")
            content = getattr(message, "content", "")
            self._write(f"{' ' * 16}├─ GİRDİ [{role}]:")
            self._write(_indent(_clip(content), " " * 19))

        self._event(
            "llm_start",
            model=model,
            messages=[
                {"role": getattr(m, "type", "?"), "content": str(getattr(m, "content", ""))}
                for m in flat
            ],
        )

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        """Düz (sohbet olmayan) LLM çağrısı. Bizde nadir; yine de kayıt altına alınır."""
        model = self._model_name(serialized, kwargs)
        self._open[str(run_id)] = (model, time.perf_counter())
        self._line(f"  ↗ LLM çağrısı başladı · {model}")
        for prompt in prompts:
            self._write(_indent(_clip(prompt), " " * 19))
        self._event("llm_start", model=model, messages=[{"role": "prompt", "content": p} for p in prompts])

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        """LLM cevabı geldi — DÜŞÜNCE / EYLEM ayrımı burada yapılır.

        Tool-calling ajanlarında (ReAct şeridi ve Plan-Execute'un executor'ı) bir
        döngü turu şudur: model düşünür ve bir araç çağırır (bu metot), araç
        çalışır ve gözlem döner (on_tool_end). Araç çağrısı OLMAYAN cevap ise
        döngünün bittiği, nihai metnin üretildiği turdur.

        Klasik ReAct'teki gibi 'Thought:' diye bir metin ayrıştırması YOK — model
        native tool-calling kullanıyor. Bu yüzden düşünce, cevabın metin kısmıdır;
        model hiç metin yazmadan doğrudan araç çağırırsa bunu açıkça belirtiriz
        (uydurma bir düşünce yazmaktansa yokluğunu göstermek doğru).
        """
        model, started = self._open.pop(str(run_id), ("?", time.perf_counter()))
        elapsed = time.perf_counter() - started
        self.llm_calls += 1

        text, tool_calls, usage = self._read_generation(response)
        prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        self.tokens_in += prompt_tokens
        self.tokens_out += completion_tokens

        total = prompt_tokens + completion_tokens
        cost = (
            f"{model} · {elapsed:.2f} sn · {total:,} token "
            f"(girdi {prompt_tokens:,} / çıktı {completion_tokens:,})"
        )

        if tool_calls:
            self._iteration += 1
            self._line(f"  ┌─ DÖNGÜ {self._iteration} · {cost}")
            # Model ayrıca serbest metin de yazdıysa (nadir) onu da göster.
            if text:
                self._write(f"{' ' * 16}│ NOT:")
                self._write(_indent(_clip(text), " " * 16 + "│   "))
            # Model bir turda BİRDEN ÇOK aracı paralel çağırabiliyor. Turun kapanış
            # çizgisi, bekleyen gözlemlerin sonuncusu geldiğinde yazılsın diye sayılır.
            self._pending_tools = len(tool_calls)
            for call in tool_calls:
                reason = call.get("reason") or "(gerekçe yok)"
                args = _fmt_args(call["args"])
                self._write(f"{' ' * 16}│ DÜŞÜNCE: {_clip(reason, 400)}")
                self._write(f"{' ' * 16}│ EYLEM  : {call['name']}({_clip(args, 300)})")
        else:
            self._line(f"  ✔ SONUÇ · {cost}")
            self._write(_indent(_clip(text) or "(boş)", " " * 16 + "  "))

        self._event(
            "llm_end",
            model=model,
            seconds=elapsed,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            output=text,
            tool_calls=tool_calls,
            iteration=self._iteration if tool_calls else None,
        )

    @staticmethod
    def _read_generation(response) -> tuple:
        """LLM cevabından metni, araç çağrılarını ve token kullanımını çıkarır.

        Üçü ayrı döner çünkü log bunları FARKLI şeyler olarak gösteriyor:
        metin = DÜŞÜNCE, araç çağrıları = EYLEM. Bir tool-calling ajanında bir
        cevap ikisini birden içerebilir (model hem gerekçe yazıp hem araç çağırır)
        ya da yalnızca birini içerir.

        Token bilgisi sağlayıcıya göre iki ayrı yerde bulunabiliyor: mesajın
        usage_metadata'sı (yeni) ya da llm_output['token_usage'] (klasik).
        """
        text = ""
        tool_calls: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            text = (getattr(generation, "text", "") or "").strip()
            if message is not None:
                usage = dict(getattr(message, "usage_metadata", None) or {})
                for call in getattr(message, "tool_calls", None) or []:
                    # reason zorunlu şema alanı: DÜŞÜNCE olarak ayrı gösterilir,
                    # gerçek argümanlardan (EYLEM) ayıklanır.
                    args = dict(call.get("args") or {})
                    reason = str(args.pop("reason", "") or "")
                    tool_calls.append(
                        {"name": call.get("name", "?"), "args": args, "reason": reason}
                    )
        except (AttributeError, IndexError, TypeError):
            pass

        if not usage:
            usage = dict((getattr(response, "llm_output", None) or {}).get("token_usage") or {})
        return text, tool_calls, usage

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        model, started = self._open.pop(str(run_id), ("?", time.perf_counter()))
        self._line(f"  ✖ LLM HATASI · {model} · {time.perf_counter() - started:.2f} sn · {error}")
        self._event("llm_error", model=model, error=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs) -> None:
        """Araç çalışmaya başladı.

        Loga ayrı satır YAZMAZ: aracın adı ve girdisi zaten bir satır yukarıda
        EYLEM olarak görünüyor, tekrarı döngüyü okumayı zorlaştırır. Yapılandırılmış
        olay yine de kaydedilir.
        """
        name = (serialized or {}).get("name", "?")
        self._open[str(run_id)] = (name, time.perf_counter())
        self._event("tool_start", tool=name, input=str(input_str))

    def on_tool_end(self, output, *, run_id=None, **kwargs) -> None:
        """Araç sonucu geldi — döngünün GÖZLEM adımı."""
        name, started = self._open.pop(str(run_id), ("?", time.perf_counter()))
        elapsed = time.perf_counter() - started
        self.tool_calls += 1
        text = str(getattr(output, "content", output))

        self._line(f"  │ GÖZLEM : {name} → {elapsed:.2f} sn · {len(text)} karakter")
        self._write(_indent(_clip(text, 800), " " * 16 + "│   "))

        self._pending_tools = max(0, self._pending_tools - 1)
        if self._pending_tools == 0:  # turun tüm gözlemleri geldi, çerçeveyi kapat
            self._write(f"{' ' * 16}└{'─' * 40}")

        self._event("tool_end", tool=name, seconds=elapsed, output=text)

    def on_tool_error(self, error, *, run_id=None, **kwargs) -> None:
        name, started = self._open.pop(str(run_id), ("?", time.perf_counter()))
        self._line(f"  ✖ ARAÇ HATASI {name} · {time.perf_counter() - started:.2f} sn · {error}")
        self._event("tool_error", tool=name, error=str(error))

    # --- kapanış -----------------------------------------------------------

    def finish(self, response: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Koşuyu kapatır, özeti loga yazar ve API'nin döneceği özeti üretir."""
        total_seconds = time.perf_counter() - self.started_at
        total_tokens = self.tokens_in + self.tokens_out

        self._write(_SUB)
        if error:
            self._write(f"HATA: {error}")
        else:
            self._write("CEVAP:")
            self._write(_indent(_clip(response, 2000), " " * 2))
        self._write(_SUB)
        self._write(
            f"ÖZET · {total_seconds:.2f} sn · {self.llm_calls} LLM çağrısı · "
            f"{total_tokens:,} token (girdi {self.tokens_in:,} / çıktı {self.tokens_out:,}) · "
            f"{self.tool_calls} araç çağrısı"
        )
        for name, totals in self._phase_totals.items():
            self._write(
                f"  {name:<14}: {totals['sn']:.2f} sn · {int(totals['token']):,} token"
            )
        if self.artifacts:
            self._write(f"  üretilen dosyalar: {', '.join(self.artifacts)}")
        self._write(_SEP)

        summary = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "lane": self.lane,
            "seconds": round(total_seconds, 3),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.tokens_in,
            "output_tokens": self.tokens_out,
            "total_tokens": total_tokens,
            "artifacts": list(self.artifacts),
            "phases": {k: {"seconds": round(v["sn"], 3), "tokens": int(v["token"])}
                       for k, v in self._phase_totals.items()},
            "error": error,
            "log_file": str(self.log_path),
        }
        self._event("run_end", **summary)
        return summary
