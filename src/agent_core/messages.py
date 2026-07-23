"""LangChain mesaj listelerinden bilgi çıkarma — şeritlerden bağımsız yardımcılar.

Hem Plan-Execute'un executor'ı hem ReAct şeridi bir tool-calling ajanı çalıştırır ve
ikisinin de aynı iki soruya cevabı gerekir: "hangi araçlar hangi girdiyle çağrıldı,
ne döndürdü?" ve "bu koşuda hangi dosyalar üretildi?".

Bu modül, ReAct şeridi yazılırken plan_execute/memory.py'den ayrıldı; daha önce
orada duruyordu ama kayıt birimi "plan adımı" olmayan her şey aslında şeride özgü
değildi.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from typing_extensions import TypedDict


class ToolCall(TypedDict):
    """Bir ajan koşusu içinde yapılmış tek araç çağrısı.

    ``reason`` modelin o aracı neden seçtiğidir (araç şemasındaki zorunlu alan).
    input'tan AYRI tutulur: input aracın gerçek girdisidir, reason ise gözlem
    amaçlı gerekçe — ikisini karıştırmak logu ve hafızayı kirletir.
    """

    tool: str
    input: str
    output: str
    reason: str


# Araç çıktısında geçen üretilmiş dosya yolları. plot_chart/visualize_data bir yol
# döndürür ve bu yolun nihai cevaba kadar SAĞLAM gitmesi gerekir; bu yüzden yollar
# metinden ayrı bir alana çıkarılır ve hiçbir kırpmaya tabi tutulmaz.
_ARTIFACT_RE = re.compile(
    r"[^\s\"'(),;]+\.(?:png|jpg|jpeg|svg|pdf|csv|xlsx)", re.IGNORECASE
)


def _format_args(args: Dict[str, Any]) -> str:
    """Araç argümanlarını okunur tek satıra çevirir.

    Araçlarımız tek string girdi alıyor (StructuredTool sarmalayıcısı), o yüzden
    tek anahtarlı sözlükte doğrudan değeri yazmak gürültüyü azaltır.
    """
    if not args:
        return ""
    if len(args) == 1:
        (value,) = args.values()
        return str(value)
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)


def extract_tool_calls(messages: List[Any]) -> List[ToolCall]:
    """Ajanın mesaj geçmişinden (araç, girdi, çıktı) üçlülerini çıkarır.

    AIMessage'lar çağrının ADINI ve ARGÜMANLARINI, ToolMessage'lar ÇIKTISINI taşır;
    ikisi tool_call_id üzerinden eşleştirilir. Yalnızca son mesaja bakmak araç
    çıktılarını kaybettirir — Plan-Execute'ta replanner'ın tamamlanmış adımı
    yeniden planlamasının (döngü) sebebi buydu.
    """
    args_by_id: Dict[str, tuple] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            args_by_id[call.get("id")] = (call.get("name", ""), call.get("args") or {})

    calls: List[ToolCall] = []
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        call_id = getattr(message, "tool_call_id", None)
        name, args = args_by_id.get(call_id, (getattr(message, "name", "") or "", {}))
        # reason gözlem alanıdır, aracın gerçek girdisi değil: input'a karışmasın
        # diye ayrılır (bkz. _AgentToolInput).
        args = dict(args)
        reason = str(args.pop("reason", "") or "")
        calls.append(
            {
                "tool": getattr(message, "name", "") or name,
                "input": _format_args(args),
                "output": str(message.content),
                "reason": reason,
            }
        )
    return calls


def extract_artifacts(tool_calls: List[ToolCall], result: str) -> List[str]:
    """Araç çıktılarında ve nihai metinde geçen üretilmiş dosya yolları (tekilleştirilmiş)."""
    found: List[str] = []
    haystack = [call["output"] for call in tool_calls] + [result or ""]
    for text in haystack:
        for match in _ARTIFACT_RE.findall(text):
            if match not in found:
                found.append(match)
    return found
