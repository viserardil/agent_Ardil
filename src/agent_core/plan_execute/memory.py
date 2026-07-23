"""Plan-Execute'un kısa süreli hafızası — bir koşu boyunca ADIMLARIN çalışma belleği.

KAPSAM: Bu modül Plan-Execute'a özgüdür; kayıt birimi bir "plan adımı"dır. ReAct
şeridinde plan adımı diye bir kavram yok (tek bir serbest döngü var), o yüzden
burada durur. Mesajlardan araç çağrısı ve dosya çıkarma işi ise iki şeritte de
aynı olduğu için agent_core.messages'a taşındı.

NEDEN VAR: Executor bir tool-calling alt-ajanıdır; bir adım içinde birden çok araç
çağırır. Eğer yalnızca alt-ajanın SON mesajı saklanırsa aradaki araç çıktıları
kaybolur. O zaman replanner, adımın gerçekte ne ürettiğini göremez; "bu iş
yapılmamış" sanıp aynı adımı yeniden plana koyar → sonsuz replan döngüsü, boşa
token ve süre.

Kırpma politikası: metin çıktıları kırpılabilir, ama ARTEFAKT (üretilen dosya
yolları) ASLA kırpılmaz — döngünün ikinci sebebi, .png yolunun kırpmaya denk
gelip kaybolmasıydı.
"""

from __future__ import annotations

from typing import Any, List

from typing_extensions import TypedDict

from agent_core.messages import ToolCall, extract_artifacts, extract_tool_calls


class StepRecord(TypedDict):
    """Tamamlanmış bir plan adımının tam kaydı."""

    step: str
    tool_calls: List[ToolCall]
    result: str
    artifacts: List[str]


def record_step(step: str, messages: List[Any]) -> StepRecord:
    """Bir adımın alt-ajan koşusunu tam kayda çevirir."""
    tool_calls = extract_tool_calls(messages)
    result = str(messages[-1].content) if messages else ""
    return {
        "step": step,
        "tool_calls": tool_calls,
        "result": result,
        "artifacts": extract_artifacts(tool_calls, result),
    }


def _clip(text: Any, limit: int) -> str:
    """Uzun metni kırpar; kırpınca ham uzunluğu işaretler (kayıp görünür olsun)."""
    text = str(text if text is not None else "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[kesildi, ham {len(text)} kar]"


def render_steps(
    memory: List[StepRecord],
    *,
    result_limit: int = 800,
    tool_output_limit: int = 0,
) -> str:
    """Hafızayı prompt'a gömülecek metne çevirir.

    tool_output_limit > 0 ise araç çağrıları da (girdi/çıktı ile) yazılır — replanner
    için şart: kararını alt-ajanın özetine değil, HAM araç çıktısına dayandırsın.
    0 ise yalnızca adım sonuçları yazılır (executor'ın bağlamı için yeterli, ucuz).
    Artefaktlar her iki modda da kırpılmadan listelenir.
    """
    if not memory:
        return ""

    blocks: List[str] = []
    for index, record in enumerate(memory, 1):
        lines = [f"{index}. ADIM: {record['step']}"]

        if tool_output_limit > 0 and record["tool_calls"]:
            for call in record["tool_calls"]:
                lines.append(f"   • araç: {call['tool']}({_clip(call['input'], 200)})")
                lines.append(f"     çıktı: {_clip(call['output'], tool_output_limit)}")

        lines.append(f"   → sonuç: {_clip(record['result'], result_limit)}")

        if record["artifacts"]:
            # Kırpma YOK: yolun tam olması, replanner'ın "grafik üretildi mi?"
            # kararını doğru vermesinin ön şartı.
            lines.append(f"   → üretilen dosyalar: {', '.join(record['artifacts'])}")

        blocks.append("\n".join(lines))

    return "\n".join(blocks)


def all_artifacts(memory: List[StepRecord]) -> List[str]:
    """Koşu boyunca üretilmiş tüm dosya yolları (tekilleştirilmiş, sırası korunur)."""
    found: List[str] = []
    for record in memory:
        for path in record["artifacts"]:
            if path not in found:
                found.append(path)
    return found
