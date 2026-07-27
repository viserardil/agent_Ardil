"""Triyaj düğümü."""

from __future__ import annotations

from functools import lru_cache

from agent_core.config import get_triage_llm, structured_output_kwargs
from agent_core.history import to_messages
from agent_core.tools import render_tool_descriptions, resolve_tool_names
from agent_core.tracing import get_tracer
from agent_core.triage.prompts import TRIAGE_PROMPT
from agent_core.triage.schemas import TriageDecision


@lru_cache(maxsize=1)
def _get_triage():
    """Triyaj zinciri (tembel kurulur, tek örnek)."""
    return TRIAGE_PROMPT | get_triage_llm().with_structured_output(
        TriageDecision, **structured_output_kwargs()
    )


def triage_step(state, config=None) -> dict:
    """İsteği bir şeride yönlendirir; direct ise cevabı da burada üretir.

    direct'in TEK ÇAĞRI olması bilinçli: triyaj modeli şeride karar verirken cevabı
    da aynı yanıtta döndürür. "Nasılsın" gibi bir mesaj için ikinci bir LLM turu
    beklemek, ses girişli bir sistemde konuşma akışını bozar.

    Araç adları registry'ye karşı süzülür: model katalogta olmayan bir ad uydurursa
    sessizce elenir (bkz. resolve_tool_names).
    """
    tracer = get_tracer(config)
    if tracer:
        tracer.phase_start("triyaj")

    decision = _get_triage().invoke(
        {
            # Geçmiş turlar + güncel mesaj birlikte verilir: "onu MSFT için de yap"
            # gibi terse bir devam isteği ancak önceki konuşmayla doğru sınıflanır.
            "messages": to_messages(state.get("history"), state["input"]),
            # Triaj araç SEÇER; girdi formatına gerek yok -> kısa katalog (token tasarrufu).
            "tool_catalog": render_tool_descriptions(brief=True),
        },
        config=config,
    )

    update = {
        "lane": decision.lane,
        "domain": decision.domain,
        "reason": decision.reason,
        "tools": list(resolve_tool_names(decision.tools)) if decision.tools else [],
    }

    if decision.lane == "direct":
        # Emniyet: model direct dedi ama cevabı yazmadıysa akış cevapsız bitmesin.
        update["response"] = (
            decision.answer.strip()
            or "Bu isteğe doğrudan cevap veremedim, biraz daha açar mısın?"
        )
        update["tools"] = []

    if tracer:
        tracer.triage_decision(
            {
                "lane": update["lane"],
                "domain": update["domain"],
                "reason": update["reason"],
                "tools": update["tools"],
            }
        )
        tracer.phase_end("triyaj")

    return update


def route(state) -> str:
    """Koşullu kenar: triyajın seçtiği şerit."""
    return state["lane"]
