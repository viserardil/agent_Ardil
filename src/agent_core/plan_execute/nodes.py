"""Plan-Execute grafiğinin düğümleri (planner / executor / replanner)."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

# LangChain'in hazır tool-calling ajanı. Adı "ReAct" olan langgraph.prebuilt.
# create_react_agent LangGraph v1.0'da deprecate edildi (v2.0'da kalkacak) ve
# buraya taşındı; tek çağrı yerimiz varken geçtik.
from langchain.agents import create_agent

from agent_core.config import get_llm, structured_output_kwargs
from agent_core.history import to_messages
from agent_core.plan_execute.memory import all_artifacts, record_step, render_steps
from agent_core.plan_execute.prompts import (
    EXECUTOR_PROMPT,
    PLANNER_PROMPT,
    REPLANNER_PROMPT,
)
from agent_core.plan_execute.schemas import Act, Plan, Response
from agent_core.plan_execute.state import PlanExecute
from agent_core.tools import get_tools, resolve_tool_names

# LLM/ajan nesneleri TEMBEL (lazy) kurulur: modül import edilirken değil, ilk
# kullanımda. Böylece paketi API anahtarı olmadan import etmek çökmez.

# Hafıza prompt'lara gömülürken uygulanan kırpma sınırları (KARAKTER, token değil;
# kabaca ÷4 token eder). Hafızada çıktılar TAM saklanır — kırpma yalnızca render
# anında olur, dolayısıyla veri kaybolmaz, sadece o prompt'ta görünmez.
#
# Değerler ölçümle seçildi. Araçların gerçek çıktı boyutları (AAPL, canlı koşu):
#   get_historical_stock_prices 666 · get_balance_sheet 615 · get_company_info 585
#   get_income_statements 554 · get_cash_flow 466 · diğerleri <300
# Araçlar zaten kendi içinde özetliyor (son 10 kayıt / 4 yıl / 3 dönem); en uzun
# araç olan web_search bile ~2000'i geçmiyor. Dolayısıyla 6000'lik sınırlar mevcut
# araç setinde HİÇBİR ŞEYİ kırpmaz — kırpma burada aktif bir bütçe değil, ileride
# çok uzun çıktılı bir araç eklenirse prompt'un patlamasını önleyen emniyet supabı.
#
# Sınırların ayrı ayrı durmasının sebebi, ileride farklılaşmaları gerekebileceği:
#   - Executor yalnızca BAĞLAMA ihtiyaç duyar (ne üzerinde çalışılıyor, şimdiye kadar
#     ne bulunmuştu); daralması gerekirse ilk buradan daraltılmalı.
#   - Replanner hem kararını HAM araç çıktısına dayandırır (alt-ajanın özeti eksik
#     olduğunda tamamlanmış adımı yeniden planlamasının sebebi buydu) hem de NİHAİ
#     CEVABI bu metinlerden kurar; buradaki kırpma doğrudan kullanıcının gördüğü
#     cevaptan eksiltir, en son daraltılacak yer burasıdır.
# Artefakt yolları hiçbir durumda kırpılmaz (bkz. plan_execute.memory).
_EXECUTOR_RESULT_LIMIT = 6000
_REPLANNER_RESULT_LIMIT = 6000
_REPLANNER_TOOL_OUTPUT_LIMIT = 6000


def _resolve_tools(state: PlanExecute) -> Tuple[str, ...]:
    """Bu istekte kullanılacak araç adları (triyajın ön-seçimi, yoksa tümü)."""
    return resolve_tool_names(state.get("tools"))


@lru_cache(maxsize=32)
def _get_executor_agent(tool_names: Tuple[str, ...]):
    """Adımları yürüten ReAct alt-ajanı.

    Cache araç KÜMESİNE göre anahtarlanır: triyaj her istekte farklı bir alt küme
    seçebildiği için tek bir global ajan yeterli değil, ama her istekte sıfırdan
    kurmak da gereksiz. Aynı kümeyle gelen istekler aynı ajanı paylaşır.
    """
    return create_agent(
        get_llm(), get_tools(list(tool_names)), system_prompt=EXECUTOR_PROMPT
    )


# Structured output VARSAYILAN (json_schema) yöntemle yapılır. Neden:
#   - Replanner'ın Act şeması Union[Response, Plan] içerir; json_schema bu union'ı
#     üretim anında ZORLAR ve güvenilir parse eder. function_calling ise union'da
#     bazen 'action'a düz string koyup pydantic ValidationError'a yol açar.
#   - Sağlayıcı json_schema desteklemiyorsa LLM_STRUCTURED_METHOD=function_calling.
@lru_cache(maxsize=1)
def _get_planner():
    """Görevi plan adımlarına bölen zincir."""
    return PLANNER_PROMPT | get_llm().with_structured_output(Plan, **structured_output_kwargs())


@lru_cache(maxsize=1)
def _get_replanner():
    """Devam mı bitiş mi kararını veren replanner zinciri."""
    return REPLANNER_PROMPT | get_llm().with_structured_output(Act, **structured_output_kwargs())


def plan_step(state: PlanExecute) -> dict:
    """Görevi yürütülebilir adımlara böler.

    Eski projedeki "araç gerekli mi?" triyajı burada YOK — o karar üstteki triyaj
    katmanında verilip görev bu şeride yönlendirildi. Buraya gelen her görev
    araç gerektiriyor kabul edilir.
    """
    tool_names = _resolve_tools(state)
    plan = _get_planner().invoke(
        {
            # Geçmiş turlar + güncel istek: planner referansı ("onu MSFT için de
            # yap") burada somut adımlara çevirir; downstream düğümler geçmişi
            # görmese de çalışabilsin diye çözüm bu erken noktada yapılır.
            "messages": to_messages(state.get("history"), state["input"]),
            "tool_names": ", ".join(tool_names),
        }
    )
    # Emniyet: adım boş dönerse görevi tek adım yap.
    return {"plan": plan.steps or [state["input"]]}


def execute_step(state: PlanExecute) -> dict:
    """Plandaki ilk adımı ReAct alt-ajanıyla yürütür.

    Alt-ajanın turlar arası hafızası YOKTUR; her çağrıda sıfırdan kurulur. İhtiyaç
    duyacağı bağlam bu yüzden prompt'a açıkça konur: kullanıcının asıl isteği +
    kısa süreli hafızadaki tamamlanmış adımlar. Aksi halde "elde edilen veriyle
    görseli üret" gibi bir adımda neyin kastedildiğini bilemez ve kullanıcıya geri
    soru sorar.

    Adımın çıktısı TAM olarak kaydedilir (araç çağrıları + ham çıktılar + üretilen
    dosyalar) — yalnızca son mesajı saklamak, replanner'ın adımı tamamlanmış
    sayamamasına ve döngüye yol açıyordu.
    """
    plan = state["plan"]
    if not plan:  # emniyet: boş plan gelirse patlama, replanner karar versin
        return {}
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]

    parts = [
        f"Kullanıcının asıl isteği: {state['input']}",
        "",
        f"Kalan plan:\n{plan_str}",
    ]

    memory = state.get("memory") or []
    if memory:
        parts += [
            "",
            "Şimdiye kadar tamamlanan adımlar:",
            render_steps(memory, result_limit=_EXECUTOR_RESULT_LIMIT),
        ]
        artifacts = all_artifacts(memory)
        if artifacts:
            parts += ["", f"Bu koşuda üretilmiş dosyalar: {', '.join(artifacts)}"]

    parts += ["", f"Şimdi şu adımı yürüt: {task}"]

    agent = _get_executor_agent(_resolve_tools(state))
    agent_response = agent.invoke({"messages": [("user", "\n".join(parts))]})
    return {"memory": [record_step(task, agent_response["messages"])]}


def replan_step(state: PlanExecute) -> dict:
    """Sonuçlara göre planı günceller veya nihai cevabı üretir.

    Kararını alt-ajanın ÖZETİNE değil, hafızadaki ham araç çıktılarına dayandırır;
    üretilen dosyalar ayrıca açıkça listelenir. İkisi de aynı amaca hizmet eder:
    tamamlanmış bir adımın yeniden planlanmasını (replan döngüsü) engellemek.

    Nihai cevap (Response) üretmek bir "bitir" kararıdır, plan revizyonu değildir;
    o yüzden replan_count YALNIZCA yeni bir Plan döndürüldüğünde artırılır.
    """
    memory = state.get("memory") or []
    artifacts = all_artifacts(memory)

    output = _get_replanner().invoke(
        {
            "input": state["input"],
            "plan": state.get("plan") or [],
            "completed": render_steps(
                memory,
                result_limit=_REPLANNER_RESULT_LIMIT,
                tool_output_limit=_REPLANNER_TOOL_OUTPUT_LIMIT,
            )
            or "(henüz adım tamamlanmadı)",
            "artifacts": ", ".join(artifacts) if artifacts else "(yok)",
            "tool_names": ", ".join(_resolve_tools(state)),
        }
    )

    if isinstance(output.action, Response):
        return {"response": output.action.response}

    steps = output.action.steps
    if not steps:
        # Boş plan = yapacak yeni adım yok. Sonsuz döngü / execute_step'te plan[0]
        # IndexError yerine son adımın sonucunu nihai cevap yap.
        return {"response": memory[-1]["result"] if memory else "Görev tamamlandı."}
    return {"plan": steps, "replan_count": 1}


def should_end(state: PlanExecute) -> str:
    """Koşullu kenar: cevap üretildiyse bitir, değilse yürütmeye devam et."""
    if state.get("response"):
        return "__end__"
    return "executor"
