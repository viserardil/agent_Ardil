"""Sistemin tek giriş kapısı: triyaj + şeritler.

    START → triage ─┬→ (direct)       → END      cevap triyajda üretildi
                    ├→ plan_execute   → END
                    └→ react          → END

Şeritler ayrı grafiklerdir ve kendi state şemalarını taşır. Burada düğüm olarak
sarılmalarının sebebi, dışarıya TEK ve sade bir arayüz vermek: çağıran taraf
{input} verir, {response, artifacts, tool_calls} alır — hangi şeridin çalıştığını
bilmek zorunda değildir. Arayüz (ses/React) bu grafiğe bağlanacak.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent_core.history import Turn
from agent_core.messages import ToolCall
from agent_core.plan_execute import build_plan_execute_graph
from agent_core.plan_execute.memory import all_artifacts
from agent_core.react import build_react_graph
from agent_core.tracing import get_tracer
from agent_core.triage import route, triage_step


class AgentState(TypedDict):
    """Sistemin dışarıya görünen durumu.

    - input:      Kullanıcının isteği (ses tanımadan gelen metin).
    - history:    Bu oturumdaki önceki turlar (kullanıcı mesajları + ajan cevapları).
                  Boşsa hafızasız (ilk mesaj ya da CLI/test). Triyaj ve şeritlere
                  geçmiş bağlamı buradan akar.
    - lane:       Triyajın seçtiği şerit.
    - domain:     Triyajın etiketi (yalnızca gözlem/log).
    - tools:      Triyajın ön-seçtiği araç adları.
    - reason:     Triyajın gerekçesi (yalnızca gözlem/log).
    - response:   Nihai cevap; hangi şerit çalışırsa çalışsın buraya yazılır.
    - tool_calls: Koşuda yapılan araç çağrıları, şeritten bağımsız normalize edilmiş.
    - artifacts:  Koşuda üretilen dosya yolları.
    """

    input: str
    history: List[Turn]
    lane: str
    domain: str
    tools: List[str]
    reason: str
    response: str
    tool_calls: List[ToolCall]
    artifacts: List[str]


# Şerit grafikleri bir kez derlenir; her istekte yeniden kurmak gereksiz.
@lru_cache(maxsize=1)
def _plan_execute():
    return build_plan_execute_graph()


@lru_cache(maxsize=1)
def _react():
    return build_react_graph()


def _run_lane(graph, name: str, state: AgentState, config) -> dict:
    """Bir şerit grafiğini çalıştırıp son durumunu döndürür.

    invoke yerine stream kullanılıyor: şerit İÇİNDEKİ düğüm geçişleri (planner →
    executor → replanner) ancak böyle görünür oluyor ve izleyiciye adım adım
    yazılabiliyor. invoke tek bir sonuç döndürür, aradaki hiçbir şey görünmez.

    Üst config OLDUĞU GİBİ aktarılmaz: içindeki LangGraph iç anahtarları (__pregel_*)
    alt-grafiği iç içe koşu sanıp stream'in çıktı şeklini değiştiriyor — düğüm
    güncellemesi sözlük yerine string olarak geliyordu. Bu yüzden alt-grafiğe yalnızca
    gereken üç şey verilir: özyineleme sınırı, izleyici ve onun callback'i.
    """
    tracer = get_tracer(config)
    if tracer:
        tracer.phase_start(name)

    lane_config = {
        "recursion_limit": 50,
        "callbacks": [tracer] if tracer else [],
        "configurable": {"tracer": tracer},
    }
    final: dict = {}
    # memory alt-grafikte operator.add ile BİRİKEN bir alan; stream ise her düğümün
    # yalnızca kendi deltasını verir. Düz update() ile birleştirmek önceki adımların
    # kayıtlarını ezer (2 adımlık koşuda 1. adım kaybolurdu), o yüzden ayrı toplanır.
    memory: List[dict] = []

    for event in graph.stream(
        {
            "input": state["input"],
            "tools": state.get("tools") or [],
            "history": state.get("history") or [],
        },
        config=lane_config,
    ):
        for node_name, update in event.items():
            if not isinstance(update, dict):  # beklenmedik stream şekli: loga takılma
                continue
            if tracer:
                tracer.node(node_name, update)
            for key, value in update.items():
                if key == "memory" and isinstance(value, list):
                    memory.extend(value)
                else:
                    final[key] = value

    if memory:
        final["memory"] = memory
    if tracer:
        tracer.phase_end(name)
    return final


def _run_plan_execute(state: AgentState, config=None) -> dict:
    """Plan-Execute şeridini çalıştırır ve çıktısını ortak alanlara çevirir."""
    result = _run_lane(_plan_execute(), "plan_execute", state, config)

    # stream'de her düğüm yalnızca KENDİ güncellemesini döndürür; memory ise
    # operator.add ile birikir. Bu yüzden parçaları burada yeniden topluyoruz.
    memory = result.get("memory") or []
    # Adım adım tutulan araç çağrılarını tek listeye düzleştir: dışarıya hangi
    # şeridin çalıştığından bağımsız aynı şekilde görünsün.
    calls: List[ToolCall] = [call for record in memory for call in record["tool_calls"]]
    return {
        "response": result.get("response", ""),
        "tool_calls": calls,
        "artifacts": all_artifacts(memory),
    }


def _run_react(state: AgentState, config=None) -> dict:
    """ReAct şeridini çalıştırır ve çıktısını ortak alanlara çevirir."""
    result = _run_lane(_react(), "react", state, config)
    return {
        "response": result.get("response", ""),
        "tool_calls": result.get("tool_calls") or [],
        "artifacts": result.get("artifacts") or [],
    }


def build_agent_graph():
    """Derlenmiş tam sistemi (triyaj + şeritler) döndürür."""
    workflow = StateGraph(AgentState)

    workflow.add_node("triage", triage_step)
    workflow.add_node("plan_execute", _run_plan_execute)
    workflow.add_node("react", _run_react)

    workflow.add_edge(START, "triage")
    # direct'te cevap triyajda üretildiği için ayrı bir düğüm yok: doğrudan END.
    workflow.add_conditional_edges(
        "triage",
        route,
        {"direct": END, "plan_execute": "plan_execute", "react": "react"},
    )
    workflow.add_edge("plan_execute", END)
    workflow.add_edge("react", END)

    return workflow.compile()
