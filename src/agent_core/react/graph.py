"""ReAct StateGraph'ının kurulumu."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_core.react.nodes import run_agent
from agent_core.react.state import ReActState


def build_react_graph():
    """Derlenmiş ReAct grafiğini döndürür.

    Akış:  START → agent → END

    Tek düğüm, çünkü asıl döngü (model ⇄ tools) hazır ajanın KENDİ içinde dönüyor.
    Peki neden yine de bir StateGraph? Şeritleri triyaj altında aynı arayüzle
    çağırabilmek için: her şerit {input, tools} alır, {response, ...} döndürür.
    Böylece triyaj katmanı şeritler arasında seçim yaparken içlerinin nasıl
    çalıştığını bilmek zorunda kalmaz.
    """
    workflow = StateGraph(ReActState)

    workflow.add_node("agent", run_agent)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()
