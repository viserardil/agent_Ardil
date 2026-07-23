"""ReAct şeridinin tek düğümü."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

# Hazır tool-calling ajanı — döngüyü elle yazmıyoruz. langgraph.prebuilt'teki
# create_react_agent LangGraph v1.0'da deprecate edildi, buraya taşındı.
from langchain.agents import create_agent

from agent_core.config import get_llm
from agent_core.history import to_messages
from agent_core.messages import extract_artifacts, extract_tool_calls
from agent_core.react.prompts import REACT_PROMPT
from agent_core.react.state import ReActState
from agent_core.tools import get_tools, resolve_tool_names


@lru_cache(maxsize=32)
def _get_agent(tool_names: Tuple[str, ...]):
    """Görevin tamamını üstlenen tool-calling ajanı.

    Cache araç KÜMESİNE göre anahtarlanır: triyaj her istekte farklı bir alt küme
    seçebilir, ama aynı kümeyle gelen istekler aynı ajanı paylaşsın.
    """
    return create_agent(
        get_llm(), get_tools(list(tool_names)), system_prompt=REACT_PROMPT
    )


def run_agent(state: ReActState) -> dict:
    """Ajanı çalıştırır ve koşuyu dışarı görünür hale getirir.

    Ajanın kendi mesaj geçmişi döngü boyunca zaten taşınıyor; buradaki çıkarım
    karar akışı için değil, gözlemlenebilirlik için: hangi araçlar çağrıldı,
    hangi dosyalar üretildi. Artefaktlar ayrı alanda tutulur ki nihai cevapta
    yol geçmese bile kaybolmasın (arayüz grafiği yine gösterebilsin).
    """
    agent = _get_agent(resolve_tool_names(state.get("tools")))
    # Geçmiş turlar konuşma sırası olarak verilir: ajan referansı ve sürekliliği
    # kendi mesaj geçmişinden çözer.
    result = agent.invoke(
        {"messages": to_messages(state.get("history"), state["input"])}
    )

    messages = result["messages"]
    tool_calls = extract_tool_calls(messages)
    response = str(messages[-1].content) if messages else ""

    return {
        "response": response,
        "tool_calls": tool_calls,
        "artifacts": extract_artifacts(tool_calls, response),
    }
