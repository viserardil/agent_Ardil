"""ReAct şeridinin durumu (state).

Plan-Execute'un aksine burada plan/adım/replan yok: tek bir serbest döngü görevin
tamamını üstlenir. Bu yüzden state de sade — birikimli (operator.add) alan yok,
tek düğüm her alanı bir kez yazar.
"""

from __future__ import annotations

from typing import List

from typing_extensions import TypedDict

from agent_core.history import Turn
from agent_core.messages import ToolCall


class ReActState(TypedDict):
    """Graf boyunca taşınan durum.

    - input:      Kullanıcının görevi.
    - tools:      Triyajın ön-seçtiği araç ADLARI; boşsa registry'nin tamamı.
    - history:    Session (sohbet) geçmişi. ReAct tek döngü olduğu için ajan bunu
                  baştan sona görür — referansı da, konuşma sürekliliğini de burada
                  çözer.
    - response:   Ajanın nihai cevabı.
    - tool_calls: Koşuda yapılan araç çağrıları (araç, girdi, çıktı). Karar akışı
                  için gerekli DEĞİL — ajan kendi mesaj geçmişini zaten görüyor —
                  ama gözlemlenebilirlik için dışarı çıkarılır: hangi araçların
                  çağrıldığı loglanabilsin ve arayüzde gösterilebilsin.
    - artifacts:  Koşuda üretilen dosya yolları (.png vb.), kırpılmamış.
    """

    input: str
    tools: List[str]
    history: List[Turn]
    response: str
    tool_calls: List[ToolCall]
    artifacts: List[str]
