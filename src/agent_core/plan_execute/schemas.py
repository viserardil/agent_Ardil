"""Planner ve replanner'ın structured output şemaları.

NOT: Eski projedeki ``PlannerDecision`` (needs_tools / direct_answer) burada YOK.
"Bu görev araç gerektiriyor mu?" sorusunu artık üstteki triyaj katmanı cevaplıyor;
Plan-Execute şeridine bir görev geldiyse araç gerektirdiği zaten kesindir. Planner
yalnızca "nasıl" sorusuna cevap verir.
"""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Görevi çözmek için sıralı adımlar planı."""

    steps: List[str] = Field(
        description="Takip edilecek, doğru sırada dizilmiş adımlar."
    )


class Response(BaseModel):
    """Kullanıcıya verilecek nihai cevap."""

    response: str = Field(description="Kullanıcıya dönülecek nihai yanıt.")


class Act(BaseModel):
    """Replanner'ın kararı: ya devam etmek için yeni plan, ya da bitirmek için cevap."""

    action: Union[Response, Plan] = Field(
        description=(
            "Yapılacak eylem. Kullanıcıya cevap verilecekse Response kullan. "
            "Hedefe ulaşmak için daha fazla araç kullanımı gerekiyorsa Plan kullan."
        )
    )
