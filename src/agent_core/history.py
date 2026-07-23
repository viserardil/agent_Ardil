"""Session (sohbet) hafızası — turlar arası konuşma geçmişi.

KAPSAM (bilinçli kararlar):
- Hafıza YALNIZCA konuşma metnidir: önceki kullanıcı mesajları + ajanın nihai
  cevapları. Ara süreç (araç çıktıları, plan adımları, üretilen dosyalar) hafızaya
  GİRMEZ — "az önceki grafiği 1 yıllık yap" gibi bir devam isteğinde ajan, önceki
  cevabın ne yapıldığını söylemesine dayanarak referansı çözer ve gerekirse aracı
  yeniden çalıştırır. Bu, bağlamı hafif tutar.
- Tüm session hatırlanır (kırpma yok); istenirse ileride limit eklenebilir.

TASARIM: Geçmiş, referansın ÇÖZÜLDÜĞÜ giriş noktalarına verilir (triyaj, plan
planner'ı, ReAct döngüsü). Bu katmanlar geçmişi somut bir isteğe/adıma çevirdikten
sonra alt katmanlar (executor, replanner) yalnızca o somut çıktıyla çalışır —
geçmişi tekrar görmelerine gerek yoktur ve görmemeleri, tamamlanmış işi yeniden
yapma riskini de ortadan kaldırır.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from typing_extensions import TypedDict


class Turn(TypedDict):
    """Sohbetteki tek tur (bir mesaj)."""

    role: str  # "user" | "ai"
    text: str


def to_messages(history: Any, current_input: str) -> List[Tuple[str, str]]:
    """Geçmiş turları + güncel kullanıcı mesajını, bir prompt placeholder'ına
    verilecek (rol, metin) çiftlerine çevirir.

    Placeholder (``{messages}``) çok mesajlı bir listeyi doğrudan kabul ettiğinden,
    geçmişi ayrı bir metin bloğu olarak enjekte etmek yerine gerçek konuşma sırası
    olarak veriyoruz — model bağlamı doğal biçimde görür.
    """
    messages: List[Tuple[str, str]] = []
    for turn in history or []:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        role = "ai" if turn.get("role") == "ai" else "user"
        messages.append((role, text))
    messages.append(("user", current_input))
    return messages
