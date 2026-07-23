"""Triyajın structured output şeması."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

# Şerit adları tek kaynaktan gelsin: hem şema hem yönlendirme aynı listeyi kullanır.
Lane = Literal["direct", "react", "plan_execute"]


class TriageDecision(BaseModel):
    """Gelen isteğin hangi şeride gideceği ve o şeridin neye ihtiyacı olduğu.

    ``lane`` KATI bir enum'dur: model geçersiz bir şerit adı üretemez, çünkü
    json_schema bunu üretim anında zorlar. Yanlış şerit SEÇMESİ mümkündür — ona
    karşı bilinçli olarak telafi mekanizması yok — ama GEÇERSİZ bir şerit dönmesi
    mümkün değil.
    """

    lane: Lane = Field(
        description=(
            "İsteğin yönlendirileceği şerit. "
            "direct: araç gerektirmeyen istek (selamlaşma, sohbet, tanım/genel bilgi, "
            "kapsam dışı istek) — cevabı sen yaz. "
            "react: araç gerekiyor ama yol baştan belli değil; ara sonuca göre yön "
            "değişebilir. Tek araçla çözülen basit sorgular da buraya. "
            "plan_execute: birden çok adım gerektiren, adımları ve sırası BAŞTAN "
            "çıkarılabilen, birbirine bağımlı işler."
        )
    )
    domain: str = Field(
        default="",
        description=(
            "İsteğin konu alanı, tek kelime/kısa ifade (ör. finans, hava_durumu, genel). "
            "Yönlendirmeyi etkilemez; loglama ve sonradan analiz içindir."
        ),
    )
    tools: List[str] = Field(
        default_factory=list,
        description=(
            "Şeridin kullanabileceği araç adları. Kataloktaki adlardan BİREBİR seç. "
            "CÖMERT ol: gerekebilecek araçları da ekle — listede olmayan bir aracı "
            "alt katman GÖREMEZ ve telafi mekanizması yoktur. lane=direct ise boş bırak."
        ),
    )
    reason: str = Field(
        default="",
        description="Bu şeridi neden seçtiğin, tek cümle. Karar akışını etkilemez; logdan okunur.",
    )
    answer: str = Field(
        default="",
        description=(
            "YALNIZCA lane=direct ise doldur: kullanıcıya verilecek nihai cevap. "
            "Diğer şeritlerde boş bırak — cevabı o şerit üretecek."
        ),
    )
