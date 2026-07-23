"""Plan-Execute şeridinin akış boyunca taşıdığı durum (state)."""

from __future__ import annotations

import operator
from typing import Annotated, List

from typing_extensions import TypedDict

from agent_core.history import Turn
from agent_core.plan_execute.memory import StepRecord


class PlanExecute(TypedDict):
    """Graf boyunca güncellenen ortak durum.

    - input:        Kullanıcının orijinal görevi.
    - tools:        Triyajın ön-seçtiği araç ADLARI. Planner ve executor yalnızca
                    bu araçları görür; amaç context'i şişirmemek. Boş bırakılırsa
                    (ör. triyaj devrede değilken) registry'deki tüm araçlar kullanılır.
    - history:      Session (sohbet) geçmişi. YALNIZCA planner görür: referansı
                    ("az önceki grafiği 1 yıllık yap") somut adımlara çevirmek için.
                    Executor ve replanner somut plandan çalışır, geçmişi görmez.
    - plan:         Henüz yürütülmemiş adımlar.
    - memory:       KISA SÜRELİ HAFIZA. Tamamlanan her adımın tam kaydı: yapılan
                    araç çağrıları, ham çıktıları, adım sonucu ve üretilen dosyalar.
                    operator.add ile birikir. Eski "past_steps: (adım, sonuç)"
                    ikilisinin yerini alır — o yapı araç çıktılarını taşımadığı
                    için replanner adımın gerçekte ne ürettiğini göremiyor ve
                    tamamlanmış adımı yeniden planlıyordu.
    - response:     Doluysa graf sonlanır ve bu cevap döner.
    - replan_count: Replanner'ın planı GERÇEKTEN revize ettiği (Response değil,
                    yeni bir Plan döndürdüğü) sayı; operator.add ile birikir.
    """

    input: str
    tools: List[str]
    history: List[Turn]
    plan: List[str]
    memory: Annotated[List[StepRecord], operator.add]
    response: str
    replan_count: Annotated[int, operator.add]
