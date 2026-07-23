"""Sistemi çalıştıran CLI.

Varsayılan: tam sistem — triyaj isteği kendisi bir şeride yönlendirir.

    python main.py "Apple'ın güncel fiyatı nedir?"
    python main.py "nasılsın?"

Şeridi elle zorlamak (triyajı atlar, tek bir şeridi test etmek için):

    python main.py --lane react "AAPL ile MSFT'yi karşılaştır"
    python main.py --lane plan_execute --tools get_current_stock_price,plot_chart "AAPL grafiği"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_core.plan_execute import build_plan_execute_graph  # noqa: E402
from agent_core.react import build_react_graph  # noqa: E402
from agent_core.router import build_agent_graph  # noqa: E402

DEFAULT_TASK = "Apple'ın güncel hisse fiyatı nedir?"

LANES = {
    "plan_execute": build_plan_execute_graph,
    "react": build_react_graph,
}


def _print_update(node: str, update: dict) -> None:
    """Bir düğümün state güncellemesini okunur şekilde basar."""
    print(f"--- [{node}] ---")

    if update.get("lane"):
        print(f"  şerit : {update['lane']}")
        if update.get("domain"):
            print(f"  alan  : {update['domain']}")
        if update.get("reason"):
            print(f"  gerekçe: {update['reason']}")
        print(f"  araçlar: {', '.join(update.get('tools') or []) or '(yok)'}")

    if "plan" in update:
        for i, step in enumerate(update["plan"], 1):
            print(f"  {i}. {step}")

    # Plan-Execute şeridi doğrudan çalıştırıldığında adım adım hafıza kayıtları
    for record in update.get("memory") or []:
        print(f"  ✓ {record['step']}")
        for call in record["tool_calls"]:
            print(f"      · {call['tool']}({call['input']})")
        print(f"    → {record['result']}")
        if record["artifacts"]:
            print(f"    → dosyalar: {', '.join(record['artifacts'])}")

    for call in update.get("tool_calls") or []:
        print(f"      · {call['tool']}({call['input']})")
    if update.get("artifacts"):
        print(f"    → dosyalar: {', '.join(update['artifacts'])}")

    if update.get("response"):
        print(f"\n=== NİHAİ CEVAP ===\n{update['response']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Triyaj tabanlı hibrit agent")
    parser.add_argument("task", nargs="*", help="Yürütülecek görev")
    parser.add_argument(
        "--lane",
        choices=sorted(LANES),
        help="Şeridi elle zorla (triyajı atlar). Verilmezse triyaj karar verir.",
    )
    parser.add_argument(
        "--tools",
        default="",
        help="Virgülle ayrılmış araç adları. Yalnızca --lane ile anlamlı; "
        "triyaj çalışıyorsa araçları o seçer.",
    )
    args = parser.parse_args()

    task = " ".join(args.task) or DEFAULT_TASK
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]

    if args.lane:
        graph = LANES[args.lane]()
        payload = {"input": task, "tools": tools}
        baslik = f"{args.lane} (elle zorlandı)"
    else:
        graph = build_agent_graph()
        payload = {"input": task}
        baslik = "triyaj karar verecek"

    print(f"\n=== ŞERİT ===\n{baslik}")
    print(f"=== GÖREV ===\n{task}\n")

    for event in graph.stream(payload, config={"recursion_limit": 50}):
        for node, update in event.items():
            _print_update(node, update)


if __name__ == "__main__":
    main()
