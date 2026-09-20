"""Small, reproducible evaluation of the language-model boundary.

The baseline freezes outputs from an earlier prompt that asked the model to choose the evidence.
The current product computes the evidence first and uses the model only for prose. Keeping the old
selections outside the live brief cache makes the comparison reproducible after briefs are regenerated.

Run:  python evaluate_briefs.py
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import ai
import triage as T

BASELINE = Path(__file__).parent / "data" / "evaluation_baseline.json"


def coverage(chosen: set[str], target: set[str]) -> float:
    return 1.0 if not target else len(chosen & target) / len(target)


def pct(values: list[float]) -> str:
    return f"{100 * statistics.mean(values):.0f}%" if values else "n/a"


def main() -> None:
    _, packages = T.build_queue(T.default_csv())
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    rows = []
    for case_id, saved in baseline["cases"].items():
        pkg = packages[case_id]
        model = set(saved["model_selected_signals"])
        engine = {signal["key"] for signal in ai.evidence_set(pkg)}
        flags = {signal["key"] for signal in pkg["signals"]
                 if signal["kind"] == "flag" and signal["band"] != "reference"}
        clashes = {note["signal"] for note in pkg["care_notes"] if note["effect"] == "clash"}
        target = flags | clashes
        rows.append({
            "model_target": coverage(model, target),
            "model_flags": coverage(model, flags),
            "engine_target": coverage(engine, target),
            "engine_flags": coverage(engine, flags),
            "model_missed": bool(target - model),
            "engine_missed": bool(target - engine),
            "has_flags": bool(flags),
        })

    n = len(rows)
    flagged = [row for row in rows if row["has_flags"]]
    print(f"Frozen evidence-selection comparison ({n} cases)")
    print("-" * 65)
    print(f"{'':27}{'model selected':>18}{'engine selected':>20}")
    print(f"{'briefs missing required evidence':27}{sum(r['model_missed'] for r in rows):>15} / {n:<2}"
          f"{sum(r['engine_missed'] for r in rows):>17} / {n:<2}")
    print(f"{'mean required-evidence coverage':27}{pct([r['model_target'] for r in rows]):>18}"
          f"{pct([r['engine_target'] for r in rows]):>20}")
    print(f"{'mean triggered-flag coverage':27}{pct([r['model_flags'] for r in flagged]):>18}"
          f"{pct([r['engine_flags'] for r in flagged]):>20}")

    modes = Counter(brief["mode"] for brief in ai.briefs_for_queue(packages).values())
    print("\nCurrent delivered briefs")
    print("-" * 65)
    print("  " + ", ".join(f"{mode}: {count}" for mode, count in sorted(modes.items())))
    print("\nInterpretation")
    print("  The model should not select evidence. The engine covers every triggered flag and care-setting clash.")
    print("  This comparison does not prove that model prose is better than rule-built prose. That requires a")
    print("  blinded investigator study of clarity, actionability and time to decision. The rule brief remains")
    print("  the fallback until that study shows a benefit.")


if __name__ == "__main__":
    main()
