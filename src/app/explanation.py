import json
from typing import Any

from kg.kg_client import KGClient

SYM_LABELS: dict[str, str] = {
    "high_vibration": "Vibration is High",
    "low_coolant_flow": "Coolant flow is Low",
    "high_spindle_temp": "Spindle temperature is High",
}


def build_text_explanation(
    evidence: dict[str, Any],
    decision: dict[str, Any],
    kg_client: KGClient
) -> str:
    active_syms = [
        SYM_LABELS.get(name, name)
        for name, val in evidence.items()
        if val == 1
    ]

    because_part = " and ".join(
        active_syms) if active_syms else "no active symptoms"

    if decision["best_action"]["action"] == "schedule":
        proc_id = decision["best_action"]["proc_id"]
        procedure = kg_client.get_procedure_explanation(proc_id)

        cause = decision["best_action"]["cause"]
        cause_context = kg_client.get_cause_context(cause)

        proc_name = procedure["proc_name"]
        effort_h = procedure["effort_h"]
        parts_cost = procedure["spare_parts_cost_eur"]

        explanation = (
            f"Because {because_part}, the system estimates Overheat probability of {decision['overheat_prob']:.2f}. "
            f"Likely cause is {cause_context['cause_name']}. "
            f"Recommended action: {proc_name} ({effort_h:.0f}h, {parts_cost:.0f}€)."
        )
    else:
        return (
            f"Because {because_part}, the system estimates"
            f"Overheat probability of {decision['overheat_prob']:.2f}. "
            "It recommends continuing operation without scheduling maintenance."
        )

    return explanation


def build_mle_text_explanation(result: dict[str, Any]) -> str:
    evidence = result.get("evidence", {})
    overheat_prob = float(result.get("overheat_prob", 0.0))

    active_syms = [
        label
        for name, label in SYM_LABELS.items()
        if evidence.get(name, 0) == 1
    ]

    if active_syms:
        because = ", ".join(active_syms)
        return (
            f"Because {because}. The system estimates Overheat probability of {overheat_prob:.2f}. "
            f"For cause-level diagnosis and maintenance recommendations, see the EM-based model below."
        )
    else:
        return (
            "No symptoms are active. "
            f"Estimated Overheat probability of spindle is {overheat_prob:.2f}."
        )


def build_prompt(
    method: str,
    result: dict[str, Any],
    raw_data: dict[str, Any],
    decision: dict[str, Any] | None = None,
) -> str:
    base = (
        f"You are an expert in rotating machinery diagnostics.\n\n"
        f"Sensor readings (input data):\n"
        f"{json.dumps(raw_data, indent=2)}\n\n"
        f"Diagnosis probabilities using method \"{method}\":\n"
        f"{json.dumps(result, indent=2)}\n"
    )

    if decision is not None:
        base += (
            f"Decision module output (recommended action and expected costs):\n"
            f"{json.dumps(decision, indent=2)}\n\n"
            f"Explain in clear language:\n"
            f"- What the most likely root causes are,\n"
            f"- Why the recommended action is reasonable,\n"
            f"- How the sensor readings relate to the inferred causes and the recommended action.\n"
        )
    else:
        base += (
            f"Explain in clear language what the most likely issues are and how the sensor readings support that conclusion."
        )

    return base
