import json
from pathlib import Path
from typing import Any

import streamlit as st

from bn.diagnose_bn import DiagnoseBN
from bn.diagnose_bn_config import DiagnoseBNConfig
from decision.action_decision import ActionDecision
from decision.decision_config import DecisionConfig
from kg.kg_client import KGClient
from kg.kg_client_config import KGClientConfig
from utils import round_dict, call_ollama


def build_text_explanation(evidence: dict[str, Any], decision: dict[str, Any], kg_client: KGClient) -> str:
    SYM_LABELS = {
        "high_vibration": "Vibration is High",
        "low_coolant_flow": "Coolant flow is Low",
        "high_spindle_temp": "Spindle temperature is High",
    }

    active_syms = [
        SYM_LABELS.get(name, name) 
        for name, val in evidence.items() 
        if val == 1
    ]

    because_part = " and ".join(active_syms) if active_syms else "no active symptoms"

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

    SYM_LABELS = {
        "high_vibration": "Vibration is High",
        "low_coolant_flow": "Coolant flow is Low",
        "high_spindle_temp": "Spindle temperature is High",
    }
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
    base = f"""You are an expert in rotating machinery diagnostics.

Sensor readings (input data):
{json.dumps(raw_data, indent=2)}

Diagnosis probabilities using method "{method}":
{json.dumps(result, indent=2)}
"""

    if decision is not None:
        base += f"""
Decision module output (recommended action and expected costs):
{json.dumps(decision, indent=2)}

Explain in clear language:
- What the most likely root causes are,
- Why the recommended action is reasonable,
- How the sensor readings relate to the inferred causes and the recommended action.
"""
    else:
        base += """
Explain in clear language what the most likely issues are and how the sensor readings support that conclusion.
"""

    return base



def get_trained_model():
    config = DiagnoseBNConfig.from_yaml("./../configs/bn.yaml")
    model = DiagnoseBN(config)
    model.train()
    return model


def get_kg_client():
    kg_cfg = KGClientConfig.from_yaml("./../configs/kg.yaml")
    kg_client = KGClient(kg_cfg)
    return kg_client


def get_action_decision(kg_client: KGClient):
    decision_cfg = DecisionConfig.from_yaml("./../configs/decision.yaml")
    action_decision = ActionDecision(kg_client, decision_cfg)
    return action_decision


def main():
    st.set_page_config(page_title="BN Diagnosis Demo", layout="centered")

    # -------- Session
    if "mle_result" not in st.session_state:
        st.session_state.mle_result = None
    if "em_result" not in st.session_state:
        st.session_state.em_result = None
    if "em_result_raw" not in st.session_state:
        st.session_state.em_result_raw = None
    if "mle_explanation" not in st.session_state:
        st.session_state.mle_explanation = ""
    if "em_explanation" not in st.session_state:
        st.session_state.em_explanation = ""
    if "em_text_explanation" not in st.session_state:
        st.session_state.em_text_explanation = ""
    if "model" not in st.session_state:
        st.session_state.model = None
    if "kg_client" not in st.session_state:
        st.session_state.kg_client = None
    if "em_decision" not in st.session_state:
        st.session_state.em_decision = None

    st.title("Bayesian Network Diagnosis")
    st.write(
        "Interactively diagnose conditions using your Bayesian Network model. "
        "Adjust the sensor readings below and run the diagnosis."
    )

    st.subheader("Input data")

    col1, col2 = st.columns(2)

    with col1:
        vibration_rms = st.number_input(
            "Vibration RMS",
            value=0.5,
            step=0.1,
            format="%.3f",
        )
        spindle_temp = st.number_input(
            "Spindle temperature (°C)",
            value=80.0,
            step=1.0,
            format="%.1f",
        )
    with col2:
        ambient_temp = st.number_input(
            "Ambient temperature (°C)",
            value=20.0,
            step=1.0,
            format="%.1f",
        )
        coolant_flow = st.number_input(
            "Coolant flow",
            value=10.0,
            step=0.5,
            format="%.2f",
        )

    raw_data = {
        "vibration_rms": vibration_rms,
        "spindle_temp": spindle_temp,
        "ambient_temp": ambient_temp,
        "coolant_flow": coolant_flow,
    }

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Current input data:")
        st.json(raw_data)
    with col2:
        if st.session_state.model is not None:
            st.caption("Derived binary symptoms from input:")
            model = st.session_state.model
            st.json(model.raw2symptoms(raw_data))

    st.subheader("Diagnosis options")
    method = st.selectbox(
        "Inference method",
        options=["mle", "em", "both"],
        index=2,  # default "both"
    )

    # --- Run diagnosis
    if st.button("Run diagnosis"):
        if st.session_state.model is None:
            st.session_state.model = get_trained_model()
        model = st.session_state.model

        if st.session_state.kg_client is None:
            st.session_state.kg_client = get_kg_client()
        kg_client = st.session_state.kg_client

        # reset explanations when running again
        st.session_state.mle_explanation = ""
        st.session_state.em_explanation = ""
        st.session_state.em_text_explanation = ""
        st.session_state.em_decision = None

        if method in ("mle", "both"):
            result_mle = model.diagnose(raw_data, method="mle")
            st.session_state.mle_result = round_dict(result_mle, ndigits=4)
        else:
            st.session_state.mle_result = None

        if method in ("em", "both"):
            result_em = model.diagnose(raw_data, method="em")
            st.session_state.em_result_raw = result_em
            st.session_state.em_result = round_dict(result_em, ndigits=4)

            action_decision = get_action_decision(kg_client)
            decision = action_decision.decide(result_em)

            # --- Enrich decision with KG context ---
            if decision["best_action"]["action"] == "schedule":
                best = decision["best_action"]
                proc_id = best["proc_id"]
                procedure = kg_client.get_procedure_explanation(proc_id)
                cause = best["cause"]
                cause_context = kg_client.get_cause_context(cause)
                full_decision = {
                    **decision,
                    "procedure_details": procedure,
                    "cause_context": cause_context,
                }

                st.session_state.em_decision = full_decision
            else:
                st.session_state.em_decision = decision

            em_text_expl = build_text_explanation(
                result_em["evidence"], st.session_state.em_decision, kg_client
            )
            st.session_state.em_text_explanation = em_text_expl
        else:
            st.session_state.em_result = None
            st.session_state.em_result_raw = None
            st.session_state.em_decision = None
            st.session_state.em_text_explanation = ""

    # --- Results rendering
    if st.session_state.mle_result is not None or st.session_state.em_result is not None:
        st.write("### Results")

    # --- MLE
    if st.session_state.mle_result is not None:
        rounded_mle = st.session_state.mle_result

        st.markdown("**MLE diagnosis**")
        mle_tabs = st.tabs(["JSON", "Text", "Ollama"], default="Text")

        with mle_tabs[0]:
            st.json(rounded_mle)

        with mle_tabs[1]:
            if st.session_state.mle_result is not None:
                st.markdown(build_mle_text_explanation(rounded_mle))

        with mle_tabs[2]:
            prompt = build_prompt("Maximum Likelihood Estimation", rounded_mle, raw_data)

            with st.expander("Prompt"):
                st.code(prompt, language="markdown")

            if st.button("Generate explanation (MLE)", key="mle_ollama"):
                with st.spinner("Asking..."):
                    explanation = call_ollama(prompt)
                if not explanation:
                    st.error("Could not reach Ollama. Please start the Ollama server.")
                    st.code(
                        "sudo docker start ollama14",
                        language="bash",
                    )

                st.session_state.mle_explanation = explanation

            if st.session_state.mle_explanation:
                st.markdown(st.session_state.mle_explanation)

    # --- EM
    if st.session_state.em_result is not None:
        rounded_em = st.session_state.em_result

        st.markdown("**EM diagnosis**")
        em_tabs = st.tabs(["JSON", "Text", "Ollama"], default="Text")

        with em_tabs[0]:
            st.caption("EM diagnosis (BN probabilities)")
            st.json(rounded_em)

            if st.session_state.em_decision is not None:
                st.caption("Decision & actions (BN → KG → decision)")
                st.json(round_dict(st.session_state.em_decision, ndigits=4))

        with em_tabs[1]:
            if st.session_state.em_text_explanation:
                st.markdown(st.session_state.em_text_explanation)

        with em_tabs[2]:
            prompt = build_prompt(
                "Expectation Maximization",
                rounded_em,
                raw_data,
                decision=st.session_state.em_decision,
            )

            with st.expander("Prompt"):
                st.code(prompt, language="markdown")

            if st.button("Generate explanation (EM)", key="em_ollama"):
                with st.spinner("Asking..."):
                    explanation = call_ollama(prompt)

                if not explanation:
                    st.error("Could not reach Ollama. Please start the Ollama server.")
                    st.code(
                        "sudo docker start ollama14",
                        language="bash",
                    )

                st.session_state.em_explanation = explanation

            if st.session_state.em_explanation:
                st.markdown(st.session_state.em_explanation)


if __name__ == "__main__":
    main()
