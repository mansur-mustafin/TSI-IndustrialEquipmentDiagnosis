from typing import Any

import streamlit as st
from app.explanation import build_text_explanation
from app.utils import round_dict
from bn.diagnose_bn import DiagnoseBN
from bn.diagnose_bn_config import DiagnoseBNConfig
from decision.action_decision import ActionDecision
from decision.action_decision_config import ActionDecisionConfig
from kg.kg_client import KGClient
from kg.kg_client_config import KGClientConfig


def get_trained_model():
    if st.session_state.model is None:
        config = DiagnoseBNConfig.from_yaml("./../configs/bn.yaml")
        model = DiagnoseBN(config)
        model.train()
        st.session_state.model = model
    return st.session_state.model


def get_kg_client():
    if st.session_state.kg_client is None:
        kg_cfg = KGClientConfig.from_yaml("./../configs/kg.yaml")
        st.session_state.kg_client = KGClient(kg_cfg)
    return st.session_state.kg_client


def get_action_decision(kg_client: KGClient) -> ActionDecision:
    decision_cfg = ActionDecisionConfig.from_yaml("./../configs/decision.yaml")
    return ActionDecision(kg_client, decision_cfg)


def run_diagnosis(
    method: str,
    raw_data: dict[str, Any] | None,
    symptoms: dict[str, int] | None = None,
    input_mode: str = "sensors",
) -> None:
    model = get_trained_model()
    kg_client = get_kg_client()
    action_decision = get_action_decision(kg_client)

    # reset explanations
    st.session_state.mle_explanation = ""
    st.session_state.em_explanation = ""
    st.session_state.em_text_explanation = ""
    st.session_state.em_decision = None

    # --- MLE
    if method in ("mle", "both"):
        if input_mode == "symptoms" and symptoms is not None:
            result_mle = model.diagnose(symptoms, method="mle")
        if input_mode == "sensors" and raw_data is not None:
            result_mle = model.diagnose_from_raw(raw_data, method="mle")

        st.session_state.mle_result = round_dict(result_mle, ndigits=4)
    else:
        st.session_state.mle_result = None

    # --- EM
    if method in ("em", "both"):
        if input_mode == "symptoms" and symptoms is not None:
            result_em = model.diagnose(symptoms, method="em")
        if input_mode == "sensors" and raw_data is not None:
            result_em = model.diagnose_from_raw(raw_data, method="em")

        st.session_state.em_result_raw = result_em
        st.session_state.em_result = round_dict(result_em, ndigits=4)

        decision = action_decision.decide(result_em)

        # add KG context
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
            evidence=result_em["evidence"],
            decision=st.session_state.em_decision,
            kg_client=kg_client,
        )
        st.session_state.em_text_explanation = em_text_expl
    else:
        st.session_state.em_result = None
        st.session_state.em_result_raw = None
        st.session_state.em_decision = None
        st.session_state.em_text_explanation = ""

    st.rerun()
