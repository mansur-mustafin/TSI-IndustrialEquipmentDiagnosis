from typing import Any

import streamlit as st

from app.diagnosis import run_diagnosis
from app.ui import render_input_section, render_diagnosis_options, render_results

SESSION_DEFAULTS: dict[str, Any] = {
    "mle_result": None,
    "em_result": None,
    "em_result_raw": None,
    "mle_explanation": "",
    "em_explanation": "",
    "em_text_explanation": "",
    "model": None,
    "kg_client": None,
    "em_decision": None,
}


def init_session_state() -> None:
    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


def main():
    st.set_page_config(page_title="BN Diagnosis Demo", layout="centered")
    init_session_state()

    st.title("Bayesian Network Diagnosis")
    st.write(
        "Interactively diagnose conditions using your Bayesian Network model. "
        "Adjust the sensor readings below and run the diagnosis."
    )

    input_mode, raw_data, symptoms = render_input_section()
    method = render_diagnosis_options()

    if st.button("Run diagnosis"):
        run_diagnosis(
            raw_data=raw_data,
            method=method,
            symptoms=symptoms,
            input_mode=input_mode,
        )

    raw_for_results: dict[str, Any] = raw_data if raw_data is not None else symptoms
    render_results(raw_for_results)


if __name__ == "__main__":
    main()
