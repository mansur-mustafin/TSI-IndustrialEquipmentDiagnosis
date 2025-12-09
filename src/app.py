import json
from pathlib import Path

import streamlit as st

from bn.diagnose_bn_config import DiagnoseBNConfig
from bn.diagnose_bn import DiagnoseBN
from utils import round_dict, call_ollama


def build_prompt(method: str, result: dict, raw_data: dict) -> str:
    return f"""You are an expert in rotating machinery diagnostics. 
Sensor readings (input data):
{json.dumps(raw_data, indent=2)}
Diagnosis probabilities using method "{method}":
{json.dumps(result, indent=2)}
Explain in clear language what the most likely issues are.
"""


def get_trained_model():
    config_path = Path(__file__).resolve().parent / ".." / "configs" / "bn.yaml"
    config = DiagnoseBNConfig.from_yaml(str(config_path))
    model = DiagnoseBN(config)
    model.load_data()
    model.train()
    return model


def main():
    st.set_page_config(page_title="BN Diagnosis Demo", layout="centered")

    # -------- Session
    if "mle_result" not in st.session_state:
        st.session_state.mle_result = None
    if "em_result" not in st.session_state:
        st.session_state.em_result = None
    if "mle_explanation" not in st.session_state:
        st.session_state.mle_explanation = ""
    if "em_explanation" not in st.session_state:
        st.session_state.em_explanation = ""
    if "model" not in st.session_state:
        st.session_state.model = None

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
            model = get_trained_model()
            st.session_state.model = model
        model = st.session_state.model

        # reset explanations when running again
        st.session_state.mle_explanation = ""
        st.session_state.em_explanation = ""

        if method in ("mle", "both"):
            result_mle = model.diagnose(raw_data, method="mle")
            st.session_state.mle_result = round_dict(result_mle, ndigits=4)
        else:
            st.session_state.mle_result = None

        if method in ("em", "both"):
            result_em = model.diagnose(raw_data, method="em")
            st.session_state.em_result = round_dict(result_em, ndigits=4)
        else:
            st.session_state.em_result = None

    # --- Results rendering
    if st.session_state.mle_result is not None or st.session_state.em_result is not None:
        st.write("### Results")

    # --- MLE
    if st.session_state.mle_result is not None:
        rounded_mle = st.session_state.mle_result

        st.markdown("**MLE diagnosis**")
        mle_tabs = st.tabs(["JSON", "Text", "Ollama"])

        with mle_tabs[0]:
            st.json(rounded_mle)

        with mle_tabs[1]:
            st.text("TODO: plain text rendering of MLE results")

        with mle_tabs[2]:
            prompt = build_prompt("Maximum Likelihood Estimation", rounded_mle, raw_data)

            with st.expander("Prompt"):
                st.code(prompt, language="markdown")

            if st.button("Generate explanation (MLE)", key="mle_ollama"):
                with st.spinner("Asking..."):
                    explanation = call_ollama(prompt)
                st.session_state.mle_explanation = explanation

            if st.session_state.mle_explanation:
                st.markdown(st.session_state.mle_explanation)

    # --- EM
    if st.session_state.em_result is not None:
        rounded_em = st.session_state.em_result

        st.markdown("**EM diagnosis**")
        em_tabs = st.tabs(["JSON", "Text", "Ollama"])

        with em_tabs[0]:
            st.json(rounded_em)

        with em_tabs[1]:
            st.text("TODO: plain text rendering of EM results")

        with em_tabs[2]:
            prompt = build_prompt("Expectation Maximization", rounded_em, raw_data)

            with st.expander("Prompt"):
                st.code(prompt, language="markdown")

            if st.button("Generate explanation (EM)", key="em_ollama"):
                with st.spinner("Asking..."):
                    explanation = call_ollama(prompt)
                st.session_state.em_explanation = explanation

            if st.session_state.em_explanation:
                st.markdown(st.session_state.em_explanation)


if __name__ == "__main__":
    main()
