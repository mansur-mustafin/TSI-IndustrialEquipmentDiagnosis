from typing import Any

import streamlit as st
from app.explanation import build_mle_text_explanation, build_prompt
from app.ollama import call_ollama
from app.utils import round_dict


def render_input_section() -> tuple[str, dict[str, Any] | None, dict[str, int] | None]:
    st.subheader("Input data")

    input_mode = st.radio(
        "Input mode",
        options=["sensors", "symptoms"],
        format_func=lambda x: "Sensors" if x == "sensors" else "Symptoms",
        horizontal=True,
    )

    raw_data: dict[str, Any] | None = None
    symptoms: dict[str, int] | None = None

    if input_mode == "sensors":
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
            spindle_speed = st.number_input(
                "Spindle speed (RPM)",
                value=2000,
                step=400,
                format="%d",
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
                value=1.0,
                step=0.5,
                format="%.2f",
            )

        raw_data = {
            "vibration_rms": vibration_rms,
            "spindle_temp": spindle_temp,
            "ambient_temp": ambient_temp,
            "coolant_flow": coolant_flow,
            "spindle_speed": spindle_speed,
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

    else:  # input_mode == "symptoms"
        st.caption("Set binary symptoms directly:")

        col1, col2 = st.columns(2)
        with col1:
            high_vibration = st.checkbox("High vibration")
            high_spindle_temp = st.checkbox("High spindle temperature")
        with col2:
            high_spindle_speed = st.checkbox("High spindle speed")
            low_coolant_flow = st.checkbox("Low coolant flow")

        symptoms = {
            "high_vibration": int(high_vibration),
            "high_spindle_temp": int(high_spindle_temp),
            "high_spindle_speed": int(high_spindle_speed),
            "low_coolant_flow": int(low_coolant_flow),
        }

    return input_mode, raw_data, symptoms


def render_diagnosis_options() -> str:
    st.subheader("Diagnosis options")
    method = st.selectbox(
        "Inference method",
        options=["mle", "em", "both"],
        index=2,  # default "both"
        format_func=lambda x: {
            "mle": "Maximum Likelihood Estimation (MLE)",
            "em": "Expectation Maximization (EM)",
            "both": "Both MLE and EM",
        }[x],
    )
    return method


def rendeer_mle_results(raw_data: dict[str, Any]) -> None:
    rounded_mle = st.session_state.mle_result
    if rounded_mle is None:
        return

    st.markdown("**MLE diagnosis**")

    tab_json, tab_text, tab_ollama = st.tabs(["JSON", "Text", "Ollama"])

    with tab_json:
        st.json(rounded_mle)

    with tab_text:
        st.markdown(build_mle_text_explanation(rounded_mle))

    with tab_ollama:
        prompt = build_prompt("Maximum Likelihood Estimation", rounded_mle, raw_data)

        with st.expander("Prompt"):
            st.code(prompt, language="markdown")

        if st.button("Generate explanation (MLE)", key="mle_ollama"):
            with st.spinner("Asking..."):
                explanation = call_ollama(prompt)
            if not explanation:
                st.error("Could not reach Ollama. Please start the Ollama server.")
                st.code("sudo docker start ollama14", language="bash")
            st.session_state.mle_explanation = explanation

        if st.session_state.mle_explanation:
            st.markdown(st.session_state.mle_explanation)


def render_em_results(raw_data: dict[str, Any]) -> None:
    rounded_em = st.session_state.em_result
    if rounded_em is None:
        return

    st.markdown("**EM diagnosis**")

    tab_json, tab_text, tab_ollama = st.tabs(["JSON", "Text", "Ollama"])

    with tab_json:
        st.caption("EM diagnosis (BN probabilities)")
        st.json(rounded_em)

        if st.session_state.em_decision is not None:
            st.caption("Decision & actions (BN → KG → decision)")
            st.json(round_dict(st.session_state.em_decision, ndigits=4))

    with tab_text:
        if st.session_state.em_text_explanation:
            st.markdown(st.session_state.em_text_explanation)

    with tab_ollama:
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
                st.code("sudo docker start ollama14", language="bash")

            st.session_state.em_explanation = explanation

        if st.session_state.em_explanation:
            st.markdown(st.session_state.em_explanation)


def render_results(raw_data: dict[str, Any]) -> None:
    if st.session_state.mle_result is None and st.session_state.em_result is None:
        return

    st.write("### Results")

    if st.session_state.mle_result is not None:
        rendeer_mle_results(raw_data)

    if st.session_state.em_result is not None:
        render_em_results(raw_data)
