from typing import Any

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from bn.diagnose_bn_config import DiagnoseBNConfig


class DiagnoseBN:
    def __init__(self, cfg: DiagnoseBNConfig):
        self.cfg = cfg

        self.vib_hi = None
        self.temp_hi = None
        self.flow_lo = None

        self.df = None
        self._model_mle = None
        self._model_em = None
        self._infer_mle = None
        self._infer_em = None

    def _load_data(self) -> None:
        cfg = self.cfg
    
        tele = pd.read_csv(cfg.telemetry_path, parse_dates=["timestamp"])
        labels = pd.read_csv(cfg.labels_path, parse_dates=["timestamp"])

        df = tele.merge(labels, on=["timestamp", "machine_id"])

        if cfg.machine_filter is not None:
            df = df[df[cfg.machine_id_col] == cfg.machine_filter].copy()

        # Setting labels
        df["delta_temp"] = df["spindle_temp"] - df["ambient_temp"]

        self.vib_hi = df["vibration_rms"].quantile(cfg.vib_hi)
        self.temp_hi = df["delta_temp"].quantile(cfg.temp_hi)
        self.flow_lo = df["coolant_flow"].quantile(cfg.flow_lo)

        df["high_vibration"] = (df["vibration_rms"] > self.vib_hi).astype(int)
        df["high_spindle_temp"] = (df["delta_temp"] > self.temp_hi).astype(int)
        df["low_coolant_flow"] = (df["coolant_flow"] < self.flow_lo).astype(int)
        df["spindle_overheat"] = df["spindle_overheat"].astype(int)

        self.df = df
    
    def _train_mle(self) -> None:
        model = DiscreteBayesianNetwork(
            ebunch=[
                ("spindle_overheat", "high_vibration"),
                ("spindle_overheat", "high_spindle_temp"),
                ("spindle_overheat", "low_coolant_flow"),
            ],
        )

        data_mle = self.df[
            ["spindle_overheat", "high_vibration", "high_spindle_temp", "low_coolant_flow"]
        ]

        model.fit(data_mle, estimator=MaximumLikelihoodEstimator)
        model.check_model()
        
        self._model_mle = model
        self._infer_mle = VariableElimination(model)

    def _train_em(self, verbose: bool = False) -> None:
        latent_vars = [
            "bearing_wear_high",
            "fan_fault",
            "clogged_filter",
            "low_cooling_efficiency",
        ]

        model = DiscreteBayesianNetwork(
            ebunch=[
                ("bearing_wear_high", "high_vibration"),
                ("bearing_wear_high", "spindle_overheat"),

                ("fan_fault", "high_spindle_temp"),
                ("fan_fault", "spindle_overheat"),

                ("clogged_filter", "low_coolant_flow"),
                ("clogged_filter", "spindle_overheat"),

                ("low_cooling_efficiency", "high_spindle_temp"),
                ("low_cooling_efficiency", "spindle_overheat"),
            ],
            latents=set(latent_vars)
        )

        data_em = self.df[
            ["spindle_overheat", "high_vibration", "high_spindle_temp", "low_coolant_flow"]
        ]

        em = ExpectationMaximization(model, data_em)

        cpds_em = em.get_parameters(
            latent_card={lv: 2 for lv in latent_vars},
            max_iter=self.cfg.em_max_iter,
            atol=self.cfg.em_atol,
            seed=self.cfg.em_seed,
            show_progress=verbose,
        )

        model.add_cpds(*cpds_em)
        model.check_model()
        
        self._model_em = model
        self._infer_em = VariableElimination(model)

    def train(self) -> None:
        self._load_data()
        print("Data loaded.")
        self._train_mle()
        print("MLE model fitted.")
        self._train_em()
        print("EM model fitted.")

    def raw2symptoms(self, raw_data: dict[str, float]) -> dict[str, int]:
        """
        Convert raw telemetry data to symptom variables.
        
        :param raw_data: A dictionary with keys: 'vibration_rms', 'spindle_temp', 'ambient_temp', 'coolant_flow'
        :type raw_data: dict[str, float]
        :return: A dictionary with symptom variables
        :rtype: dict[str, int]
        """
        if self.vib_hi is None or self.temp_hi is None or self.flow_lo is None:
            raise ValueError("Model thresholds are not set. Train the model first.")
        
        high_vibration = int(raw_data["vibration_rms"] > self.vib_hi)
        high_spindle_temp = int((raw_data["spindle_temp"] - raw_data["ambient_temp"]) > self.temp_hi)
        low_coolant_flow = int(raw_data["coolant_flow"] < self.flow_lo)

        return {
            "high_vibration": high_vibration,
            "high_spindle_temp": high_spindle_temp,
            "low_coolant_flow": low_coolant_flow,
        }
    
    def diagnose(self, raw_data: dict[str, float], method: str = "mle") -> dict[str, Any]:
        """
        Run inference for single telemetry data.
        
        :param raw_data: A dictionary with keys: 'vibration_rms', 'spindle_temp', 'ambient_temp', 'coolant_flow'
        :type raw_data: dict[str, float]
        :param method: Inference method: 'mle' or 'em'
        :type method: str
        :return: Dictionary with diagnosis results
        :rtype: dict[str, float]
        """

        if method == "mle" and self._infer_mle is None:
            raise ValueError("MLE inference model is not available. Train the model first.")
        if method == "em" and self._infer_em is None:
            raise ValueError("EM inference model is not available. Train the model first.")

        symptoms = self.raw2symptoms(raw_data)

        if method == "mle":
            infer = self._infer_mle
        else:
            infer = self._infer_em

        query_result = infer.query(
            variables=["spindle_overheat"],
            evidence=symptoms,
        )

        prob_overheat = query_result.values[1]

        result = {
            "method": method,
            "evidence": symptoms,
            "overheat_prob": prob_overheat,
        }

        if method == "em":
            cause_probs = {}
            for cause in ["bearing_wear_high", "fan_fault",
                          "clogged_filter", "low_cooling_efficiency"]:
                q_cause = infer.query(
                    variables=[cause],
                    evidence=symptoms,
                )
                cause_probs[cause] = float(q_cause.values[1])

            result["cause_probs"] = cause_probs

        return result
