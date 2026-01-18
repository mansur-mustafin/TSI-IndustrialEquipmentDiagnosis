from pathlib import Path
import pickle
from typing import Any

import pandas as pd
from bn.diagnose_bn_config import DiagnoseBNConfig
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from app.utils import repo_path

class DiagnoseBN:
    def __init__(self, cfg: DiagnoseBNConfig, path: str = None):
        self.cfg = cfg

        self.vib_hi = None
        self.temp_hi = None
        self.flow_lo = None
        self.speed_hi = None

        self.df = None
        self._model_mle = None
        self._model_em = None
        self._infer_mle = None
        self._infer_em = None

        self._load_data()

        if path is not None:
            self._load_model(path)

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
        self.speed_hi = df["spindle_speed"].quantile(cfg.speed_hi)

        df["high_vibration"] = (df["vibration_rms"] > self.vib_hi).astype(int)
        df["high_spindle_temp"] = (df["delta_temp"] > self.temp_hi).astype(int)
        df["high_spindle_speed"] = (df["spindle_speed"] >= self.speed_hi).astype(int)
        df["low_coolant_flow"] = (df["coolant_flow"] < self.flow_lo).astype(int)

        # Binning variables - alternative approach
        # df["high_vibration"] = pd.qcut(df["vibration_rms"], q=self.cfg.n_bins, labels=False, duplicates="drop")
        # df["high_spindle_temp"] = pd.qcut(df["delta_temp"], q=self.cfg.n_bins, labels=False, duplicates="drop")
        # df["low_coolant_flow"] = pd.qcut(df["coolant_flow"], q=self.cfg.n_bins, labels=False, duplicates="drop")
        # df["high_spindle_speed"] = pd.qcut(df["spindle_speed"], q=self.cfg.n_bins, labels=False, duplicates="drop")

        # self.vib_edges = pd.qcut(df["vibration_rms"], q=self.cfg.n_bins, retbins=True, duplicates="drop")[1]
        # self.temp_edges = pd.qcut(df["delta_temp"], q=self.cfg.n_bins, retbins=True, duplicates="drop")[1]
        # self.flow_edges = pd.qcut(df["coolant_flow"], q=self.cfg.n_bins, retbins=True, duplicates="drop")[1]
        # self.speed_edges = pd.qcut(df["spindle_speed"], q=self.cfg.n_bins, retbins=True, duplicates="drop")[1]

        df["spindle_overheat"] = df["spindle_overheat"].astype(int)
        self.df = df

    def _train_mle(self) -> None:
        model = DiscreteBayesianNetwork(
            ebunch=[
                ("spindle_overheat", "high_vibration"),
                ("spindle_overheat", "high_spindle_temp"),
                ("spindle_overheat", "high_spindle_speed"),
                ("spindle_overheat", "low_coolant_flow"),
            ],
        )

        data_mle = self.df[[
            "spindle_overheat", 
            "high_vibration",
            "high_spindle_temp", 
            "high_spindle_speed", 
            "low_coolant_flow"
        ]]

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
            "low_load_condition",
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

                ("low_load_condition", "high_spindle_speed"),
                ("low_load_condition", "spindle_overheat"),

                # Experimental edges
                # ("fan_fault", "high_vibration"),
                ("low_cooling_efficiency", "low_coolant_flow"),
                
                # Experimental: explicitly connect latents to symptoms
                # ("bearing_wear_high", "high_vibration"),
                # ("fan_fault", "high_spindle_temp"),
                # ("clogged_filter", "low_coolant_flow"),
                # ("low_cooling_efficiency", "high_spindle_temp"),
                # ("low_load_condition", "high_spindle_speed"),

                # # Symptoms drive failure:
                # ("high_vibration", "spindle_overheat"),
                # ("high_spindle_temp", "spindle_overheat"),
                # ("low_coolant_flow", "spindle_overheat"),
                # ("high_spindle_speed", "spindle_overheat"),
            ],
            latents=set(latent_vars),
        )

        data_em = self.df[
            ["spindle_overheat", "high_vibration", "high_spindle_temp", "high_spindle_speed", "low_coolant_flow"]
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

    def train(self, verbose: bool = False) -> None:
        self._train_mle()
        print("MLE model fitted.")
        self._train_em(verbose=verbose)
        print("EM model fitted.")
        self._save_model(repo_path("weights", "diagnose_bn_model.pkl"))
        print("Model saved.")

    def raw2symptoms(self, raw_data: dict[str, float]) -> dict[str, int]:
        """
        Convert raw telemetry data to symptom variables.
        
        :param raw_data: A dictionary with keys: 'vibration_rms', 'spindle_temp', 'ambient_temp', 'coolant_flow', 'spindle_speed'
        :type raw_data: dict[str, float]
        :return: A dictionary with symptom variables
        :rtype: dict[str, int]
        """
        if self.vib_hi is None or self.temp_hi is None or self.flow_lo is None or self.speed_hi is None:
            raise ValueError("Model thresholds are not set. Train the model first.")

        high_vibration = int(raw_data["vibration_rms"] > self.vib_hi)
        high_spindle_temp = int((raw_data["spindle_temp"] - raw_data["ambient_temp"]) > self.temp_hi)
        low_coolant_flow = int(raw_data["coolant_flow"] < self.flow_lo)
        high_spindle_speed = int(raw_data["spindle_speed"] >= self.speed_hi)

        return {
            "high_vibration": high_vibration,
            "high_spindle_temp": high_spindle_temp,
            "high_spindle_speed": high_spindle_speed,
            "low_coolant_flow": low_coolant_flow,
        }
    
    def raw2bins(self, raw_data: dict[str, float]) -> dict[str, int]:
        def digitize(val, edges):
            return int(pd.cut([val], bins=edges, labels=False, include_lowest=True)[0])
        
        return {
            "high_vibration": digitize(raw_data["vibration_rms"], self.vib_edges),
            "high_spindle_temp": digitize(raw_data["spindle_temp"] - raw_data["ambient_temp"], self.temp_edges),
            "low_coolant_flow": digitize(raw_data["coolant_flow"], self.flow_edges),
            "high_spindle_speed": digitize(raw_data["spindle_speed"], self.speed_edges),
        }

    def diagnose(self, symptoms: dict[str, int], method: str = "mle") -> dict[str, Any]:
        """
        Run inference for single telemetry data.
        
        :param symptoms: Binary symptom variables (0/1) keyed by symptom name.
        :type symptoms: dict[str, int]
        :param method: Inference method: 'mle' or 'em'
        :type method: str
        :return: Dictionary with diagnosis results
        :rtype: dict[str, float]
        """

        if method == "mle" and self._infer_mle is None:
            raise ValueError("MLE inference model is not available. Train the model first.")
        if method == "em" and self._infer_em is None:
            raise ValueError("EM inference model is not available. Train the model first.")

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
            for cause in ["bearing_wear_high", "fan_fault", "clogged_filter", 
                          "low_cooling_efficiency", "low_load_condition"]:
                q_cause = infer.query(
                    variables=[cause],
                    evidence=symptoms,
                )
                cause_probs[cause] = float(q_cause.values[1])

            result["cause_probs"] = cause_probs

        return result

    def diagnose_from_raw(self, raw_data: dict[str, float], method: str = "mle", use_bins: bool = False) -> dict[str, Any]:
        """Run inference when raw telemetry data is given.

        :param dict[str, float] raw_data: Raw telemetry data with keys: 
            'vibration_rms', 'spindle_temp', 'ambient_temp', 'coolant_flow', 'spindle_speed'.
        :param str method: Inference method: 'mle' or 'em'.
        :param bool use_bins: Whether to use binned variables for inference.
        """

        evidence = self.raw2bins(raw_data) if use_bins else self.raw2symptoms(raw_data)
        return self.diagnose(evidence, method=method)

    def overheat_prob(self, evidence: dict[str, int], method: str = "em") -> float:
        if method == "em":
            infer = self._infer_em
        else:
            infer = self._infer_mle

        if infer is None:
            raise ValueError("Inference model not available. Train/load first.")

        q = infer.query(variables=["spindle_overheat"], evidence=evidence)
        return float(q.values[1])

    def residual_overheat_prob_after_fix(
        self,
        evidence: dict[str, int],
        cause: str,
        method: str = "em",
    ) -> float:
        if method not in ("mle", "em"):
            raise ValueError("method must be 'mle' or 'em'.")

        fixed_evidence = dict(evidence)
        fixed_evidence[cause] = 0  # assume fixed

        return self.overheat_prob(fixed_evidence, method=method)

    def _save_model(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_mle": self._model_mle,
            "model_em": self._model_em,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _load_model(self, path: Path) -> None:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with path.open("rb") as f:
            payload = pickle.load(f)

        self._model_mle = payload["model_mle"]
        self._model_em = payload["model_em"]

        self._infer_mle = VariableElimination(self._model_mle)
        self._infer_em = VariableElimination(self._model_em)
