from dataclasses import dataclass
from typing import Optional


@dataclass
class DiagnoseBNConfig:
    telemetry_path: str
    labels_path: str
    machine_filter: Optional[str] = None

    # --- thresholds
    vib_hi: float = 0.9
    temp_hi: float = 0.9
    flow_lo: float = 0.1
    speed_hi: float = 0.9
    n_bins: int = 5

    # --- hyperparameters
    em_max_iter: int = 100
    em_atol: float = 1e-4
    em_seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str, verbose: bool = False) -> "DiagnoseBNConfig":
        import yaml

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        if verbose:
            print(f"Loaded BN config from: {yaml_path}\n")
            print(yaml.dump(raw, sort_keys=False))

        data = raw.get("data", {})
        disc = raw.get("thresholds", {})
        em = raw.get("em", {})

        merged = {}
        merged.update(data)
        merged.update(disc)
        merged.update(em)

        return cls(**merged)
