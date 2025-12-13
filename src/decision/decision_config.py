from dataclasses import dataclass


@dataclass
class DecisionConfig:
    labor_rate: float       # cost per hour of maintenance
    risk_penalty: float     # cost per risk
    fail_cost: float        # cost of spindle overheat (fail)

    @classmethod
    def from_yaml(cls, path: str) -> "DecisionConfig":
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        data = raw.get("decision", {})
        return cls(**data)
