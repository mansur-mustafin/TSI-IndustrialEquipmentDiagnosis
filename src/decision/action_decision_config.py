from dataclasses import dataclass


@dataclass
class ActionDecisionConfig:
    labor_rate: float       # cost per hour of maintenance
    risk_penalty: float     # cost per risk
    fail_cost: float        # cost of spindle overheat (fail)

    slowdown_lost_cost: float   # operational cost of slowing down the spindle
    slowdown_risk_factor: float # α: fraction of risk remaining after slowing down

    @classmethod
    def from_yaml(cls, path: str) -> "ActionDecisionConfig":
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        data = raw.get("decision", {})
        return cls(**data)
