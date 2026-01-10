from typing import Any

from decision.action_decision_config import ActionDecisionConfig
from kg.kg_client import KGClient


class ActionDecision:

    def __init__(self, kg_client: KGClient, decision_config: ActionDecisionConfig):
        self.kg = kg_client
        self.cfg = decision_config

    def _cost(self, proc: dict[str, Any]) -> float:
        labor = proc["effort_h"] * self.cfg.labor_rate
        parts = proc["spare_parts_cost_eur"]
        risk_cost = proc["risk_rating"] * self.cfg.risk_penalty
        return float(labor + parts + risk_cost)

    def decide(self, diagnosis: dict[str, float]) -> dict[str, Any]:
        """
        Docstring for decide
        
        :param self: Description
        :param diagnosis: Description
        :type diagnosis: dict[str, float]
        :return: Description
        :rtype: dict[str, Any]
        """
        if diagnosis.get("method") != "em":
            raise ValueError("Diagnosis must be provided with 'em' method.")

        overheat_prob = float(diagnosis.get("overheat_prob"))
        cause_probs = diagnosis.get("cause_probs")

        actions: list[dict[str, Any]] = [{
            "action": "continue",
            "total_cost": overheat_prob * self.cfg.fail_cost,
        }]

        for cause, prob in cause_probs.items():
            procedures = self.kg.get_procedures_for_cause(cause)

            for proc in procedures:

                if proc["proc_name"] == "SlowDown":
                    action = "slow_down"
                    proc_cost = self.cfg.slowdown_lost_cost
                    alpha = self.cfg.slowdown_risk_factor
                    total_cost = proc_cost + alpha * overheat_prob * self.cfg.fail_cost
                else:
                    action = "schedule"
                    proc_cost = self._cost(proc)
                    # cost + (1 - P(cause | evidence)) * P(overheat | evidence) * fail_cost
                    total_cost = proc_cost + (1 - prob) * overheat_prob * self.cfg.fail_cost

                actions.append({
                    "action": action,
                    "proc_id": proc["proc_id"],
                    "proc_name": proc["proc_name"],
                    "cause": cause,
                    "p_cause": prob,
                    "proc_cost": proc_cost,
                    "total_cost": total_cost,
                    "target_component": proc["target_component"],
                })

        best = min(actions, key=lambda a: a["total_cost"])

        return {
            "overheat_prob": overheat_prob,
            "cause_probs": cause_probs,
            "actions": actions,
            "best_action": best,
        }
