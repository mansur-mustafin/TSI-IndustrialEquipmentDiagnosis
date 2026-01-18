from typing import Any

from bn.diagnose_bn import DiagnoseBN
from decision.action_decision_config import ActionDecisionConfig
from kg.kg_client import KGClient


class ActionDecision:

    def __init__(self, kg_client: KGClient, bn: DiagnoseBN, decision_config: ActionDecisionConfig):
        self.kg = kg_client
        self.cfg = decision_config
        self.bn = bn

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

        actions = sorted(actions, key=lambda a: a["total_cost"])
        best = min(actions, key=lambda a: a["total_cost"])

        return {
            "overheat_prob": overheat_prob,
            "cause_probs": cause_probs,
            "actions": actions,
            "best_action": best,
        }

    def decide_with_post_evidence(self, diagnosis: dict[str, float]) -> dict[str, Any]:
        """
        This method will be not used, kept for reference.
        """
        if diagnosis.get("method") != "em":
            raise ValueError("Diagnosis must be provided with 'em' method.")

        evidence = diagnosis["evidence"]
        overheat_prob = float(diagnosis.get("overheat_prob"))
        cause_probs = diagnosis.get("cause_probs")

        # Baseline: "do nothing"
        actions: list[dict[str, Any]] = [{
            "action": "continue",
            "total_cost": overheat_prob * self.cfg.fail_cost,
            "post_overheat_prob": overheat_prob,
        }]

        for cause, prob in cause_probs.items():
            procedures = self.kg.get_procedures_for_cause(cause)

            for proc in procedures:

                if proc["proc_name"] == "SlowDown":
                    modified = dict(evidence)
                    modified["high_spindle_speed"] = 0
                    
                    post_risk = self.bn.overheat_prob(modified, method="em")
                    proc_cost = float(self.cfg.slowdown_lost_cost)
                    total_cost = proc_cost + post_risk * self.cfg.fail_cost

                    actions.append({
                        "action": "slow_down",
                        "proc_id": proc["proc_id"],
                        "proc_name": proc["proc_name"],
                        "cause": cause,
                        "p_cause": float(prob),
                        "proc_cost": proc_cost,
                        "post_overheat_prob": float(post_risk),
                        "total_cost": float(total_cost),
                        "target_component": proc["target_component"],
                    })
                    continue

                
                proc_cost = self._cost(proc)
                post_evidence = self._post_evidence_after_fix(evidence, cause)
                post_risk = self.bn.overheat_prob(post_evidence, method="em")
                total_cost = proc_cost + post_risk * self.cfg.fail_cost

                actions.append({
                    "action": "schedule",
                    "proc_id": proc["proc_id"],
                    "proc_name": proc["proc_name"],
                    "cause": cause,
                    "p_cause": float(prob),
                    "proc_cost": float(proc_cost),
                    "post_overheat_prob": float(post_risk),
                    "total_cost": float(total_cost),
                    "target_component": proc["target_component"],
                })

        actions = sorted(actions, key=lambda a: a["total_cost"])
        best = actions[0]

        return {
            "overheat_prob": overheat_prob,
            "cause_probs": cause_probs,
            "actions": actions,
            "best_action": best,
        }

    def _post_evidence_after_fix(self, evidence: dict[str, int], cause: str) -> dict[str, int]:
        KG_SYMPTOM_TO_BN = {
            "HighVibration": "high_vibration",
            "HighSpindleTemp": "high_spindle_temp",
            "LowCoolantFlow": "low_coolant_flow",
            "HighSpindleSpeed": "high_spindle_speed",
        }
        
        post = dict(evidence)

        ctx = self.kg.get_cause_context(cause)   # uses causesSymptom in KG
        for kg_sym in ctx.get("symptoms", []):
            bn_sym = KG_SYMPTOM_TO_BN.get(kg_sym)
            if bn_sym is not None:
                post[bn_sym] = 0

        # force the latent to 0
        post[cause] = 0

        return post
