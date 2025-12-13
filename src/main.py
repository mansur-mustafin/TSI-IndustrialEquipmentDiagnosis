import json

from bn.diagnose_bn import DiagnoseBN
from bn.diagnose_bn_config import DiagnoseBNConfig
from decision.action_decision import ActionDecision
from decision.decision_config import DecisionConfig
from kg.kg_client import KGClient
from kg.kg_client_config import KGClientConfig


def main():
    raw_data = {
        "vibration_rms": 0.893,
        "spindle_temp": 98.26,
        "ambient_temp": 19.73,
        "coolant_flow": 0.612,
    }

    # BN
    config = DiagnoseBNConfig.from_yaml("./../configs/bn.yaml")
    model = DiagnoseBN(config)
    model.train()
    print("Model trained.\n")

    result_mle = model.diagnose(raw_data, method="mle")
    print("MLE diagnosis:", json.dumps(result_mle, indent=2))

    result_em = model.diagnose(raw_data, method="em")
    print("EM diagnosis:", json.dumps(result_em, indent=2))

    # KG
    kg_cfg = KGClientConfig.from_yaml("./../configs/kg.yaml")
    kg_client = KGClient(kg_cfg)

    # Action
    decision_cfg = DecisionConfig.from_yaml("./../configs/decision.yaml")
    action_decision = ActionDecision(kg_client, decision_cfg)

    decision = action_decision.decide(result_em)
    print("Decision:", json.dumps(decision, indent=2))

    if decision["best_action"]["action"] != "continue":
        proc_id = decision["best_action"]["proc_id"]
        procedure = kg_client.get_procedure_explanation(proc_id)
        print("Scheduled procedure details:", json.dumps(procedure, indent=2))

        cause = decision["best_action"]["cause"]
        cause_context = kg_client.get_cause_context(cause)
        print("Cause context:", json.dumps(cause_context, indent=2))


if __name__ == "__main__":
    main()
