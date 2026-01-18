import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from app.utils import repo_path
from bn.diagnose_bn import DiagnoseBN
from bn.diagnose_bn_config import DiagnoseBNConfig
from decision.action_decision import ActionDecision
from decision.action_decision_config import ActionDecisionConfig
from kg.kg_client import KGClient
from kg.kg_client_config import KGClientConfig


def main():
    raw_data = {
        "vibration_rms": 1.177,
        "spindle_temp": 72.04,
        "ambient_temp": 21.72,
        "coolant_flow": 0.384,
        "spindle_speed": 2800,
    }

    # BN
    config = DiagnoseBNConfig.from_yaml(repo_path("configs", "bn.yaml"))
    # model = DiagnoseBN(config, path=repo_path("weights", "diagnose_bn_model.pkl"))
    model = DiagnoseBN(config)
    model.train(True)
    print("Model trained.\n")

    result_mle = model.diagnose_from_raw(raw_data, method="mle")
    print("MLE diagnosis:", json.dumps(result_mle, indent=2))

    result_em = model.diagnose_from_raw(raw_data, method="em")
    print("EM diagnosis:", json.dumps(result_em, indent=2))

    # KG
    kg_cfg = KGClientConfig.from_yaml(repo_path("configs", "kg.yaml"))
    kg_client = KGClient(kg_cfg)

    # Action
    decision_cfg = ActionDecisionConfig.from_yaml(repo_path("configs", "decision.yaml"))
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


def evaluate():

    def action_accuracy(y_true, y_pred):
        return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    
    def action_confusion(y_true, y_pred):

        labels = [
            "continue",
            "SlowDown",
            "CleanFilter",
            "FlushCoolant",
            "RepairFan",
            "ReplaceBearing",
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels= labels
        )
        disp.plot(
            cmap="PuBu",
            xticks_rotation=45,
            ax=ax
        )

        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def total_cost(y_pred, cost_map):
        return sum(cost_map[p] for p in y_pred)

    def mean_cost(y_pred, cost_map):
        return total_cost(y_pred, cost_map) / len(y_pred)
    
    def regret(y_true, y_pred, cost_map):
        regrets = []
        for t, p in zip(y_true, y_pred):
            optimal_cost = cost_map[t]
            chosen_cost = cost_map[p]
            regrets.append(chosen_cost - optimal_cost)
        return regrets
    
    def mean_regret(y_true, y_pred, cost_map):
        """
        The mean_regret metrics calculates the mean difference between, the choosen action and the optimal action (ground truth), ideal is 0 bigger
        values mean that the system is using actions that are more costly, lower values meand the model is choosing lower  
        
        :param y_true: Ground truth 
        :param y_pred: Predictions of the decision system
        :param cost_map: Maps each action to its cost
        """
        r = regret(y_true, y_pred, cost_map)
        return sum(r) / len(r)
    
    def severity_error(y_true, y_pred, severity_map):
        """
        Calculates the absolute difference in risk between the choosen action and the optimal action, bigger vaules represent worse errors
        
        :param y_true: Ground truth
        :param y_pred: Predictions of the decision system
        :param severity_map: Maps each action to its risk
        """
        sev = [abs(severity_map[p] - severity_map[t]) for t, p in zip(y_true, y_pred) if p!=t]


        if len(sev)==0:
            return 0.0
        else:
            return sum(sev)/len(sev)
    
    def continue_rate(y_pred):
        """
        Calculates the percentage the predictions that are the action "continue"
        
        :param y_pred: Predictions of the decision system
        """
        return y_pred.count("continue") / len(y_pred)
    
    def false_continue_rate(y_true, y_pred):
        """
        Calculates the percentage of the predictions of the action "continue" that incorrect
        
        :param y_true: Ground truth
        :param y_pred: Predictions of the decision system
        """
        errors = [ 1 for t, p in zip(y_true, y_pred) if p == "continue" and t != "continue"]

        if y_pred.count("continue") == 0:
            return 0
        else:
            return sum(errors) / y_pred.count("continue")


    procedures_df = pd.read_csv("./../data/actions/procedures_man.csv", parse_dates=["procedure_timestamp","selected_sensor_timestamp"])

    # BN
    config = DiagnoseBNConfig.from_yaml(repo_path("configs", "bn.yaml"))
    model = DiagnoseBN(config, path=repo_path("weights", "diagnose_bn_model.pkl"))

    # KG
    kg_cfg = KGClientConfig.from_yaml(repo_path("configs", "kg.yaml"))
    kg_client = KGClient(kg_cfg)

    # Action
    decision_cfg = ActionDecisionConfig.from_yaml(repo_path("configs", "decision.yaml"))
    action_decision = ActionDecision(kg_client, decision_cfg)
    
    model_actions = []

    for _, row in procedures_df.iterrows():

        raw_data = {
            "high_vibration": row["high_vibration"],
            "high_spindle_temp": row["high_spindle_temp"],
            "low_coolant_flow": row["low_coolant_flow"],
            "high_spindle_speed": row["high_spindle_speed"],
        }

        result_em = model.diagnose(raw_data, method="em")

        decision = action_decision.decide(result_em)

        if decision["best_action"]["action"] != "continue":
            proc_id = decision["best_action"]["proc_id"]
            model_actions.append(proc_id)
        else:
            model_actions.append("continue")
    
    
    y_true = procedures_df["proc_id"].tolist()

    ACTION_COST = {
        "ReplaceBearing": 600,
        "RepairFan": 150,
        "CleanFilter": 50,
        "FlushCoolant": 90,
        "SlowDown": 200,
        "continue": 0,
    }

    SEVERITY = {
        "continue": 0,
        "SlowDown": 0.5,
        "CleanFilter": 1,
        "FlushCoolant": 2,
        "RepairFan": 2,
        "ReplaceBearing": 3,
    }

    print("Accuracy:", action_accuracy(y_true, model_actions))
    print("Mean cost:", mean_cost(model_actions, ACTION_COST))
    print("Mean regret:", mean_regret(y_true, model_actions, ACTION_COST))
    print("Continue rate:", continue_rate(model_actions))
    print("False continue rate:", false_continue_rate(y_true, model_actions))
    print("Mean severity error:", severity_error(y_true, model_actions, SEVERITY))

    action_confusion(y_true=y_true, y_pred=model_actions)


if __name__ == "__main__":
    main()
    # print("-----------------------------Evaluation of the decision system-----------------------------")
    evaluate()
