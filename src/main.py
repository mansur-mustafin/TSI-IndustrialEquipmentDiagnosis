import json

from bn.diagnose_bn_config import DiagnoseBNConfig
from bn.diagnose_bn import DiagnoseBN
from kg.kg_client import KGClient, KGClientConfig

def main():
    config = DiagnoseBNConfig.from_yaml("./../configs/bn.yaml")
    model = DiagnoseBN(config)
    model.load_data()
    model.train()

    print("Model trained.\n\n")

    raw_data = {
        "vibration_rms": 0.893,
        "spindle_temp": 58.26,
        "ambient_temp": 19.73,
        "coolant_flow": 0.612,
    }

    result_mle = model.diagnose(raw_data, method="mle")
    print("MLE diagnosis:", json.dumps(result_mle, indent=2))

    result_em = model.diagnose(raw_data, method="em")
    print("EM diagnosis:", json.dumps(result_em, indent=2))


    # KG
    kg_cfg = KGClientConfig(
        ttl_path="../data/knowledge_graph.ttl",
    )
    kg_client = KGClient(kg_cfg)

    cause_probs = result_em["cause_probs"]

    for bn_cause, p_c in sorted(cause_probs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"--- Cause: {bn_cause}  (P = {p_c:.4f})")

        cause_ctx = kg_client.get_cause_context(bn_cause)
        print("Cause context:")
        print(json.dumps(cause_ctx, indent=2))

        procs = kg_client.get_procedures_for_bn_cause(bn_cause)
        print("\nProcedures for this cause:")
        print(json.dumps(procs, indent=2))

        for proc in procs:
            # proc_uri will look like 'http://example.org/cnc#P1' → extract 'P1'
            proc_uri = proc["proc_uri"]
            proc_id = proc_uri.split("#")[-1]

            proc_expl = kg_client.get_procedure_explanation(proc_id)
            print(f"\nProcedure explanation for {proc_id}:")
            print(json.dumps(proc_expl, indent=2))

        print()


if __name__ == "__main__":
    main()
