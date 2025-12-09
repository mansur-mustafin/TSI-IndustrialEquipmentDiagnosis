import json

from bn.diagnose_bn_config import DiagnoseBNConfig
from bn.diagnose_bn import DiagnoseBN

def main():
    config = DiagnoseBNConfig.from_yaml("./../configs/bn.yaml")
    model = DiagnoseBN(config)
    model.load_data()
    model.train()

    print("Model trained.\n\n")

    raw_data = {
        "vibration_rms": 0.5,
        "spindle_temp": 80.0,
        "ambient_temp": 20.0,
        "coolant_flow": 10.0,
    }

    result_mle = model.diagnose(raw_data, method="mle")
    print("MLE diagnosis:", json.dumps(result_mle, indent=2))

    result_em = model.diagnose(raw_data, method="em")
    print("EM diagnosis:", json.dumps(result_em, indent=2))

if __name__ == "__main__":
    main()
