# --- Helper functions

def round_dict(d, ndigits=4):
    rounded = {}
    for k, v in d.items():
        if isinstance(v, float):
            rounded[k] = round(float(v), ndigits)
        elif isinstance(v, dict):
            rounded[k] = round_dict(v, ndigits=ndigits)
        else:
            rounded[k] = v
    return rounded
