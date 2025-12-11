import requests


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


def call_ollama(prompt: str) -> str:
    """
    Call the local Ollama HTTP API and return the generated text.
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False, # get one full JSON response
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return ""
