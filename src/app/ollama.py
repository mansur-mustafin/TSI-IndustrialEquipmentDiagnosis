import requests


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
                "stream": False,  # get one full JSON response
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return ""
