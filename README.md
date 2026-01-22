# TSI-IndustrialEquipmentDiagnosis

## Entry points
- `src/main.py` is a lightweight end-to-end pipeline test (diagnosis + KG lookup + decision logic) intended for quick sanity checks.
- The Streamlit UI lives in `src/machine_diagnosis.py`.

## How to run

### Prerequisites
- Python 3.10+ (use a virtual environment)

### Setup

macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows
```powershell
python -m venv venv
call venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Run app

From `src/` folder run:
* `streamlit run machine_diagnosis.py`

Ollama setup:
* Just once: `sudo docker exec -it ollama14 ollama pull llama3.2`
* Start server: `sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama14 ollama/ollama`
* Check if it's alive: `curl http://localhost:11434/api/generate -d '{"model": "llama3.2", "prompt": "Say hello from Ollama", "stream": false}'`

## Project Structure

- `configs/` - YAML configs for each module (BN, KG client, decision policy).
- `data/` - datasets and KG assets:
	- `data/raw/` - source CSVs (telemetry, labels, relations, maintenance, etc.).
	- `data/processed/` - created labels
	- `data/knowledgeGraph/` - the Turtle KG file (`knowledgeGraph.ttl`).
	- `data/actions/` - procedure/action tables used to evaluate decisions.
- `notebooks/` - data preparation, KG construction, and evaluation notebooks.
- `src/` - application code:
	- `src/app/` - Streamlit UI components + explanation/LLM helpers.
	- `src/bn/` - Bayesian Network training and inference.
	- `src/kg/` - Knowledge Graph client and queries.
	- `src/decision/` - Action Decision logic (cost-based ranking).
- `weights/` - trained BN model pickle files.
