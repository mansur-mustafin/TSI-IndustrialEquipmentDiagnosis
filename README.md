# TSI-IndustrialEquipmentFailureDiagnosis

## How to run
Streamlit app:
* `streamlit run app.py`

Ollama setup:
* Just once: `sudo docker exec -it ollama14 ollama pull llama3.2`
* Start server: `sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama14 ollama/ollama`
* Check if it's alive: `curl http://localhost:11434/api/generate -d '{"model": "llama3.2", "prompt": "Say hello from Ollama", "stream": false}'`
