.PHONY: setup pipeline bronze silver gold app clean

PYTHON := .venv/bin/python

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "\n✓  Environment ready. Copy .env.example → .env and add your ANTHROPIC_API_KEY."

bronze:
	$(PYTHON) pipeline/01_bronze_ingest.py

silver:
	$(PYTHON) pipeline/02_silver_transform.py

gold:
	$(PYTHON) pipeline/03_gold_generate.py

pipeline: bronze silver gold

app:
	.venv/bin/streamlit run app/app.py

clean:
	rm -rf data/bronze data/silver data/gold
	mkdir -p data/bronze data/silver data/gold
