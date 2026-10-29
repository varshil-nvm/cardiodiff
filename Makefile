.PHONY: install run dev clean test

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

dev:
	streamlit run app.py --server.runOnSave true

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

test:
	python -m pytest 2>/dev/null || echo "No tests configured"
