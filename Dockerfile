FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/app/.cache/huggingface TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements-app.txt
RUN useradd --create-home --uid 1000 app
COPY app.py app_backend.py app_workspaces.py app_storage.py app_access.py llm_provider.py paper_ingestion.py grounded_answer.py rag_evaluation.py evaluation_reference_labels.json paper_titles.json ./
RUN mkdir -p /app/evaluation_results /app/chroma_db && chown -R app:app /app
COPY .streamlit/config.toml .streamlit/config.toml
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
