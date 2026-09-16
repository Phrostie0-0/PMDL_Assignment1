FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY code/deployment/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app/app.py ./app.py

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=2)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

