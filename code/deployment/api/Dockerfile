FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

COPY code/deployment/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api/main.py ./main.py
COPY models/model.joblib ./models/model.joblib
COPY models/model_metadata.json ./models/model_metadata.json

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

