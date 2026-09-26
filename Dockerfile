FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OUTPUT_DIR=/app/output

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analysis.py heart.csv pytest.ini ./
COPY tests ./tests

CMD ["python", "analysis.py"]
