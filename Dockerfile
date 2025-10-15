FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y freetds-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_engine.py geo_engine.py app.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
