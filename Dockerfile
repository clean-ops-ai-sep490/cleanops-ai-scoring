FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG REQUIREMENTS_FILE=requirements.inference.txt

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY requirements.inference.txt ./
COPY requirements.training.txt ./
RUN pip install --no-cache-dir --default-timeout=1200 --retries 10 -r ${REQUIREMENTS_FILE}

COPY src ./src
RUN mkdir -p /app/model-cache /app/outputs

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
