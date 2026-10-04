FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/var/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.runtime.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && python -m pip install -r requirements.runtime.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN test -f artifacts/affinity.joblib \
    && test -f artifacts/toxicity.joblib \
    && mkdir -p "${HF_HOME}" \
    && python -m py_compile api.py service.py generator.py filters.py models.py optimizer.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
