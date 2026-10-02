# Build context is ./agents so the shared agent_core package can be copied in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

COPY core/requirements.txt .
RUN pip install -r requirements.txt

COPY core/agent_core ./agent_core
COPY agent-websocket/ws_app.py .

RUN useradd --system --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "ws_app:app", "--host", "0.0.0.0", "--port", "8000"]
